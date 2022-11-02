import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.vgg import *
from torchvision.models.vgg import model_urls, cfgs
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url
from PIL import Image
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from skimage.segmentation import mark_boundaries
import cv2
import time
import argparse
from lib.dataset import bsds, augmentation
from lib.utils.meter import Meter
from lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse
from model import SSNModel

relu_layers = (1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29)
pooling_layers = (4, 9, 16, 23, 30)

@torch.no_grad()
def eval(model, loader, color_scale, pos_scale, device):
    def achievable_segmentation_accuracy(superpixel, label):
        """
        Function to calculate Achievable Segmentation Accuracy:
            ASA(S,G) = sum_j max_i |s_j \cap g_i| / sum_i |g_i|
        Args:
            input: superpixel image (H, W),
            output: ground-truth (H, W)
        """
        TP = 0
        unique_id = np.unique(superpixel)
        for uid in unique_id:
            mask = superpixel == uid
            label_hist = np.histogram(label[mask])
            maximum_regionsize = label_hist[0].max()
            TP += maximum_regionsize
        return TP / label.size

    model.eval()
    sum_asa = 0
    for data in loader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        height, width = inputs.shape[-2:]

        nspix_per_axis = int(math.sqrt(model.nspix))
        pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)

        coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), 0)
        coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

        inputs = torch.cat([color_scale*inputs, pos_scale*coords], 1)

        Q, H, feat = model(inputs)

        H = H.reshape(height, width)
        labels = labels.argmax(1).reshape(height, width)

        asa = achievable_segmentation_accuracy(H.to("cpu").detach().numpy(), labels.to("cpu").numpy())
        sum_asa += asa
    model.train()
    return sum_asa / len(loader)

def custom_forward(model, x):
    num_children_modules = len(list(model.features.children()))
    output = {}
    for i in range(num_children_modules):
        x = model.features[i](x)
        if i in relu_layers or i in pooling_layers:
            output[f'layer_{i}'] = x
    return output

def update_params(data, model, optimizer, compactness, color_scale, pos_scale, device):
    inputs, labels = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    height, width = inputs.shape[-2:]
    print(f"height: {height}")

    nspix_per_axis = int(math.sqrt(model.nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)
    coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), 0)
    coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

    inputs = torch.cat([color_scale * inputs, pos_scale * coords], 1)

    Q, H, feat = model(inputs)

    recons_loss = reconstruct_loss_with_cross_etnropy(Q, labels)
    compact_loss = reconstruct_loss_with_mse(Q, coords.reshape(*coords.shape[:2], -1), H)

    loss = recons_loss + compactness * compact_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item()}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Path to dataset")
    parser.add_argument("--batchsize", default=8, type=int, help="Batch size")
    parser.add_argument("--train_iter", default=15000, type=int)
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for optimizer")
    parser.add_argument("--compactness", default=1e-5, type=float)
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--test_interval", default=10000, type=int)
    parser.add_argument("--out_dir", default='./checkpoints', type=str)
    parser.add_argument("--f_dim", default=20, type=int, help="Embedding dimension")
    parser.add_argument("--n_spix", default=100, type=int, help='Number of superpixels')
    parser.add_argument("--n_iter", default=5, type=int, help='Number of iterations of differentiable SLIC')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    # train_transforms = T.Compose([T.RandomHorizontalFlip(0.5), T.RandomCrop((200, 200))])
    train_transforms = augmentation.Compose([augmentation.RandomHorizontalFlip(), augmentation.RandomScale(), augmentation.RandomCrop()])
    train_dataset = bsds.BSDS(args.root, geo_transforms=train_transforms)
    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True)

    test_dataset = bsds.BSDS(args.root, split='val')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f'Number of training batches: {len(train_loader)}')
    print(f'Number of testing samples: {len(test_loader)}')

    # Initialize Model
    model = SSNModel(args.f_dim, args.n_spix, args.n_iter).to(device)
    print(f'Model: {model}')

    # Set Optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)

    meter = Meter()
    iterations = 0
    max_val_asa = 0


    # Train model
    while iterations < args.train_iter:
        for data in train_loader:
            iterations += 1
            metric = update_params(data, model, optimizer, args.compactness, args.color_scale, args.pos_scale, device)
            meter.add(metric)
            state = meter.state(f'[{iterations}/{args.train_iter}]')
            print(state)
            if (iterations % args.test_interval) == 0:
                asa = eval(model, test_loader, args.color_scale, args.pos_scale, device)
                print(f'Validation ASA: {asa}')
                if asa > max_val_asa:
                    max_val_asa = asa
                    torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))
            if iterations == args.train_iter:
                break
        unique_id = str(int(time.time()))
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_' + unique_id + '.pth'))





    # model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # model.to(device)
    # input_size = (256, 256)
    # val_transform = T.Compose([
    #     T.Resize(input_size),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225]),
    # ])
    # img_path = os.path.join('/mnt/nfs-students/fine_grained_segmentation/Sooty_Albatross_0031_1066.jpg')
    # img = Image.open(img_path)
    # img = val_transform(img).unsqueeze(0)
    # img = img.to(device)
    #
    # features = custom_forward(model, img)
    #
    # # Set Parameters
    # nspix = 25
    # pos_scale = 0.25
    # enforce_connectivity = True
    #
    # # Inference
    # model = SSNModel(nspix, training=True)
    #
    # for label, feature_map in features.items():
    #     height = width = feature_map.shape[-1]
    #     print(f'height: {height}, width: {width}')
    #     image = Image.open('/mnt/nfs-students/fine_grained_segmentation/Sooty_Albatross_0031_1066.jpg')
    #     image = image.resize((height, width))
    #     image = np.array(image)
    #     # plt.imsave(f'resized_{label}.png', image)
    #     _, H, _ = model(feature_map)
    #     labels = H.reshape(height, width).to("cpu").detach().numpy()
    #
    #     if enforce_connectivity:
    #         segment_size = height * width / nspix
    #         min_size = int(0.06 * segment_size)
    #         max_size = int(3.0 * segment_size)
    #         labels = _enforce_label_connectivity_cython(
    #             labels[None], min_size, max_size)[0]
    #
    #     plt.imsave(f'vgg_results/result_{label}.png', mark_boundaries(image, labels))

    # for key, value in features.items():
    #     print(f'{key}: {value.shape}')
