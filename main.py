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
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from PIL import Image
from model.model import SSNModel
import cv2
from lib.dataset import bsds, augmentation
from lib.utils.meter import Meter
from lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from skimage.segmentation import mark_boundaries


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
        pos_scale = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)

        coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), 0)
        coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

        inputs = torch.cat([color_scale * inputs, pos_scale * coords], 1)

        Q, H, feat = model(inputs)

        H = H.reshape(height, width)
        labels = labels.argmax(1).reshape(height, width)

        asa = achievable_segmentation_accuracy(H.to("cpu").detach().numpy(), labels.to("cpu").numpy())
        sum_asa += asa
    model.train()
    return sum_asa / len(loader)


def update_param(data, model, optimizer, compactness, color_scale, pos_scale, device):
    inputs, labels = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    height, width = inputs.shape[-2:]

    nspix_per_axis = int(math.sqrt(model.n_spix))
    pos_scale = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)

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


def predict(model, device):
    input_size = (224, 224)
    img_path = os.path.join(os.getcwd(), 'Sooty_Albatross_0031_1066.jpg')
    val_transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    model.to(device)
    img = Image.open(img_path).convert('RGB')
    img = val_transform(img).unsqueeze(0)
    img = img.to(device)
    print(f'Image shape: {img.shape}')
    features = custom_forward(model, img)
    return features


def custom_forward(model, x):
    # See note [TorchScript super()]
    features = {}
    input_size = x.shape[-1]
    input = x
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    l1 = model.layer1(x)
    l2 = model.layer2(l1)
    l3 = model.layer3(l2)
    l4 = model.layer4(l3)

    # l1 = nn.functional.interpolate(l1, size=input_size, mode='bilinear', align_corners=False)
    # l2 = nn.functional.interpolate(l2, size=input_size, mode='bilinear', align_corners=False)
    # l3 = nn.functional.interpolate(l3, size=input_size, mode='bilinear', align_corners=False)
    # l4 = nn.functional.interpolate(l4, size=input_size, mode='bilinear', align_corners=False)
    #
    # cat_feat = torch.cat([input, l1, l2, l3, l4], 1)

    # x = model.avgpool(x)
    # x = torch.flatten(x, 1)
    # x = model.fc(x)

    features['layer1'] = l1
    features['layer2'] = l2
    features['layer3'] = l3
    features['layer4'] = l4
    return features


if __name__ == '__main__':
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    input_size = (224, 224)
    with torch.no_grad():
        model.eval()
        extracted_features = predict(model, device)
    print(f'Feature Extraction complete')

    # Load Data
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    augment = augmentation.Compose(
        [augmentation.RandomHorizontalFlip(), augmentation.RandomScale(), augmentation.RandomCrop()])
    train_dataset = bsds.BSDS('/mnt/nfs-students/fine_grained_segmentation/data/BSDS500', geo_transforms=augment)
    train_loader = DataLoader(train_dataset, 6, shuffle=True, drop_last=True, num_workers=4)

    test_dataset = bsds.BSDS('/mnt/nfs-students/fine_grained_segmentation/data/BSDS500', split="val")
    test_loader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)

    # Set Parameters
    nspix = 5
    pos_scale = 2.5
    enforce_connectivity = True

    # Inference
    model = SSNModel(nspix, training=True)

    # height, width = extracted_features.shape[-1], extracted_features.shape[-1]
    # print(f'height: {height}')
    # print(f'weight: {width}')

    # nspix_per_axis = int(math.sqrt(nspix))
    # pos_scale = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)
    #
    # coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda")), 0)
    # coords = coords[None].float()

    # image = rgb2lab(image)
    # image = torch.from_numpy(image).permute(2, 0, 1)[None].to("cuda").float()

    # inputs = torch.cat([color_scale * image, pos_scale * coords], 1)

    # _, H, _ = model(extracted_features)
    #
    # labels = H.reshape(height, width).to("cpu").detach().numpy()
    #
    # if enforce_connectivity:
    #     segment_size = height * width / nspix
    #     min_size = int(0.06 * segment_size)
    #     max_size = int(3.0 * segment_size)
    #     labels = _enforce_label_connectivity_cython(
    #         labels[None], min_size, max_size)[0]

    ###################

    for label, feature_map in extracted_features.items():
        height = width = feature_map.shape[-1]
        print(f'height: {height}, width: {width}')
        image = Image.open('/mnt/nfs-students/fine_grained_segmentation/Sooty_Albatross_0031_1066.jpg')
        image = image.resize((height, width))
        image = np.array(image)
        plt.imsave(f'resized_{label}.png', image)
        _, H, _ = model(feature_map)
        labels = H.reshape(height, width).to("cpu").detach().numpy()

        if enforce_connectivity:
            segment_size = height * width / nspix
            min_size = int(0.06 * segment_size)
            max_size = int(3.0 * segment_size)
            labels = _enforce_label_connectivity_cython(
                labels[None], min_size, max_size)[0]

        plt.imsave(f'result_{label}.png', mark_boundaries(image, labels))
    ##################
