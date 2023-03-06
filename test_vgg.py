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
from torchvision.models import vgg16
from torchvision.models.vgg import *
from torchvision.models.vgg import model_urls, cfgs
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url
from PIL import Image
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from skimage.segmentation import mark_boundaries
import cv2
import wandb
import time
import argparse
from lib.dataset import bsds, augmentation
from lib.utils.meter import Meter
from lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse
from model import SSNModel, SSN_VGG, MLP
from graph import Graph, Superpixel

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
        pos_scale = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)

        coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), 0)
        coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

        inputs = torch.cat([color_scale * inputs, pos_scale * coords], 1)

        Q, H, superpixel_features, num_spixels_width = model(inputs)

        # print('Q: ', Q.shape)
        # print('H: ', H.shape)
        # print('superpixel features: ', superpixel_features.shape)
        # print('extracted features: ', extracted_features.shape)

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


def update_params(data, model, optimizer, compactness, color_scale, pos_scale, device, mlp_model, batch_size, iteration):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    height, width = inputs.shape[-2:]

    nspix_per_axis = int(math.sqrt(model.nspix))
    pos_scale = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)
    coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), 0)
    coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

    inputs = torch.cat([color_scale * inputs, pos_scale * coords], 1)

    Q, H, superpixel_features, num_spixels_width = model(inputs)

    # Extract superpixel data
    superpixels_list = []
    reshaped_labels = H.reshape(-1, 200, 200)
    # print(f'Labels shape: {H.shape}, reshaped labels shape: {reshaped_labels.shape}, num_spixels_width: {num_spixels_width}')
    edge_indices = []
    for i in range(batch_size):
        # print(f'Sample {i} in batch:')
        mylist = []
        superpixels = {}
        for j in range(len(torch.unique(reshaped_labels))):
            pixel_indices_2d = torch.argwhere(reshaped_labels[i, :, :] == j)
            pixel_indices_1d = torch.argwhere(reshaped_labels[i, :, :].flatten() == j)
            superpixels[j] = Superpixel(index=j, features=torch.unsqueeze(superpixel_features[i, :, j], 0), pixel_indices_2d=pixel_indices_2d.double(),
                                        num_spixels_width=torch.tensor(num_spixels_width),
                                        image_width=width, num_spixels=torch.max(reshaped_labels),
                                        pixel_indices_1d=pixel_indices_1d.double())
        mylist.append(superpixels)
        edges = None
        for key in superpixels.keys():
            superpixel = superpixels[key]
            cds = superpixel.convert_spixel_index_to_coordinates()
            if edges is None:
                edges = cds
            else:
                edges = np.hstack([edges, cds])

        if(len(edge_indices) == 0):
            edge_indices = edges
        else:
            np.append(edge_indices, edges)

    superpixel_features = torch.transpose(torch.squeeze(superpixel_features), 0, 1)
    
    edge_indices =torch.tensor(edge_indices, dtype=torch.int64)
    edge_indices = edge_indices.to(device)

    # Compute original losses
    # Compute original recons loss
    recons_loss = reconstruct_loss_with_cross_etnropy(Q, labels)
    compact_loss = reconstruct_loss_with_mse(Q, coords.reshape(*coords.shape[:2], -1), H)

    # Setup MLP
    updated_spixel_features, prob_vector = mlp_model(x=superpixel_features, edge_index=edge_indices)
    edge_indices = torch.transpose(edge_indices, 0, 1)

    """
    components = {}
    val = 0
    for i in range(len(edge_indices)):
        # print(f'Checking {edge_indices[i][0]} and {edge_indices[i][1]}, p: {prob_vector[i]}')
        if (prob_vector[i] > 0.5):
            # print(f'Combining: {edge_indices[i][0]} and {edge_indices[i][1]}')
            if (int(edge_indices[i][0]) in components.keys()):
                components[int(edge_indices[i][1])] = components[int(edge_indices[i][0])]
            elif (int(edge_indices[i][1]) in components.keys()):
                components[int(edge_indices[i][0])] = components[int(edge_indices[i][1])]
            else:
                components[int(edge_indices[i][0])] = val
                components[int(edge_indices[i][1])] = val
                val += 1
        else:
            if (int(edge_indices[i][0]) not in components.keys()):
                components[int(edge_indices[i][0])] = val
                val += 1
            if (int(edge_indices[i][1]) not in components.keys()):
                components[int(edge_indices[i][1])] = val
                val += 1

    # print(f'Components: {components}')
    
    H_prime = H.detach().clone()
    for i in range(len(torch.unique(H))):
        if i in components.keys():
            H_prime[H_prime == i] = components[i]
    """

    # Compute Q_prime
    Q_prime = torch.zeros(Q.shape[1], Q.shape[1])
    for edge, p in zip(edge_indices, prob_vector):
        Q_prime[edge[0]][edge[1]] = p
    Q_prime = torch.unsqueeze(Q_prime, 0)
    Q_prime = Q_prime.to(device)

    # print(f"Edge Indices shape: {edge_indices.shape}, prob vector shape: {prob_vector.shape}, Q shape: {Q.shape}, Q_prime shape: {Q_prime.shape}")
    # loss = recons_loss + compactness * compact_loss + recons_loss_with_comb_prop
    
    Q_product = torch.einsum("ijk, ijj->ikj", Q, Q_prime)
    Q_product = torch.einsum("ijk->ikj", Q_product)
    Q_product = Q_product.to(device)

    recons_loss_with_Q_prime = reconstruct_loss_with_cross_etnropy(Q_product, labels)
    loss = recons_loss + compactness * compact_loss + recons_loss_with_Q_prime

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item(), "Q_prime_recons": recons_loss_with_Q_prime.item()}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Path to dataset")
    parser.add_argument("--batchsize", default=1, type=int, help="Batch size")
    parser.add_argument("--train_iter", default=50000, type=int)
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for optimizer")
    parser.add_argument("--compactness", default=1e-5, type=float)
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--test_interval", default=1000, type=int)
    parser.add_argument("--out_dir", default='./checkpoints', type=str)
    parser.add_argument("--f_dim", default=20, type=int, help="Embedding dimension")
    parser.add_argument("--n_spix", default=100, type=int, help='Number of superpixels')
    parser.add_argument("--n_iter", default=5, type=int, help='Number of iterations of differentiable SLIC')
    parser.add_argument("--layer_number", default=3, type=int, help='Number of layers of VGG backbone')
    parser.add_argument("--wandb_key", type=str, help="Your wandb key")
    parser.add_argument("--wandb_entity", type=str, help="Entity name")

    args = parser.parse_args()

    # Setup wandb
    wandb.login(key=args.wandb_key)
    wandb.init(project='fine-grained_segmentation', entity=args.wandb_entity)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    # train_transforms = T.Compose([T.RandomHorizontalFlip(0.5), T.RandomCrop((200, 200))])
    train_transforms = augmentation.Compose(
        [augmentation.RandomHorizontalFlip(), augmentation.RandomScale(), augmentation.RandomCrop()])
    train_dataset = bsds.BSDS(args.root, geo_transforms=train_transforms)
    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True)

    test_dataset = bsds.BSDS(args.root, split='val')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f'Number of training batches: {len(train_loader)}')
    print(f'Number of testing samples: {len(test_loader)}')

    # Initialize Model
    model = SSNModel(feature_dim=args.f_dim, nspix=args.n_spix, training=True, n_iter=args.n_iter).to(device)
    mlp_model = MLP().to(device)

    # model = SSN_VGG(args.layer_number, args.n_spix, args.n_iter).to(device)
    print(f'Model: {model}')

    # Set Optimizer
    optimizer = optim.Adam(list(model.parameters()) + list(mlp_model.parameters()), args.lr)
    # optimizer = optim.Adam(model.parameters(), args.lr)

    meter = Meter()
    iterations = 0
    max_val_asa = 0


    # Train model
    while iterations < args.train_iter:
        for data in train_loader:
            iterations += 1
            metric = update_params(data, model, optimizer, args.compactness, args.color_scale, args.pos_scale, device, mlp_model, batch_size=args.batchsize, iteration=iterations)
            meter.add(metric)
            state = meter.state(f'[{iterations}/{args.train_iter}]')
            print(state)
            wandb.log(metric)
            # wandb.watch(model, log="all", idx=0)
            # wandb.watch(mlp_model, log="all", idx=1)
            wandb.watch([model, mlp_model], log='all')

            if (iterations % args.test_interval) == 0:
                asa = eval(model, test_loader, args.color_scale, args.pos_scale, device)
                print(f'Validation ASA: {asa}')
                if asa > max_val_asa:
                    max_val_asa = asa
                    torch.save(model.state_dict(),
                               os.path.join(args.out_dir, 'best_model_.pth'))
                    torch.save(mlp_model.state_dict(), os.path.join(args.out_dir, 'best_mlp_50.pth'))
            if iterations == args.train_iter:
                break
        unique_id = str(int(time.time()))
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_' + unique_id + '.pth'))
        torch.save(mlp_model.state_dict(), os.path.join(args.out_dir, 'mlp_' + unique_id + '.pth'))
