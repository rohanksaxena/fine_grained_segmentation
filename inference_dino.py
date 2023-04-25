import argparse
import math
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import dino
import torch.nn as nn
import torch.nn.functional as F
from graph import Superpixel
from model import SSNModel
from PIL import Image
from skimage.color import rgb2lab
from skimage.segmentation import mark_boundaries
from torchvision import transforms as pth_transforms
from dino import utils
from dino import vision_transformer as vits
from skimage.segmentation._slic import _enforce_label_connectivity_cython


def inference(image, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None):
    model = SSNModel(fdim, nspix, n_iter).to(device)
    model.load_state_dict(torch.load(weight))
    model.eval()

    height, width = image.shape[:2]

    # SSN Inference
    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)
    coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), 0)
    coords = coords[None].float()
    image = rgb2lab(image)
    image = torch.from_numpy(image).permute(2, 0, 1)[None].to(device).float()
    inputs = torch.cat([color_scale * image, pos_scale * coords], 1)
    Q, H, superpixel_features, num_spixels_width = model(inputs)

    labels = H.reshape(height, width).cpu().detach().numpy()
    return Q, H, superpixel_features, num_spixels_width


def extract_superpixels(spixel_features, reshaped_labels, num_spixels_width, width):
    spixel_list = []
    spixel_dict = {}
    for j in range(len(torch.unique(reshaped_labels))):
        pixel_indices_2d = torch.argwhere(reshaped_labels[0, :, :] == j)
        pixel_indices_1d = torch.argwhere(reshaped_labels[0, :, :].flatten() == j)
        spixel_dict[j] = Superpixel(index=j, features=torch.unsqueeze(spixel_features[0, :, j], 0),
                                    pixel_indices_2d=pixel_indices_2d.double(),
                                    num_spixels_width=torch.tensor(num_spixels_width),
                                    image_width=width, num_spixels=torch.max(reshaped_labels),
                                    pixel_indices_1d=pixel_indices_1d.double())
        spixel_list.append(spixel_dict[j])
    return spixel_list, spixel_dict


def extract_edges(spixel_dict):
    edge_indices = []
    edges = None
    for key in superpixels.keys():
        superpixel = superpixels[key]
        cds = superpixel.convert_spixel_index_to_coordinates()
        if edges is None:
            edges = cds
        else:
            edges = np.hstack([edges, cds])

    if (len(edge_indices) == 0):
        edge_indices = edges
    else:
        np.append(edge_indices, edges)
    edge_indices = torch.tensor(edge_indices, dtype=torch.int64)
    edge_indices = torch.transpose(edge_indices, 0, 1).to("cuda")
    return edge_indices


def get_dino_model(arch, patch_size, device, pretrained_weights, checkpoint_key):
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    return model

def get_attention_maps(image_path, patch_size, threshold, model, height, width):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')

    transform = pth_transforms.Compose([
        pth_transforms.Resize((height, width)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_selfattention(img.to(device))
    # print(f"attentions from model: {attentions.shape}")

    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
            0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    # print(f"attentions 1: {attentions.shape}")
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), size=(height, width), mode="nearest")[
        0].cpu().numpy()
    # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
    #     0].cpu().numpy()
    print(f"attentions 2: {attentions.shape}")
    return attentions

def get_pairwise_spixel_similarities(edge_indices, spixel_dict):
    similarities = []
    for edge in edge_indices:
        idx_s1, idx_s2 = edge[0], edge[1]
        s1_feat, s2_feat = superpixels[idx_s1.item()].feat, superpixels[idx_s2.item()].feat
        s1_feat, s2_feat = torch.unsqueeze(s1_feat, 0), torch.unsqueeze(s2_feat, 0)
        s1_feat, s2_feat = F.normalize(s1_feat), F.normalize(s2_feat)
        similarities.append(torch.dot(torch.squeeze(s1_feat), torch.squeeze(s2_feat)))
    return similarities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Path to dataset")
    parser.add_argument("--dataset", type=str, help="Name of dataset")
    parser.add_argument("--weight", default="", type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=10, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--layer_number", default=3, type=int)
    parser.add_argument("--mlp_weight", type=str, help="path to pretrained mlp")
    parser.add_argument("--wandb_key", type=str, help="Your wandb key")
    parser.add_argument("--wandb_entity", type=str, help="Entity name")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
                        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", type=str,
                        help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
            obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img_list = []
    print(f"Dataset: {args.dataset}")
    if args.dataset == "cub":
        data_dir = f'data/CUB/CUB_200_2011/'
        output_dir = "results/attention/cub/"
        img_dir = osp.join(data_dir, 'images')
        for path, subdirs, _ in os.walk(img_dir):
            for subdir in subdirs:
                files = os.listdir(osp.join(path, subdir))
                for file in files:
                    img_list.append(osp.join(path, subdir, file))
    elif args.dataset == "bsds":
        data_dir = f'data/BSDS500/BSDS500/data/images/test/'
        output_dir = "results/attention/bsds/"
        for img in os.listdir(data_dir):
            if osp.basename(img).split(".")[-1] in ('jpg', 'png'):
                img_list.append(osp.join(data_dir, img))
            else:
                print(f"invalid image name: {osp.basename(img)}")
    else:
        img_list.append(args.image_path)
        output_dir = "results/"

    # print(f"Image list: {img_list}")


    # Get DINO model
    dino_model = get_dino_model(args.arch, args.patch_size, device, args.pretrained_weights, args.checkpoint_key)
    # print(f"dino model loaded")

    for img in img_list:
        # print(f"current image: {img}")
        image = plt.imread(img)
        height, width = image.shape[:2]
        # print(f"h: {height}, width: {width}")

        # Perform inference using SSN
        Q, H, superpixel_features, num_spixels_width = inference(image, args.nspix, args.niter, args.fdim,
                                                                 args.color_scale, args.pos_scale, args.weight)
        # print(f"ssn inference done")

        # Extract superpixel data
        reshaped_labels = H.reshape(-1, height, width)
        spixel_list, superpixels = extract_superpixels(superpixel_features, reshaped_labels, num_spixels_width, width)
        superpixel_features = torch.transpose(torch.squeeze(superpixel_features), 0, 1)
        # print(f"superpixel data extracted")

        # Extract edges
        edge_indices = extract_edges(superpixels)
        # print(f"Edges extracted")

        # GET ATTENTION MAPS FROM DINO
        attentions = get_attention_maps(img, args.patch_size, args.threshold, dino_model, height, width)
        # print(f"computed attention")

        # Add centroid coordinates to each superpixel
        for idx, spix in enumerate(spixel_list):
            spix.x = round(spix.centroid[1].item())
            spix.y = round(spix.centroid[0].item())
            print(f"{spix.x}, {spix.y}")
            spix.feat = torch.tensor(attentions[:, spix.y, spix.x])

        # Get pairwise similarities for superpixels
        similarities = get_pairwise_spixel_similarities(edge_indices, superpixels)
        # print(f"parirwise spixel similarities done")

        # Combining superpixels based on similarity
        thresholds = [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
        # thresholds = [0.95]
        for t in thresholds:
            # print(f"threshold t: {t}")
            components = {}
            val = 0
            for i in range(len(edge_indices)):
                # print(f"{edge_indices[i][0]}, {edge_indices[i][1]} : {probs[i]}")
                # print(f"Components: {components}")
                if (similarities[i] > t):
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

                H_prime = H.detach().clone()
                for i in range(len(torch.unique(H))):
                    if i in components.keys():
                        H_prime[H_prime == i] = components[i]
            H_prime = torch.unsqueeze(H_prime, 0)
            image = plt.imread(img)
            image_name, img_ext = osp.basename(img).split(".")
            labels = H_prime.reshape(height, width).to("cpu").detach().numpy()
            plt.imsave(osp.join(output_dir, image_name + f"_{int(t*100)}." + img_ext), mark_boundaries(image, labels))


