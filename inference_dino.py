import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from graph import Superpixel
from model import SSNModel
from skimage.color import rgb2lab
from skimage.segmentation import mark_boundaries
from dino import visualize_attention

def inference(image, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None):
    model = SSNModel(fdim, nspix, n_iter).to("cuda")
    model.load_state_dict(torch.load(weight))
    model.eval()

    height, width = image.shape[:2]

    # SSN Inference
    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)
    coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda")), 0)
    coords = coords[None].float()
    image = rgb2lab(image)
    image = torch.from_numpy(image).permute(2, 0, 1)[None].to("cuda").float()
    inputs = torch.cat([color_scale * image, pos_scale * coords], 1)
    Q, H, superpixel_features, num_spixels_width = model(inputs)

    labels = H.reshape(height, width).to("cpu").detach().numpy()
    image = plt.imread(args.image)
    # plt.imsave("dino.png", mark_boundaries(image, labels))
    return Q, H, superpixel_features, num_spixels_width

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Path to dataset")
    parser.add_argument("--image", default="data/BSDS500/BSDS500/data/images/train/22093.jpg", type=str,
                        help="path to image")
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
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
            obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    image = plt.imread(args.image)
    height, width = image.shape[:2]

    # Perform inference using SSN
    Q, H, superpixel_features, num_spixels_width = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)

    # Extract superpixel data
    superpixels_list = []
    reshaped_labels = H.reshape(-1, height, width)
    edge_indices = []
    mylist = []
    superpixels = {}
    i = 0
    for j in range(len(torch.unique(reshaped_labels))):
        pixel_indices_2d = torch.argwhere(reshaped_labels[i, :, :] == j)
        pixel_indices_1d = torch.argwhere(reshaped_labels[i, :, :].flatten() == j)
        superpixels[j] = Superpixel(index=j, features=torch.unsqueeze(superpixel_features[i, :, j], 0),
                                    pixel_indices_2d=pixel_indices_2d.double(),
                                    num_spixels_width=torch.tensor(num_spixels_width),
                                    image_width=width, num_spixels=torch.max(reshaped_labels),
                                    pixel_indices_1d=pixel_indices_1d.double())
    mylist.append(superpixels)

    # Extract Edges
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

    superpixel_features = torch.transpose(torch.squeeze(superpixel_features), 0, 1)

    edge_indices = torch.tensor(edge_indices, dtype=torch.int64)
    edge_indices = torch.transpose(edge_indices, 0, 1).to("cuda")

    # Plot centroids 
    # print(f"Superpixel list: {len(mylist)}")
    # for key, val in superpixels.items():
    #     print(val.centroid)

    # Get attention maps from dino
    

    # Check for simlarities
