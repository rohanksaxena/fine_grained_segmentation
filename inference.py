import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from lib.ssn.ssn import sparse_ssn_iter


@torch.no_grad()
def inference(image, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None, enforce_connectivity=True):
    """
    generate superpixels

    Args:
        image: numpy.ndarray
            An array of shape (h, w, c)
        nspix: int
            number of superpixels
        n_iter: int
            number of iterations
        fdim (optional): int
            feature dimension for supervised setting
        color_scale: float
            color channel factor
        pos_scale: float
            pixel coordinate factor
        weight: state_dict
            pretrained weight
        enforce_connectivity: bool
            if True, enforce superpixel connectivity in postprocessing

    Return:
        labels: numpy.ndarray
            An array of shape (h, w)
    """
    if weight is not None:
        from model import SSNModel, SSN_VGG
        model = SSNModel(fdim, nspix, n_iter).to("cuda")
        # model = SSN_VGG(args.layer_number, args.nspix, args.niter).to('cuda')
        model.load_state_dict(torch.load(weight))
        model.eval()
    else:
        model = lambda data: sparse_ssn_iter(data, nspix, n_iter)

    height, width = image.shape[:2]

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)

    coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda")), 0)
    coords = coords[None].float()

    image = rgb2lab(image)
    image = torch.from_numpy(image).permute(2, 0, 1)[None].to("cuda").float()

    inputs = torch.cat([color_scale * image, pos_scale * coords], 1)

    Q, H, feat, pixel_f = model(inputs)
    print(f'Q: {Q.shape}')
    print(f'H: {H.shape}')
    print(f'Feat: {feat.shape}')
    print(f'pixel_f: {pixel_f.shape}')

    labels = H.reshape(height, width).to("cpu").detach().numpy()

    if enforce_connectivity:
        segment_size = height * width / nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels[None], min_size, max_size)[0]

    return labels, pixel_f


if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="/path/to/image")
    parser.add_argument("--weight", default=None, type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=10, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--layer_number", default=3, type=int)
    enforce_connectivity = True
    args = parser.parse_args()

    image = plt.imread(args.image)
    height, width = image.shape[:2]

    s = time.time()
    _, pixel_f = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)

    # Visualize pixel features
    # extracted_features = pixel_f.squeeze(0)
    # extracted_features = extracted_features.cpu().numpy()
    # fig = plt.figure(figsize=(5, 4))
    # for i, map in enumerate(pixel_f):
    #     fig.add_subplot(5, 4, i+1)
    #     plt.imsave(f'pixel_f_{i+1}.jpg', map)


    # Compute superpixels from extracted features
    Q, H, feat = sparse_ssn_iter(pixel_f, args.nspix, args.niter)
    labels = H.reshape(height, width).to("cpu").detach().numpy()
    if enforce_connectivity:
        segment_size = height * width / args.nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels[None], min_size, max_size)[0]

    model_name = args.weight
    print(f"time {time.time() - s}sec")
    plt.imsave(f"{args.image.split('.')[0]}_oversegmented_decoupled.png", mark_boundaries(image, labels))


