import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from lib.ssn.ssn import sparse_ssn_iter
from graph import Graph, Superpixel
from lib.dataset import spixel_dataset
from torch.utils.data import DataLoader
import sys

# import networkx as nx
# from torch_geometric.data import Data


@torch.no_grad()
def inference(image, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None, mlp_weight=None, enforce_connectivity=True):
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

    # Load MLP
    from model import MLP
    if mlp_weight is not None:
        mlp_model = MLP().to("cuda")
        mlp_model.load_state_dict(torch.load(mlp_weight))
        mlp_model.eval()

    height, width = image.shape[:2]

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)

    coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda")), 0)
    coords = coords[None].float()
    # print(f'coords: {coords.shape}, values: {torch.unique(coords)}')
    # print(f'reshaped coords: {coords.reshape(*coords.shape[:2], -1).shape}')

    image = rgb2lab(image)
    image = torch.from_numpy(image).permute(2, 0, 1)[None].to("cuda").float()

    inputs = torch.cat([color_scale * image, pos_scale * coords], 1)

    Q, H, superpixel_features, num_spixels_width = model(inputs)

    # SECTION BEGIN: Get edge data from features for MLP analysis
    # Extract superpixel data
    superpixels_list = []
    # reshaped_labels = H.reshape(-1, 200, 200)
    reshaped_labels = H.reshape(-1, height, width)
    # print(f'Labels shape: {H.shape}, reshaped labels shape: {reshaped_labels.shape}, num_spixels_width: {num_spixels_width}')
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
    edge_indices = edge_indices.to("cuda")
    # SECTION END

    # USE MLP
    updated_spixel_features, prob_vector = mlp_model(x=superpixel_features, edge_index=edge_indices)

    # SECTION BEGIN

    torch.set_printoptions(threshold=10_000)
    edge_indices = torch.transpose(edge_indices, 0, 1)
    # print(f'edge indices shape: {edge_indices.shape}')
    # print(f'edge indices : {edge_indices}')
    # print(f'Prob vector: {prob_vector}')
    thresholds = [0.95, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
    # t = 0.95
    for t in thresholds:
        components = {}
        val = 0
        for i in range(len(edge_indices)):
            # print(f'Checking {edge_indices[i][0]} and {edge_indices[i][1]}')
            if (prob_vector[i] > t):
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
                    val+=1

            # print(f'Components: {components}')
            H_prime = H.detach().clone()
            for i in range(len(torch.unique(H))):
                if i in components.keys():
                    H_prime[H_prime == i] = components[i]

        # print(f'H prime: {torch.unique(H_prime)}')
        H_prime = torch.unsqueeze(H_prime, 0)

        # Plot MLP prediction
        image = plt.imread(args.image)
        labels = H_prime.reshape(height, width).to("cpu").detach().numpy()
        plt.imsave(f"result_MLP_{t}.png", mark_boundaries(image, labels))
    


    # SECTION END

    # Plot SSN prediction
    labels = H.reshape(height, width).to("cpu").detach().numpy()
    image = plt.imread(args.image)
    plt.imsave("result_SSN.png", mark_boundaries(image, labels))



    # if enforce_connectivity:
    #     segment_size = height * width / nspix
    #     min_size = int(0.06 * segment_size)
    #     max_size = int(3.0 * segment_size)
    #     labels = _enforce_label_connectivity_cython(
    #         labels[None], min_size, max_size)[0]

    return labels, Q, H


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
    parser.add_argument("--mlp_weight", type=str, help="path to pretrained mlp")
    enforce_connectivity = True
    args = parser.parse_args()

    image = plt.imread(args.image)
    height, width = image.shape[:2]

    s = time.time()
    _, _, pixel_f = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight, args.mlp_weight)

    if pixel_f is not None:
        sys.exit(0)
    # Visualize pixel features
    # extracted_features = pixel_f.squeeze(0)
    # extracted_features = extracted_features.cpu().numpy()
    # fig = plt.figure(figsize=(5, 4))
    # for i, map in enumerate(pixel_f):
    #     fig.add_subplot(5, 4, i+1)
    #     plt.imsave(f'pixel_f_{i+1}.jpg', map)

    # Compute superpixels from extracted features
    Q, H, feat, num_spixels_width = sparse_ssn_iter(pixel_f, args.nspix, args.niter)

    print(f'Q: {Q.shape}')
    print(f'H: {H.shape}, Unique value: {torch.unique(H)}')
    # feat = torch.t(torch.squeeze(feat))
    print(f'Feat: {feat.shape}')

    pixel_f = pixel_f.squeeze()
    pixel_f = pixel_f.to('cpu').detach().numpy()
    print(f'pixel_f: {pixel_f.shape}')
    labels = H.reshape(height, width).to("cpu").detach().numpy()
    print(f'Labels: {np.unique(labels)}, {labels.shape}, max_val = {np.max(labels)}')

    # if enforce_connectivity:
    #     segment_size = height * width / args.nspix
    #     min_size = int(0.06 * segment_size)
    #     max_size = int(3.0 * segment_size)
    #     labels = _enforce_label_connectivity_cython(
    #         labels[None], min_size, max_size)[0]

    print(f'Labels: {np.unique(labels)}, {labels.shape}, max_val = {np.max(labels)}')

    print(f'First superpixel feat: ', torch.squeeze(feat)[:, 0])
    # Initialize superpixels
    superpixels = {}
    for i in range(len(np.unique(labels))):
        pixel_indices_2d = np.argwhere(labels == i)
        pixel_indices_1d = np.argwhere(labels.flatten() == i)
        # print(f'Spixel:{i}, num_pixels: {pixel_indices_2d.shape}, pixel list: {pixel_indices_2d}')
        # print(f'Spixel:{i}, num_pixels: {pixel_indices_1d.shape}, pixel list: {pixel_indices_1d}')
        # mask = np.where(labels == i, 1, 0)
        # print(f'mask: {mask.shape}, unique vals: {np.unique(mask)}, nonzero: {np.count_nonzero(mask)}')
        # plt.imsave(f'mask.png', mask)
        # masked_features = np.mean(np.multiply(pixel_f, mask), axis=(1,2))
        # masked_features = torch.unsqueeze(torch.tensor(masked_features), 0)
        # print(f'masked features shape: {masked_features.shape}')
        # print(f'features shape: {feat[:, :, i].shape}')
        # print(f'nonzero elements: {np.count_nonzero(masked_features)}, shape: {masked_features.shape}')
        # print(f'Masked features: {masked_features}')
        superpixels[i] = Superpixel(index=i, features=feat[:, :, i], pixel_indices_2d=pixel_indices_2d,
                                    num_spixels_width=num_spixels_width,
                                    image_width=width, num_spixels=np.max(labels), pixel_indices_1d=pixel_indices_1d)
        # superpixels[i] = Superpixel(index=i, features=torch.tensor(masked_features), pixel_indices_2d=pixel_indices_2d,
        #                             num_spixels_width=num_spixels_width,
        #                             image_width=width, num_spixels=np.max(labels), pixel_indices_1d=pixel_indices_1d)

    print(f'No. of Superpixels: {len(superpixels)}')
    # print(f'Superpixels: {superpixels}')

    # Compute neighbor weights
    for key in superpixels.keys():
        spix = superpixels[key]
    #     spix.compute_neighbor_weights(superpixels)
        print(
            f'key: {key} Spix index: {spix.index}, label = {spix.label}, features = {spix.features.shape}, centroid = {spix.centroid}, neighbors = {spix.neighbor_spixels}')

    # Create dataset
    train_dataset = spixel_dataset.SpixelDataset(superpixels)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for data in train_loader:
        x1, x2 = data
        print(f'x1: {x1.shape}, x2: {x2.shape}')



    # Perform iterative grouping
    # num_iterations = 10
    # threshold = 0.98
    # for i in range(num_iterations):
    #     workset = set(superpixels.keys())
    #     # print(f'Iteration {i}, threshold: {threshold}')
    #     # print(f'Workset: {workset}')
    #     merges = 0
    #     while (len(workset) != 0):
    #         spix_idx = workset.pop()
    #         current_spix = superpixels[spix_idx]
    #         # print(f'Current spix: {spix_idx}, neighbors: {current_spix.neighbor_weights}')
    #         nn_index_valid = -1
    #         # Get nearest neighbor still in workset
    #         for idx in range(len(current_spix.neighbor_spixels)):
    #             # print(f'idx: {idx}')
    #             nn_index, nn_weight = list(current_spix.neighbor_weights.items())[idx]
    #         # if i < len(list(current_spix.neighbor_weights.items())):
    #         #     nn_index, nn_weight = list(current_spix.neighbor_weights.items())[i]
    #             if nn_index in workset and current_spix.label != superpixels[nn_index].label:
    #                 nn_index_valid = nn_index
    #                 # print(f'Neighbor selected: {nn_index}')
    #                 # break
    #             # elif nn_index not in workset and idx == len(current_spix.neighbor_spixels):
    #             #     nn_exists = False
    #             # elif nn_index not in workset:
    #             #     continue
    #         # print(f'current spix: {spix_idx}, nn index: {nn_index_valid}, nn weight: {nn_weight}')
    #                 if nn_index_valid != -1 and nn_weight > threshold:
    #                     # print(f'Combining {spix_idx} and {nn_index_valid}, weight: {nn_weight}')
    #                     # print(f'Labels: {spix_idx} - {superpixels[spix_idx].label}.................{nn_index_valid} - {superpixels[nn_index_valid].label}')
    #                     workset.remove(nn_index_valid)
    #                     superpixels[nn_index_valid].label = superpixels[spix_idx].label
    #                     merges += 1
    #
    #     new_labels = np.copy(labels).flatten()
    #     for key in superpixels.keys():
    #         spix = superpixels[key]
    #         # print(f'Spix index: {spix.index}, label = {spix.label}, pixel indices: {spix.pixel_indices_1d}')
    #         for spix_idx in spix.pixel_indices_1d:
    #             new_labels[spix_idx] = spix.label
    #
    #     new_labels = new_labels.reshape(height, width)
    #     # print(f'New Labels: {np.unique(new_labels)}')
    #     # plt.imsave(f"merged/merged_{i}.png", mark_boundaries(image, new_labels))
    #     # threshold -= 0.01
    #     num_iterations -= 1

    # for key in superpixels.keys():
    #     spix = superpixels[key]
    #     # spix.compute_neighbor_weights(superpixels)
        # print(
            # f'Spix index: {spix.index}, label = {spix.label}')

    # Spectral Clustering
    # adjaceny_matrix = np.zeros((len(superpixels), len(superpixels)))
    # degree_matrix = np.zeros((len(superpixels), len(superpixels)))
    # for idx, spix in superpixels.items():
    #     print(f'Superpixel :{idx}, neighbors: {spix.neighbor_weights}')
    #     degree_matrix[idx, idx] = len(spix.neighbor_weights)
    #     for neighbor, weight in spix.neighbor_weights.items():
    #         adjaceny_matrix[idx][neighbor] = weight
    #     # print(degree_matrix[idx, :])
    # # print(adjaceny_matrix.shape)
    # graph_laplacian = degree_matrix - adjaceny_matrix
    #
    # # eigenvalues and eigenvectors
    # vals, vecs = np.linalg.eig(graph_laplacian)
    #
    # # sort these based on the eigenvalues
    # vecs = vecs[:, np.argsort(vals)]
    # vals = vals[np.argsort(vals)]
    #
    # print(f'eigenvalues: {vals}')
    # new_labels = np.copy(labels).flatten()
    # for key in superpixels.keys():
    #     spix = superpixels[key]
    #     # print(f'Spix index: {spix.index}, label = {spix.label}, pixel indices: {spix.pixel_indices_1d}')
    #     for spix_idx in spix.pixel_indices_1d:
    #         new_labels[spix_idx] = spix.label
    #
    # new_labels = new_labels.reshape(height, width)
    # plt.imsave("merged.png", mark_boundaries(image, new_labels))

    # Construct Adjacency Matrix
    # edge_indices = []
    # # weights_list = []
    # # weights = None
    # edges = None
    # for superpixel in superpixels:
    #     print(f'Spixel Neighbors of {superpixel.index}: ', superpixel.neighbor_spixels)
    #     coords = superpixel.convert_spixel_index_to_coordinates()
    #     print(f'coords received: {coords.shape},  {coords}')
    # # curr_weights = superpixel.compute_neighbor_weights(superpixels)
    # # print(f'Neighbor weights: {curr_weights}, no. of neighbors: {curr_weights.shape}')
    #     edge_indices.append(coords)
    # # print(f'Current Weights: {curr_weights}')
    #     edge_indices += coords
    #     print(f'Current Edges: {coords}')
    #     edge_indices += coords
    #     if edges is None:
    #         edges = coords
    #     else:
    #         edges = np.hstack([edges, coords])
    # # if weights is None:
    # #     weights = curr_weights
    # # else:
    # #     weights = np.hstack([weights, curr_weights])
    # # edge_indices.append(coords)
    #
    # edge_indices += coords
    # # weights_list += curr_weights
    # # weights_list.append(weights)
    # edge_indices = np.asarray(edge_indices)
    # # weights_list = [*weights_list]
    #
    # # print(f'Edge indices length: {len(edge_indices)}, type: {type(edge_indices)}, out: {edge_indices}')
    # print(f'Edge indices length: {edge_indices.shape}, type: {type(edge_indices)}, out: {edge_indices}')
    # edges = edges.T
    # print(f'Edge indices length: {edges.shape}, type: {type(edges)}, out: {edges}')
    # # print(f'Weights length: {weights.shape}, type: {type(weights)}, out: {weights}')
    #
    # edges_x = edges[:, 0]
    # edges_y = edges[:, 1]
    # print(f'Edges x: {edges_x.shape}, type: {type(edges_x)}, out: {edges_x}')
    # print(f'Edges y: {edges_y.shape}, type: {type(edges_y)}, out: {edges_y}')
    # print(f'Weights length: {len(weights_list)}, type: {type(weights_list)},  out: {weights_list}')
    # print(f'Flattened Weights: {weights_list.flatten()}, shape: {weights_list.flatten().shape}')
    # for i in np.arange(edge_indices.shape[0]):
    # print(f'i:{i}')
    # print(f'Edge index {i}: {edge_indices[i]}, type:{type(edge_indices[i])}, shape: {edge_indices[i].shape}')

    # # Setup GNN
    # x = torch.t(torch.squeeze(feat))
    # x = torch.tensor(x)
    # edges = torch.tensor(edges)
    # weights = torch.tensor(weights)
    # print(f'X: {x.shape}')
    # print(f'Edge Index: {edges.shape}')
    # print(f'Edge Attr: {weights.shape}')
    # data = Data(x=x, edge_index=edges, edge_attr=weights)
    # print(f'Graph Data: {data}')
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # import torch_geometric
    # graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
    # f = plt.figure()
    # nx.draw(graph, ax=f.add_subplot(111))
    # f.savefig('data_graph.png')
    #
    # # Normalized Cut
    # res = torch_geometric.utils.normalized_cut(edge_index=edges, edge_attr=weights)
    # print(f'Normalized Cuts Result shape: {res.shape}, values:{res}')
    # print(f'Unique values:{torch.unique(res)}')

    # num_spixels_width = num_spixels_width.item()

    # superpixels_no_connectivity = []
    #
    # print('Feat values:')
    # for i in np.arange(feat.shape[-1]):
    #     print(f'feat {i}: {feat[1, 20, i]}')
    #
    # for i in range(len(np.unique(labels))):
    #     pixel_indices = np.argwhere(labels == i)
    #     superpixels_no_connectivity.append(Superpixel(index=i, features=feat[:, :, i], pixel_indices=pixel_indices,
    #                                   num_spixels_width=num_spixels_width, spixel_labels=np.unique(labels),
    #                                   image_width=width))
    #
    # print('superpixels no connectivity: ' , len(superpixels_no_connectivity))
    # superpixels_connected = []
    # if enforce_connectivity:
    #     segment_size = height * width / args.nspix
    #     min_size = int(0.06 * segment_size)
    #     max_size = int(3.0 * segment_size)
    #     labels = _enforce_label_connectivity_cython(
    #         labels[None], min_size, max_size)[0]
    #
    # print(f'Labels after connectivity: {np.unique(labels)}, {labels.shape}')
    # for i in range(len(np.unique(labels))):
    #     pixel_indices = np.argwhere(labels == i)
    #     superpixels_connected.append(Superpixel(index=i, features=feat[:, :, i], pixel_indices=pixel_indices,
    #                                   num_spixels_width=num_spixels_width, spixel_labels=np.unique(labels),
    #                                   image_width=width))
    # print('superpixels with connectivity: ', len(superpixels_connectivity))

    # centroids = []
    # image = mark_boundaries(image, labels)

    # for superpixel in superpixels:
    #     # print(f'Centroid: ', superpixel.centroid.shape)
    #     # centroids.append(superpixel.centroid)
    #     # print(f'superpixel label: {superpixel.index}, Neighbors: {superpixel.neighbor_spixels}')
    #     plt.imshow(image)
    #     plt.plot(superpixel.centroid[1], superpixel.centroid[0], 'x')
    #     for inner_superpixel in superpixels:
    #         if inner_superpixel.index in superpixel.neighbor_spixels:
    #             plt.plot(inner_superpixel.centroid[1], inner_superpixel.centroid[0], 'o')
    #     plt.savefig(f'temp/{superpixel.index}_neighbors.png')
    #     plt.show()
    #     plt.close()
    #     plt.clf()

    # G = nx.Graph()
    # for key in superpixels.keys():
    #     superpixel = superpixels[key]
    #     plt.imshow(image)
    #     plt.scatter(x=superpixel.centroid[1], y=superpixel.centroid[0], marker='X')
    #     neighbors = list(superpixel.neighbor_weights.items())
    #     nn = neighbors[0][0]
    #     nn_weight = neighbors[0][1]
    #     fn = neighbors[-1][0]
    #     fn_weight = neighbors[-1][1]
        # print(f'nn: {nn}, nn weight: {nn_weight}')
        # print(f'fn: {fn}, fn weight: {fn_weight}')
        # neighbor_weights = superpixel.compute_neighbor_weights(superpixels)
        # G.add_edges_from([(superpixel.index, neighbor,
        #                    np.dot(superpixels[superpixel.index].features.to("cpu").detach().numpy(),
        #                           superpixels[neighbor].features.to("cpu").detach().numpy().T)) for neighbor in
        #                   superpixel.neighbor_spixels])
        # for neighbor in superpixel.neighbor_spixels:
        #     weight = np.dot(superpixels[superpixel.index].features.to("cpu").detach().numpy(),
        #                     superpixels[neighbor].features.to("cpu").detach().numpy().T)
        #     neighbor_weights.append(weight[0, 0])
        #     neighbor_weights[neighbor] = weight
        # neighbor_weights = np.asarray(neighbor_weights)
        # plt.scatter()
        # print(f'Superpixel: {superpixel.index}, neighbor weights: {neighbor_weights}, nn: {superpixel.nn}, fn: {superpixel.fn}')
        # plt.scatter(x=superpixels[nn].centroid[1], y=superpixels[nn].centroid[0], marker='^')
        # plt.scatter(x=superpixels[fn].centroid[1], y=superpixels[fn].centroid[0], marker='v')
        # plt.show()
        # plt.savefig(f'temp/neighbors/custom_ft/{superpixel.index}.png')
        # plt.clf()
        # plt.close()
    # G.add_weighted_edges_from([(superpixel.index, neighbor, weight[0, 0])])
    # G.add_edge(superpixel.index, neighbor, weight=weight[0, 0])
    # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos, node_size=50)
    # nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    # nx.draw_networkx_labels(G, pos)
    # plt.show()
    # plt.savefig('graph_weighted.png')

    # print('Superpixels: ', superpixels)

    # # print(f"Spixel {i}: ", np.count_nonzero(indices))
    # if i==1 or i==87:
    #     print(indices)

    # plt.imsave(f"temp/{args.image.split('.')[0]}_oversegmented_decoupled_not_connected_lesspix.png", mark_boundaries(image, labels))
    # if enforce_connectivity:
    #     segment_size = height * width / args.nspix
    #     min_size = int(0.06 * segment_size)
    #     max_size = int(3.0 * segment_size)
    #     labels = _enforce_label_connectivity_cython(
    #         labels[None], min_size, max_size)[0]

    # centroids = np.asarray(centroids)
    # print('centroids: ', centroids.shape)
    # print('centroids: ', centroids)
    # print(f'centroids: {centroids[0]} , {type(centroids[0])}')
    # print(f'centroids 0: {centroids[:, 0]}')
    # print(f'centroids 1: {centroids[:, 1]}')

    # plt.imshow(image)
    # plt.scatter(x=centroids[:, 0], y=centroids[:, 1], marker='X', markersize='green')
    # plt.plot(centroids[:, 1], centroids[:, 0], 'o')
    # plt.savefig('centroids.png')
    # plt.show()
    # plt.text(centroids[:, 1], centroids[:, 0], np.arange(centroids.shape[0]))
    # plt.savefig('centroids_labelled.png')
    # plt.show()

    # plt.imsave(f"temp/{args.image.split('.')[0]}_oversegmented_decoupled_connected_lesspix.png",)
    # print('Labels after connectivity: ', np.unique(labels))
    # model_name = args.weight
    # print(f"time {time.time() - s}sec")
