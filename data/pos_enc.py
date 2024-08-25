import os
import torch

root = '/dcs/large/u2034358/'

def get_position_encoding2(DATASET_NAME, num_nodes):

    with open(f'../node2vec/emb/{DATASET_NAME}.emb', 'r') as f:

        pos_enc_f = f.read().split('\n')[1:]
        pos_enc_dict = {int(line.split(' ')[0]):list(map(float,line.split(' ')[1:])) for line in pos_enc_f}
        s_dict = dict(sorted(pos_enc_dict.items()))
        pos_enc = list(s_dict.values())

    return torch.tensor(pos_enc)


def get_position_encoding(DATASET_NAME, num_nodes, num_hops=None):

    if num_hops is None:
        return get_position_encoding2(DATASET_NAME, num_nodes)

    name = DATASET_NAME.lower() if DATASET_NAME in ['OGB-COLLAB'] else DATASET_NAME
    name = f'{name}Dataset' if name.lower() in ['wikics'] else name

    with open(f'{root}{name}.edgelist', 'r') as f:

        edgelist = f.read()
        adjacency = [[1 if [i, j] in set(map(tuple, edgelist)) else 0 for j in range(num_nodes)] for i in range(num_nodes)]

    adjacency = torch.tensor(adjacency) #consider passing to GPU
    adjacency = torch.linalg.matrix_power(adjacency, num_hops)

    return adjacency
