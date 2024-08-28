import os
import numpy as np
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

    file_tensor = f"{root}/adj_{DATASET_NAME}^{num_hops}.pt"

    if os.path.exists(file_tensor):
        print(f"Loading A^{num_hops} tensor file...")
        adjacency = torch.load(file_tensor)
    else:

        name = DATASET_NAME.lower() if DATASET_NAME in ['OGBL-COLLAB'] else DATASET_NAME
        name = f'{name}Dataset' if name.lower() in ['wikics'] else name

        with open(f'{root}{name}.edgelist', 'r') as f:

            edgelist = f.read().split('\n')
            
        adjacency = np.zeros((num_nodes,num_nodes))
        for ent in edgelist:
            try:
                a,b = ent.split(' ')
                adjacency[int(a), int(b)] = 1.0
            except ValueError:
                pass

        adjacency = torch.from_numpy(adjacency) #consider passing to GPU
        print(f"Computing A^{num_hops} ...")
        adjacency = torch.linalg.matrix_power(adjacency, num_hops)

        adjacency = 1./(adjacency + 1.0)

        torch.save(adjacency,file_tensor)

    return adjacency
