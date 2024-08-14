"""
    File to load dataset based on user control from main file
"""
import os
import torch

from data.COLLAB import COLLABDataset
from data.WikiCS import WikiCSDataset
from data.Pubmed import PubmedDataset


def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file
        returns:
        ; dataset object
    """
    # handling for COLLAB dataset
    if DATASET_NAME == 'OGBL-COLLAB':
        return COLLABDataset(DATASET_NAME)

    if DATASET_NAME == 'WikiCS':
        return WikiCSDataset(DATASET_NAME)

    if DATASET_NAME == 'Pubmed':
        return PubmedDataset(DATASET_NAME)


def get_position_encoding(DATASET_NAME, num_nodes):

    with open(f'../node2vec/{DATASET_NAME}.emb', 'r') as f:

        pos_enc_dict = {f.readline()[0]:f.readline()[1:] for i in range(num_nodes+1) if i != 0}
        s_dict = dict(sorted(pos_enc_dict.items()))
        pos_enc = list(s_dict.values())

        return torch.tensor(pos_enc)


def get_position_encoding(DATASET_NAME, num_nodes, num_hops):

    name = DATASET_NAME.lower() if DATASET_NAME in ['OGB-COLLAB'] else DATASET_NAME
    name = f'{name}Dataset' if name.lower() in ['wikics'] else name

    with open(f'/dcs/large/u2034358/{name}.edgelist', 'r') as f:

        edgelist = f.read()
        adjacency = [[1 if [i, j] in set(map(tuple, edgelist)) else 0 for j in range(num_nodes)] for i in range(num_nodes)]

    adjacency = torch.tensor(adjacency) #consider passing to GPU
    adjacency = torch.linalg.matrix_power(adjacency, num_hops)

    return adjacency
