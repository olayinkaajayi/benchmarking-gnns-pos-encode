import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import dgl
from dgl.data import PubmedGraphDataset
import json
import networkx as nx
from scipy import sparse as sp

from train.NAPE_modules.trans_pos_encode import TT_Pos_Encode

import itertools


def save_dgl_edgelist(g, filename):
  """Saves the edgelist of a DGL graph to a file.

  Args:
    g: A DGLGraph object.
    filename: The path to the output file.
  """

  with open('/dcs/large/u2034358/'+filename, "w") as f:
      src, dst = g.edges()
      src = src.numpy()
      dst = dst.numpy()
      for src, dst in zip(src,dst):
          f.write(f"{src} {dst}\n")



class PubmedDataset(torch.utils.data.Dataset):
    """
        PubmedDataset Dataset
        Modified the WikiCS.py code
    """
    def __init__(self, DATASET_NAME='Pubmed'):
        self.name = DATASET_NAME
        self.dataset = PubmedGraphDataset(raw_dir='/dcs/large/u2034358')
        self.data = self.dataset[0]
        self.num_split = 20

        self.g, self.labels = None, None
        self.train_masks, self.stopping_masks, self.val_masks, self.test_mask = None, None, None, None
        self.num_classes, self.n_feats = None, None
        self.num_nodes = None

        self._load()

    def _split_train_test(self, data, p=0.8, mask= None):
        # Calculate lengths of each split
        total_len = len(data)
        train_len = int(total_len * p)
        test_len = total_len - train_len

        # Generate random indices for each split
        indices = np.random.permutation(total_len)

        # Create masks for each split
        train_mask = torch.zeros(total_len, dtype=bool)
        test_mask = torch.zeros(total_len, dtype=bool)

        # Assign True values to the respective splits using the random indices
        train_mask[indices[:train_len]] = True
        test_mask[indices[train_len:]] = True

        if mask is not None:
            train_mask *= mask
            test_mask *= mask

        return train_mask, test_mask


    def _print_split(self):
        """This is just a function to print the size of each split"""
        test_mask = self.test_mask
        test_set = {idx for idx in range(len(test_mask)) if test_mask[idx]}
        nodes = set(range(self.num_nodes))
        print(f"\n\nNumber of nodes left: {len(nodes)} **\n")
        for split_num in range(len(self.train_masks)):
            train_mask = self.train_masks[split_num]
            train_set = {idx for idx in range(len(train_mask)) if train_mask[idx]}

            val_mask = self.val_masks[split_num]
            val_set = {idx for idx in range(len(val_mask)) if val_mask[idx]}

            print(f"Val mask shape: {len(val_set)}")
            print(f"Train mask shape: {len(train_set)}")
            print(f"Test mask shape: {len(test_set)}")
            combine = train_set.union(val_set).union(test_set)
            nodes = nodes.difference(combine)

            print(f"Split {split_num}:\nCombine size: {len(combine)}")
            print(f"Number of nodes left: {len(nodes)}")
            print("*******************************\n")

        exit()


    def _load(self):
        t0 = time.time()

        self.num_nodes , self.n_feats = self.data.ndata['feat'].shape
        self.num_classes = self.dataset._num_classes
        self.labels = self.data.ndata['label']
        # self.train_masks = [self.data.ndata['train_mask']]
        # self.val_masks = [self.data.ndata['val_mask']]
        # self.test_mask = self.data.ndata['test_mask']

        self.train_masks = [0]*self.num_split
        self.val_masks = [0]*self.num_split
        #### Split the dataset into train-test first (80-20)
        train_masks, self.test_mask = self._split_train_test(np.arange(self.num_nodes), p=0.8)
        #### Then split the train split into train-val
        #### Randomly select the train-val split for N number of splits
        #### The train-val split should be 75-25 of the "train split".
        for i in range(self.num_split):
            self.train_masks[i], self.val_masks[i] = self._split_train_test(np.arange(self.num_nodes), p=0.75, mask= train_masks)

        # self._print_split()

        self.g = self.dataset._g
        # self.g.ndata['feat'] already exits
        self.g.edata['feat'] = torch.zeros(self.g.number_of_edges(), 1)

        print("[I] Finished loading after {:.4f}s".format(time.time()-t0))

    def _add_positional_encodings(self, pos_enc_dim, hidden_size=None, pos_enc_name='', pos_enc_type="NAPE", scale=110000.0, save_adj=False):

        # Graph positional encoding v/ Laplacian eigenvectors
        g = self.g

        # These are the things I added
        if save_adj:
            print("Saving edgelist for Pubmed dataset...")
            save_dgl_edgelist(self.g, "Pubmed.edgelist")
            print("Done!")
            exit()
        else:
            if pos_enc_type.lower() == "NAPE".lower():
                PE = TT_Pos_Encode(hidden_size, N=self.num_nodes, d=pos_enc_dim, PE_name=pos_enc_name, scale=scale)
                pos_encode = PE.get_position_encoding()
                g.ndata['pos_enc'] = pos_encode.float()
            elif pos_enc_type.lower() == "Sepctral".lower():
                A = g.adjacency_matrix().to_dense().float().numpy()
                N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
                L = sp.eye(g.number_of_nodes()) - N * A * N

                # Eigenvectors with numpy
                # EigVal, EigVec = np.linalg.eig(L.toarray())
                EigVal, EigVec = np.linalg.eig(L)
                idx = EigVal.argsort() # increasing order
                EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])

                g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
            elif pos_enc_type.lower() == "Learn".lower():
                pass
            elif pos_enc_type.lower() == "Node-embed".lower():
                pass
            elif pos_enc_type.lower() == "Dist-enc".lower():
                pass
            elif pos_enc_type.lower() == "Relative-enc".lower():
                pass
            else:
                raise f"{pos_enc_type} is not in the list of position encoding types for this script.\nPlease select from: NAPE, Spectral, Learn, Node-embed, Dist-enc and Relative-enc."


        self.g = g
