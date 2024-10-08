import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import dgl
import json
import networkx as nx
from scipy import sparse as sp

from train.NAPE_modules.trans_pos_encode import TT_Pos_Encode
from .pos_enc import get_position_encoding

import itertools

root = '/dcs/large/u2034358/'

class WikiCSDataset(torch.utils.data.Dataset):
    """
        Wiki-CS Dataset
        Adapted from
        https://github.com/pmernyei/wiki-cs-dataset/
    """
    def __init__(self, DATASET_NAME='WikiCS', path=f"{root}data/WikiCS/"):
        self.name = DATASET_NAME
        self.data = json.load(open(os.path.join(path, 'data.json')))

        self.g, self.labels = None, None
        self.train_masks, self.stopping_masks, self.val_masks, self.test_mask = None, None, None, None
        self.num_classes, self.n_feats = None, None
        self.num_nodes = None

        self._load()

    def _load(self):
        t0 = time.time()
        print("[I] Loading WikiCS ...")
        features = torch.FloatTensor(np.array(self.data['features']))
        self.labels = torch.LongTensor(np.array(self.data['labels']))

        self.train_masks = [torch.BoolTensor(tr) for tr in self.data['train_masks']]
        self.val_masks = [torch.BoolTensor(val) for val in self.data['val_masks']]
        self.stopping_masks = [torch.BoolTensor(st) for st in self.data['stopping_masks']]
        self.test_mask = torch.BoolTensor(self.data['test_mask'])

        self.n_feats = features.shape[1]
        self.num_nodes = features.shape[0]
        self.num_classes = len(set(self.data['labels']))

        # test_mask = self.data['test_mask']
        # test_set = {idx for idx in range(len(test_mask)) if test_mask[idx]}
        # nodes = set(range(self.num_nodes))
        # print(f"\n\nNumber of nodes left: {len(nodes)} **\n")
        # for split_num in range(len(self.data['train_masks'])):
        #     train_mask = self.data['train_masks'][split_num]
        #     train_set = {idx for idx in range(len(train_mask)) if train_mask[idx]}
        #
        #     val_mask = self.data['val_masks'][split_num]
        #     val_set = {idx for idx in range(len(val_mask)) if val_mask[idx]}
        #
        #     print(f"Val mask shape: {len(val_set)}")
        #     print(f"Train mask shape: {len(train_set)}")
        #     print(f"Test mask shape: {len(test_set)}")
        #     combine = train_set.union(val_set).union(test_set)
        #     nodes = nodes.difference(combine)
        #
        #     print(f"Split {split_num}:\nCombine size: {len(combine)}")
        #     print(f"Number of nodes left: {len(nodes)}")
        #     print("*******************************\n")
        #
        # exit()

        self.g = dgl.DGLGraph()
        self.g.add_nodes(len(self.data['features']))
        edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i,nbs in enumerate(self.data['links'])]))

        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        self.g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        self.g.add_edges(dst, src)

        self.g.ndata['feat'] = features # available features
        self.g.edata['feat'] = torch.zeros(self.g.number_of_edges(), 1)

        print("[I] Finished loading after {:.4f}s".format(time.time()-t0))

    def _add_positional_encodings(self, net_params, save_adj=False):

        # Parameter list:
        pos_enc_dim, hidden_size, pos_enc_name = net_params['pos_enc_dim'], net_params['hidden_dim'], net_params['pos_enc_name']
        pos_enc_type, scale, num_hops = net_params['pos_enc_type'], float(net_params['scale']), net_params['num_hops']
        

        # Graph positional encoding v/ Laplacian eigenvectors
        g = self.g

        # These are the things I added
        if save_adj:
            A = g.adjacency_matrix().to_dense().float().numpy()
            filename = 'WikiCSDataset_adj.npy'
            print("Saving adjacency matrix for WikiCSDataset...")
            with open(root+filename,'wb') as f:
                np.save(f,np.array(A))
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
                pass # It will raise error and the exception would be caught

            elif pos_enc_type.lower() == "Node-embed".lower():
                g.ndata['pos_enc'] = get_position_encoding(self.name, self.num_nodes)

            elif pos_enc_type.lower() == "Dist-enc".lower():
                g.ndata['pos_enc'] = get_position_encoding(self.name, self.num_nodes, num_hops)
                net_params["pos_enc_dim"] = self.num_nodes

            elif pos_enc_type.lower() == "Relative-enc".lower():
                pass

            else:
                raise f"{pos_enc_type} is not in the list of position encoding types for this script.\nPlease select from: NAPE, Spectral, Learn, Node-embed, Dist-enc and Relative-enc."


        self.g = g
