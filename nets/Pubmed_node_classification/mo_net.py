import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

import numpy as np

"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""

from layers.gmm_layer import GMMLayer
from layers.mlp_readout_layer import MLPReadout

class MoNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.name = 'MoNet'
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        kernel = net_params['kernel']                       # for MoNet
        dim = net_params['pseudo_dim_MoNet']                # for MoNet
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.device = net_params['device']
        self.n_classes = n_classes
        self.pos_enc = net_params['pos_enc']
        self.proj_pos_enc = net_params['pos_enc_type'] in ["Spectral", "Learn", "Node-embed", "Dist-enc"]
        if self.pos_enc and self.proj_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            if net_params['pos_enc_type'] == "Learn":
                self.learn_param = nn.Parameter(torch.randn(net_params['num_nodes'],hidden_dim)) 
            else:
                self.embed = nn.Linear(pos_enc_dim, hidden_dim , bias=False)

        aggr_type = "sum"                                    # default for MoNet

        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Hidden layer
        for _ in range(n_layers-1):
            self.layers.append(GMMLayer(hidden_dim, hidden_dim, dim, kernel, aggr_type,
                                        dropout, batch_norm, residual))
            self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        # Output layer
        self.layers.append(GMMLayer(hidden_dim, out_dim, dim, kernel, aggr_type,
                                    dropout, batch_norm, residual))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        self.MLP_layer = MLPReadout(out_dim, n_classes)


    def embedding_pos_enc(self, pos_enc):
        """Determines if we use a learnable parameter or not"""
        if pos_enc is None:
            return self.learn_param
        else:
            return self.embed(pos_enc.float())


    def apply_pos_enc(self, h, h_pos_enc):

        if self.proj_pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc)
        else:
            h_pos_enc = h_pos_enc.float()

        return h + h_pos_enc


    def forward(self, g, h, e, h_pos_enc=None):

        h = self.embedding_h(h)
        if self.pos_enc:
            h = self.apply_pos_enc(h, h_pos_enc)

        # computing the 'pseudo' named tensor which depends on node degrees
        g.ndata['deg'] = g.in_degrees()
        g.apply_edges(self.compute_pseudo)
        pseudo = g.edata['pseudo'].to(self.device).float()

        for i in range(len(self.layers)):
            h = self.layers[i](g, h, self.pseudo_proj[i](pseudo))

        return self.MLP_layer(h)

    def compute_pseudo(self, edges):
        # compute pseudo edge features for MoNet
        # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
        srcs = 1/torch.sqrt(edges.src['deg'].float()+1)
        dsts = 1/torch.sqrt(edges.dst['deg'].float()+1)
        pseudo = torch.cat((srcs.unsqueeze(-1), dsts.unsqueeze(-1)), dim=1)
        return {'pseudo': pseudo}

    def loss(self, pred, label):

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss
