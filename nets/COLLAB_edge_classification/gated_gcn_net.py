import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer, GatedGCNLayerEdgeFeatOnly, GatedGCNLayerIsotropic
from layers.mlp_readout_layer import MLPReadout

class GatedGCNNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        self.proj_pos_enc = net_params['pos_enc_type'] in ["Spectral", "Learn", "Node-embed", "Dist-enc"]
        if self.pos_enc and self.proj_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            if net_params['pos_enc_type'] == "Learn":
                self.learn_param = nn.Parameter(torch.randn(net_params['num_nodes'],hidden_dim)) 
            else:
                self.embed = nn.Linear(pos_enc_dim, hidden_dim, bias=False)

        self.layer_type = {
            "edgereprfeat": GatedGCNLayer,
            "edgefeat": GatedGCNLayerEdgeFeatOnly,
            "isotropic": GatedGCNLayerIsotropic,
        }.get(net_params['layer_type'], GatedGCNLayer)

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([ self.layer_type(hidden_dim, hidden_dim, dropout,
                                                      self.batch_norm, self.residual) for _ in range(n_layers-1) ])
        self.layers.append(self.layer_type(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.MLP_layer = MLPReadout(2*out_dim, 1)

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

        h = self.embedding_h(h.float())
        if self.pos_enc:
            h = self.apply_pos_enc(h, h_pos_enc)
            
        if not self.edge_feat:
            e = torch.ones_like(e).to(self.device)
        e = self.embedding_e(e.float())

        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        return h

    def edge_predictor(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=1)
        x = self.MLP_layer(x)

        return torch.sigmoid(x)

    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss

        return loss
