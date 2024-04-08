"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.Pubmed_node_classification.gated_gcn_net import GatedGCNNet
from nets.Pubmed_node_classification.mo_net import MoNet as MoNet_

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'MoNet': MoNet
    }

    return models[MODEL_NAME](net_params)
