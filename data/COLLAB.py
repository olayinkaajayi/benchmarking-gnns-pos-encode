import time
import dgl
import torch
from torch.utils.data import Dataset

from ogb.linkproppred import DglLinkPropPredDataset, Evaluator

from scipy import sparse as sp
import numpy as np

from train.NAPE_modules.trans_pos_encode import TT_Pos_Encode

root = '/dcs/large/u2034358/'

def positional_encoding(g, pos_enc_dim, dataset_name, use_existing):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    if not use_existing:
        # Laplacian
        # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        A = g.adj().float()
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        # L = sp.eye(g.number_of_nodes()) - N * A * N
        L = sp.eye(g.number_of_nodes()) - N.multiply(A).multiply(N)

        # # Eigenvectors with numpy
        # EigVal, EigVec = np.linalg.eig(L.toarray())
        # idx = EigVal.argsort() # increasing order
        # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float()

        # Eigenvectors with scipy
        #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
        EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
        EigVec = EigVec[:, EigVal.argsort()] # increasing order
        PE = np.real(EigVec[:,1:pos_enc_dim+1])
        
        # Saved the position encoding
        with open(root+f'{dataset_name}_lap-PE.npy', 'wb') as f:
            np.save(f, PE)

        print("Laplacian Position encoding saved!")

    else:
        # Load saved position encoding
        with open(root+f'{dataset_name}_lap-PE.npy', 'rb') as f:
            PE = np.load(f)

        print("Successfully loaded the Laplacian Position encoding!")
    
    g.ndata['pos_enc'] = torch.from_numpy(PE).float()

    return g



def save_edgelist_with_weights(graph, filename):
    # Retrieve the edges and their weights from the DGL graph
    src, dst = graph.all_edges(form='uv')
    weights = graph.edata['weight']  # Assuming there is a 'weight' edge attribute

    # Get the source and destination nodes as lists
    src_list = src.numpy().tolist()
    dst_list = dst.numpy().tolist()
    weights_list = weights.numpy().tolist()

    with open(root+filename, "w") as file:
        for i in range(len(src_list)):
            file.write(f"{src_list[i]} {dst_list[i]} {weights_list[i][0]}\n")


def save_dgl_edgelist(g, filename):
  """Saves the edgelist of a DGL graph to a file.

  Args:
    g: A DGLGraph object.
    filename: The path to the output file.
  """

  with open(root+filename, "w") as f:
      src, dst = g.edges()
      src = src.numpy()
      dst = dst.numpy()
      for src, dst in zip(src,dst):
          f.write(f"{src} {dst}\n")



class COLLABDataset(Dataset):
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.dataset = DglLinkPropPredDataset(name='ogbl-collab')

        self.graph = self.dataset[0]  # single DGL graph

        # Create edge feat by concatenating weight and year
        self.graph.edata['feat'] = torch.cat(
            [self.graph.edata['weight'], self.graph.edata['year']],
            dim=1
        )

        self.num_nodes = self.graph.number_of_nodes()

        self.split_edge = self.dataset.get_edge_split()
        self.train_edges = self.split_edge['train']['edge']  # positive train edges
        self.val_edges = self.split_edge['valid']['edge']  # positive val edges
        self.val_edges_neg = self.split_edge['valid']['edge_neg']  # negative val edges
        self.test_edges = self.split_edge['test']['edge']  # positive test edges
        self.test_edges_neg = self.split_edge['test']['edge_neg']  # negative test edges

        self.evaluator = Evaluator(name='ogbl-collab')

        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def _add_positional_encodings(self, pos_enc_dim, hidden_size=None, pos_enc_name='', pos_enc_type="NAPE", scale=110000.0, save_adj=False, use_existing=False):

        # Graph positional encoding v/ Laplacian eigenvectors
        if not use_NAPE:
            self.graph = positional_encoding(self.graph, pos_enc_dim, self.name, use_existing)

        # These are the things I added
        if save_adj:
            print("Saving edgelist for ogb-COLLAB dataset...")
            save_dgl_edgelist(self.graph, "ogb-collab.edgelist")
            # save_edgelist_with_weights(self.graph, "ogb-collab.edgelist")
            print("Done!")
            exit()
        else:
            if pos_enc_type.lower() == "NAPE":
                PE = TT_Pos_Encode(hidden_size, N=self.num_nodes, d=pos_enc_dim, PE_name=pos_enc_name, scale=scale)
                pos_encode = PE.get_position_encoding()
                self.graph.ndata['pos_enc'] = pos_encode.float()
            elif pos_enc_type.lower() == "Sepctral":
                self.graph = positional_encoding(self.graph, pos_enc_dim, self.name, use_existing)
            elif pos_enc_type.lower() == "Learn":
                pass
            elif pos_enc_type.lower() == "Node-embed":
                pass
            elif pos_enc_type.lower() == "Dist-enc":
                pass
            elif pos_enc_type.lower() == "Relative-enc":
                pass
            else:
                raise f"{pos_enc_type} is not in the list of position encoding types for this script.\nPlease select from: NAPE, Spectral, Learn, Node-embed, Dist-enc and Relative-enc."

