




"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
from zz_average_of_mul_simulation import save_float_to_csv

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self







"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.COLLAB_edge_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset




"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device






"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, save_file_params, dirs):
    t0 = time.time()
    per_epoch_time = []

    DATASET_NAME = dataset.name

    #assert net_params['self_loop'] == False, "No self-loop support for %s dataset" % DATASET_NAME

    if MODEL_NAME in ['GatedGCN']:
        if net_params['pos_enc']:
            print("[!] Adding graph positional encoding",net_params['pos_enc_dim'])
            dataset._add_positional_encodings(net_params, save_adj=False, #set as True to get adjacency matrix
                                                    use_existing=False) #set as True to use saved laplacian
            
            if net_params['pos_enc_type'].lower() == "learn":
                # This would be used to set the learnable parameter size
                net_params['num_nodes'] = dataset.num_nodes
            
            print('Time PE:',time.time()-t0)
            # if not net_params['use_NAPE']:
            #     # We enforce this only to get the time taken for the Laplacian to be computed
            #     print('\nTIME HAS BEEN COMPUTED!')
            #     exit()

    graph = dataset.graph

    evaluator = dataset.evaluator

    train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg = dataset.train_edges, dataset.val_edges, dataset.val_edges_neg, dataset.test_edges, dataset.test_edges_neg

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Graph: ", graph)
    print("Training Edges: ", len(train_edges))
    print("Validation Edges: ", len(val_edges) + len(val_edges_neg))
    print("Test Edges: ", len(test_edges) + len(test_edges_neg))

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses = []
    epoch_train_hits, epoch_val_hits = [], []

    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        raise NotImplementedError # gave OOM while preparing dense tensor
    else:
        # import train functions for all other GCNs
        from train.train_COLLAB_edge_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        monet_pseudo = None
        if MODEL_NAME == "MoNet":
            print("\nPre-computing MoNet pseudo-edges")
            # for MoNet: computing the 'pseudo' named tensor which depends on node degrees
            us, vs = graph.edges()
            # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
            monet_pseudo = [
                [1/np.sqrt(graph.in_degree(us[i])+1), 1/np.sqrt(graph.in_degree(vs[i])+1)]
                    for i in range(graph.number_of_edges())
            ]
            monet_pseudo = torch.Tensor(monet_pseudo)

        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, optimizer = train_epoch(model, optimizer, device, graph, train_edges,
                                                            params['batch_size'], epoch, monet_pseudo,
                                                            net_params['pos_enc_type'])

                epoch_train_hits, epoch_val_hits, epoch_test_hits = evaluate_network(
                    model, device, graph, train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg, evaluator, params['batch_size'], epoch, monet_pseudo)

                epoch_train_losses.append(epoch_train_loss)
                epoch_train_hits.append(epoch_train_hits)
                epoch_val_hits.append(epoch_val_hits)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)

                writer.add_scalar('train/_hits@10', epoch_train_hits[0]*100, epoch)
                writer.add_scalar('train/_hits@50', epoch_train_hits[1]*100, epoch)
                writer.add_scalar('train/_hits@100', epoch_train_hits[2]*100, epoch)

                writer.add_scalar('val/_hits@10', epoch_val_hits[0]*100, epoch)
                writer.add_scalar('val/_hits@50', epoch_val_hits[1]*100, epoch)
                writer.add_scalar('val/_hits@100', epoch_val_hits[2]*100, epoch)

                writer.add_scalar('test/_hits@10', epoch_test_hits[0]*100, epoch)
                writer.add_scalar('test/_hits@50', epoch_test_hits[1]*100, epoch)
                writer.add_scalar('test/_hits@100', epoch_test_hits[2]*100, epoch)

                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, train_hits=epoch_train_hits[1],
                              val_hits=epoch_val_hits[1], test_hits=epoch_test_hits[1])

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_hits[1])

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')


    train_hits, val_hits, test_hits = evaluate_network(
        model, device, graph, train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg, evaluator, params['batch_size'], epoch, monet_pseudo)

    print(f"Test:\nHits@10: {test_hits[0]*100:.4f}% \nHits@50: {test_hits[1]*100:.4f}% \nHits@100: {test_hits[2]*100:.4f}% \n")
    print(f"Train:\nHits@10: {train_hits[0]*100:.4f}% \nHits@50: {train_hits[1]*100:.4f}% \nHits@100: {train_hits[2]*100:.4f}% \n")
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # Save train metric
    if save_file_params['save']:
        try:
            os.mkdir(save_file_params['folder'])
        except FileExistsError:
            print("Folder Already Exists!")

        save_float_to_csv(value=train_hits[1]*100, filename=os.path.join(save_file_params['folder'],save_file_params['train_file']))
        # Save test metric
        save_float_to_csv(value=test_hits[1]*100, filename=os.path.join(save_file_params['folder'],save_file_params['test_file']))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST HITS@10: {:.4f}\nTEST HITS@50: {:.4f}\nTEST HITS@100: {:.4f}\nTRAIN HITS@10: {:.4f}\nTRAIN HITS@50: {:.4f}\nTRAIN HITS@100: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f}hrs\nAverage Time Per Epoch: {:.4f}s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_hits[0]*100, test_hits[1]*100, test_hits[2]*100, train_hits[0]*100, train_hits[1]*100, train_hits[2]*100,
                  epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))





def main():
    """
        USER CONTROLS
    """


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--layer_type', help="Please give a value for layer_type (for GAT and GatedGCN only)")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--pos_enc', help="Please give a value for pos_enc")
    parser.add_argument('--save_folder', help="The folder to save the file holding the metrics.")
    parser.add_argument('--pos_enc_type', help="One of NAPE, Spectral, Learn, Node-embed, Dist-enc and Relative-enc.")
    parser.add_argument('--num_hops', help="Number of hops for distance encoding position encoding.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # parameters to determine where to save the computed metrics
    save_file_params = config['save_file']
    if args.save_folder is not None:
        save_file_params['folder'] = args.save_folder

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.layer_type is not None:
        net_params['layer_type'] = layer_type
    if args.pos_enc is not None:
        net_params['pos_enc'] = True if args.pos_enc=='True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)
    if args.pos_enc_type is not None:
        net_params['pos_enc_type'] = args.pos_enc_type
    if args.num_hops is not None:
        net_params['num_hops'] = int(args.num_hops)




    # COLLAB
    net_params['in_dim'] = dataset.graph.ndata['feat'].shape[-1]
    net_params['in_dim_edge'] = dataset.graph.edata['feat'].shape[-1]
    net_params['n_classes'] = 1  # binary prediction

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    train_val_pipeline(MODEL_NAME, dataset, params, net_params, save_file_params, dirs)



main()
