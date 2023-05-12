import numpy as np
import json
import argparse
from torch.utils.data import Dataset
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations

from metrics import AverageNonzeroTripletsMetric
import time
from time import localtime, strftime
import os
import pickle
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn import metrics
from model import *
from utils import *
from layers import *

"""
    Training KPGNN
    Paper: Knowledge-Preserving Incremental Social Event Detection via Heterogeneous GNNs
    Author: Yuwei Cao et al.
    github: https://github.com/RingBDStack/KPGNN
"""

st = time.time()


class args_define():

    parser = argparse.ArgumentParser()
    # Hyper-parameters
    parser.add_argument('--n_epochs', default=30, type=int,
                        help="Number of initial-training/maintenance-training epochs.")
    parser.add_argument('--n_infer_epochs', default=0, type=int,
                        help="Number of inference epochs.")
    parser.add_argument('--window_size', default=3, type=int,
                        help="Maintain the model after predicting window_size blocks.")
    parser.add_argument('--patience', default=5, type=int,
                        help="Early stop if performance did not improve in the last patience epochs.")
    parser.add_argument('--margin', default=3., type=float,
                        help="Margin for computing triplet losses")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument('--batch_size', default=2000, type=int,
                        help="Batch size (number of nodes sampled to compute triplet loss in each batch)")
    parser.add_argument('--n_neighbors', default=800, type=int,
                        help="Number of neighbors sampled for each node.")
    parser.add_argument('--hidden_dim', default=8, type=int,
                        help="Hidden dimension")
    parser.add_argument('--out_dim', default=32, type=int,
                        help="Output dimension of tweet representations")
    parser.add_argument('--num_heads', default=16, type=int,
                        help="Number of heads in each GAT layer")
    parser.add_argument('--use_residual', dest='use_residual', default=True,
                        action='store_false',
                        help="If true, add residual(skip) connections")
    parser.add_argument('--validation_percent', default=0.2, type=float,
                        help="Percentage of validation nodes(tweets)")
    parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False,
                        action='store_true',
                        help="If true, use hardest negative messages to form triplets. Otherwise use random ones")
    parser.add_argument('--use_dgi', dest='use_dgi', default=True,
                        action='store_true',
                        help="If true, add a DGI term to the loss. Otherwise use triplet loss only")
    parser.add_argument('--remove_obsolete', default=2, type=int,
                        help="If 0, adopt the All Message Strategy: keep inseting new message blocks and never remove;\n" +
                             "if 1, adopt the Relevant Message Strategy: remove obsolete training nodes (that are not connected to the new messages arrived during the last window) during maintenance;\n" +
                             "if 2, adopt the Latest Message Strategy: during each prediction, use only the new data to construct message graph, during each maintenance, use only the last message block arrived during the last window for continue training.\n" +
                             "See Section 4.4 of the paper for detailed explanations of these strategies. Note the message graphs for 0/1 and 2 need to be constructed differently, see the head comment of custom_message_graph.py")

    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=False,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--data_path', default='./data_lstm/', #default='./incremental_0808/',
                        type=str, help="Path of features, labels and edges")
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--mask_path', default=None,
                        type=str, help="File path that contains the training, validation and test masks")
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--resume_path', default=None,
                        type=str,
                        help="File path that contains the partially performed experiment that needs to be resume.")
    parser.add_argument('--resume_point', default=0, type=int,
                        help="The block model to be loaded.")
    parser.add_argument('--resume_current', dest='resume_current', default=True,
                        action='store_false',
                        help="If true, continue to train the resumed model of the current block(to resume a partally trained initial/mantenance block);\
                            If false, start the next(infer/predict) block from scratch;")
    parser.add_argument('--log_interval', default=10, type=int,
                        help="Log interval")
    parser.add_argument('--prune', default=True, type=str,
                        help="Pruning the model to reduce weights")

    args = parser.parse_args()


if __name__ == '__main__':
    args = args_define.args
    use_cuda = args.use_cuda and torch.cuda.is_available()
    print("Using CUDA:", use_cuda)

    # make dirs and save args
    if args.resume_path is None:  # build a new dir if training from scratch
        embedding_save_path = args.data_path + '/embeddings_' + strftime("%m%d%H%M%S", localtime())
        os.mkdir(embedding_save_path)

    # resume training using original dir
    else:
        embedding_save_path = args.resume_path
    print("embedding_save_path: ", embedding_save_path)

    with open(embedding_save_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load data splits
    data_split = np.load(args.data_path + '/data_split.npy')

    # Loss
    if args.use_hardest_neg:
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))
    if args.use_dgi:
        loss_fn_dgi = nn.BCEWithLogitsLoss()



    # Metrics
    # Counts average number of nonzero triplets found in minibatches
    metrics = [AverageNonzeroTripletsMetric()]

    # Initially, only use block 0 as training set (with explicit labels)
    train_i = 0

    # Train on initial graph
    # Resume model from the initial block or start the experiment from scratch. Otherwise (to resume from other blocks) skip this step.
    if ((args.resume_path is not None) and (args.resume_point == 0) and ( args.resume_current)) or args.resume_path is None:
        if not args.use_dgi:
            train_indices, indices_to_remove, model = KPGNN.initial_maintain(train_i, 0, data_split, metrics,\
                                                                       embedding_save_path, loss_fn, pruning_allowed=args.prune)
        else:
            train_indices, indices_to_remove, model = KPGNN.initial_maintain(train_i, 0, data_split, metrics,\
                                                                       embedding_save_path, loss_fn, None, loss_fn_dgi, pruning_allowed=args.prune)

    # Initialize the model, train_indices and indices_to_remove to avoid errors
    if args.resume_path is not None:
        model = None
        train_indices = None
        indices_to_remove = []

    # iterate through all blocks
    for i in range(1, data_split.shape[0]):
        # Inference (prediction)
        # Resume model from the previous, i.e., (i-1)th block or continue the new experiment. Otherwise (to resume from other blocks) skip this step.
        if ((args.resume_path is not None) and (args.resume_point == i - 1) and (not args.resume_current)) or args.resume_path is None:
            if not args.use_dgi:
                model = KPGNN.infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, train_indices, model, None,
                              indices_to_remove, pruning_allowed=args.prune)
            else:
                model = KPGNN.infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, train_indices, model,
                              loss_fn_dgi, indices_to_remove, pruning_allowed=args.prune)
        # Maintain
        # Resume model from the current, i.e., ith block or continue the new experiment. Otherwise (to resume from other blocks) skip this step.
        if ((args.resume_path is not None) and (args.resume_point == i) and (args.resume_current)) or args.resume_path is None:
            if i % args.window_size == 0:
                train_i = i
                if not args.use_dgi:
                    train_indices, indices_to_remove, model = KPGNN.initial_maintain(train_i, i, data_split, metrics,
                                                                               embedding_save_path, loss_fn, model, pruning_allowed=args.prune)
                else:
                    train_indices, indices_to_remove, model = KPGNN.initial_maintain(train_i, i, data_split, metrics,
                                                                               embedding_save_path, loss_fn, model,loss_fn_dgi, pruning_allowed=args.prune)

    print("[INFO] TIME TOOK: ",  time.time() -st )
    print('[INFO] Average TESTING NMI: {:.4f}'.format(np.average(average_test_NMI_score)))
    print('[INFO] Average VALIDATION NMI: {:.4f}'.format(np.average(average_val_NMI_score)))

##################### SAIMESE NEURAL NETWORK #########################################

    # import torch.optim as optim
    #
    #
    # # Load data
    # dataset = SocialDataset(args.data_path, 0)
    # features = torch.FloatTensor(dataset.features)
    # labels = torch.LongTensor(dataset.labels)
    #
    # # Initialize Siamese Network and Online Triplet Loss
    # siamese_net = SiameseNet()
    # triplet_selector = CRandomNegativeTripletSelector(margin=0.2)
    # triplet_loss = OnlineSiameseLoss(margin=0.2, triplet_selector=triplet_selector, siamese_net=siamese_net)
    #
    # # Define optimizer and learning rate scheduler
    # optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #
    # # Train network
    # num_epochs = 20
    # batch_size = 32
    # num_batches = int(len(features) / batch_size)
    # for epoch in range(num_epochs):
    #     for batch_idx in range(num_batches):
    #         # Select batch
    #         batch_start = batch_idx * batch_size
    #         batch_end = (batch_idx + 1) * batch_size
    #         batch_features = features[batch_start:batch_end]
    #         batch_labels = labels[batch_start:batch_end]
    #
    #         optimizer.zero_grad()
    #         loss, num_triplets = triplet_loss(batch_features, batch_labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         if batch_idx % 100 == 0:
    #             print('Epoch: {}, Batch: {}, Loss: {:.4f}, Num Triplets: {}'.format(epoch, batch_idx, loss.item(),
    #                                                                                 num_triplets))
    #
    #     # Update learning rate
    #     scheduler.step()
    #








