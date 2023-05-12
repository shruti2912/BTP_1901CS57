from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
GAT INITIAL CODE
'''
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=False):
        super(GATLayer, self).__init__()
        # equation (1) reference: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, nf, layer_id):
        h = nf.layers[layer_id].data['h']
        # equation (1)
        z = self.fc(h)
        nf.layers[layer_id].data['z'] = z
        # print("test test test")
        A = nf.layer_parent_nid(layer_id)
        # print(A)
        # print(A.shape)
        A = A.unsqueeze(-1)
        B = nf.layer_parent_nid(layer_id + 1)
        # print(B)
        # print(B.shape)
        B = B.unsqueeze(0)

        _, indices = torch.topk((A == B).int(), 1, 0)
        # print(indices)
        # print(indices.shape)
        # indices = np.asarray(indices)
        indices = indices.cpu().data.numpy()

        nf.layers[layer_id + 1].data['z'] = z[indices]
        # print(nf.layers[layer_id+1].data['z'].shape)
        # equation (2)
        nf.apply_block(layer_id, self.edge_attention)
        # equation (3) & (4)
        nf.block_compute(layer_id,  # block_id _ The block to run the computation.
                         self.message_func,  # Message function on the edges.
                         self.reduce_func)  # Reduce function on the node.

        nf.layers[layer_id].data.pop('z')
        nf.layers[layer_id + 1].data.pop('z')

        if self.use_residual:
            return z[indices] + nf.layers[layer_id + 1].data['h']  # residual connection
        return nf.layers[layer_id + 1].data['h']

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual=False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, use_residual))
        self.merge = merge

    def forward(self, nf, layer_id):
        head_outs = [attn_head(nf, layer_id) for attn_head in self.heads]
        if self.merge == 'cat':
            # concatenate the output of each head along the feature dimension
            h = torch.cat(head_outs, dim=1)
        else:
            # merge the output of each head by taking the mean
            h = torch.mean(torch.stack(head_outs), dim=0)
        return h
#
#
# class GAT(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
#         super(GAT, self).__init__()
#         self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
#         # Be aware that the input dimension is hidden_dim*num_heads since
#         # multiple head outputs are concatenated together. Also, only
#         # one attention head in the output layer.
#         self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)
#
#     def forward(self, nf, corrupt=False):
#         features = nf.layers[0].data['features']
#         if corrupt:
#             nf.layers[0].data['h'] = features[torch.randperm(features.size()[0])]
#         else:
#             nf.layers[0].data['h'] = features
#         h = self.layer1(nf, 0)
#         h = F.elu(h)
#         # print(h.shape)
#         nf.layers[1].data['h'] = h
#         h = self.layer2(nf, 1)
#
#         return h



'''
GAT CUSTOM CODE:
'''
from dgl.nn.pytorch.conv import GINConv
from torch.nn import Sequential, Linear, ReLU, GELU, ELU


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_layers=10, dropout=0.5, use_residual=False,
                 use_gin=False):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(MultiHeadGATLayer(hidden_dim * num_heads, hidden_dim, num_heads, 'cat', use_residual))

        # Output layer
        self.layers.append(MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual))

        # Optional GIN layer
        if use_gin:
            nn_gin = Sequential(Linear(hidden_dim * num_heads, hidden_dim * num_heads),
                                ReLU(),
                                Linear(hidden_dim * num_heads, hidden_dim * num_heads))
            self.gin = GINConv(nn_gin, 'sum')

        self.dropout = nn.Dropout(dropout)

    def forward(self, nf, corrupt=False):
        features = nf.layers[0].data['features']
        if corrupt:
            nf.layers[0].data['h'] = features[torch.randperm(features.size()[0])]
        else:
            nf.layers[0].data['h'] = features

        for i, layer in enumerate(self.layers):
            h = layer(nf, i)
            if i < len(self.layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)
                nf.layers[i + 1].data['h'] = h
                if hasattr(self, 'gin'):
                    nf.layers[i + 1].data['h'] = self.gin(nf.layers[i + 1].data['h'])

        return h


# Applies an average on seq, of shape (nodes, features)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)
        # print("testing, shape of logits: ", logits.size())
        return logits


class DGI(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(DGI, self).__init__()
        self.gat = GAT(in_dim, hidden_dim, out_dim, num_heads, use_residual)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim)

    def forward(self, nf):
        h_1 = self.gat(nf, False)
        c = self.read(h_1)
        c = self.sigm(c)
        h_2 = self.gat(nf, True)
        ret = self.disc(c, h_1, h_2)
        return h_1, ret

    # Detach the return variables
    def embed(self, nf):
        h_1 = self.gat(nf, False)
        return h_1.detach()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)







def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                              negative_selection_fn=hardest_negative,
                                                                                              cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                             negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)


############################################### Siamese Neural Network ####################################################################

# import torch
# import torch.nn as nn
# import numpy as np
# import random
#
#
# class SiameseNet(nn.Module):
#     def __init__(self):
#         super(SiameseNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(64 * 8 * 8, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#
#     def forward(self, x):
#         x = x.view(-1, 3, 32,302)
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(-1, 64 * 8 * 8)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
#
# class CTripletSelector:
#     def __init__(self):
#         pass
#
#     def get_triplets(self, inputs, labels):
#         raise NotImplementedError
#
# class CRandomNegativeTripletSelector(CTripletSelector):
#     def __init__(self, margin):
#         super(CRandomNegativeTripletSelector, self).__init__()
#         self.margin = margin
#
#     def get_triplets(self, inputs, labels):
#         batch_size = inputs.size(0)
#         anchor_idxs = torch.randint(0, batch_size, (batch_size,))
#         positive_idxs = anchor_idxs
#         while torch.any(positive_idxs == anchor_idxs):
#             positive_idxs = torch.randint(0, batch_size, (batch_size,))
#         negative_idxs = torch.zeros_like(anchor_idxs)
#         for i, anchor_idx in enumerate(anchor_idxs):
#             label = labels[anchor_idx]
#             label_idxs = torch.nonzero(labels == label).squeeze()
#             negative_mask = torch.ones(batch_size, dtype=torch.float32)
#             negative_mask[label_idxs] = 0
#             negative_mask[anchor_idx] = 0
#             valid_negative_idxs = torch.nonzero(negative_mask).squeeze()
#             negative_idx = torch.randint(0, valid_negative_idxs.size(0), (1,))
#             negative_idxs[i] = valid_negative_idxs[negative_idx]
#         return inputs[anchor_idxs], inputs[positive_idxs], inputs[negative_idxs]
#
#
# class CPairSelector(CTripletSelector):
#     def __init__(self, num_pairs_per_class=5, balanced=True):
#         self.num_pairs_per_class = num_pairs_per_class
#         self.balanced = balanced
#
#     def _generate_positive_pairs(self, label_indices):
#         positive_pairs = list(combinations(label_indices, 2))
#         return positive_pairs
#
#     def _generate_negative_pairs(self, label_indices, negative_indices):
#         negative_pairs = []
#         for idx in label_indices:
#             for _ in range(self.num_pairs_per_class):
#                 neg_idx = random.choice(negative_indices)
#                 negative_pairs.append((idx, neg_idx))
#         return negative_pairs
#
#     def get_pairs(self, embeddings, labels):
#         labels = labels.cpu().data.numpy()
#         pairs = []
#         labels_list = []
#
#         unique_labels = set(labels)
#         for label in unique_labels:
#             label_mask = (labels == label)
#             label_indices = np.where(label_mask)[0]
#             negative_indices = np.where(np.logical_not(label_mask))[0]
#
#             if len(label_indices) < 2:
#                 continue
#
#             np.random.shuffle(label_indices)
#             np.random.shuffle(negative_indices)
#
#             if self.balanced:
#                 positive_pairs = self._generate_positive_pairs(label_indices)[:self.num_pairs_per_class]
#                 negative_pairs = self._generate_negative_pairs(label_indices[:self.num_pairs_per_class], negative_indices[:self.num_pairs_per_class])
#             else:
#                 positive_pairs = self._generate_positive_pairs(label_indices)
#                 negative_pairs = self._generate_negative_pairs(label_indices, negative_indices)
#
#             pairs.extend(positive_pairs)
#             pairs.extend(negative_pairs)
#             labels_list.extend([1] * len(positive_pairs))
#             labels_list.extend([0] * len(negative_pairs))
#
#         pairs = np.array(pairs)
#         labels_list = np.array(labels_list)
#
#         # Shuffle pairs and labels together
#         shuffle_indices = np.random.permutation(len(pairs))
#         pairs = pairs[shuffle_indices]
#         labels_list = labels_list[shuffle_indices]
#
#         return torch.LongTensor(pairs), torch.LongTensor(labels_list)
#
#
#
#
#
# class OnlineSiameseLoss(nn.Module):
#     def __init__(self, margin, triplet_selector, siamese_net):
#         super(OnlineSiameseLoss, self).__init__()
#         self.margin = margin
#         self.triplet_selector = triplet_selector
#         self.siamese_net = siamese_net
#
#     def forward(self, inputs, target):
#         anchors, positives, negatives = self.triplet_selector.get_triplets(inputs, target)
#
#         anchor_embeddings = self.siamese_net(anchors)
#         positive_embeddings = self.siamese_net(positives)
#         negative_embeddings = self.siamese_net(negatives)
#
#         ap_distances = (anchor_embeddings - positive_embeddings).pow(2).sum(1)
#         an_distances = (anchor_embeddings - negative_embeddings).pow(2).sum(1)
#         losses = F.relu(ap_distances - an_distances + self.margin)
#         return losses.mean(), len(losses)
#
#
# class OnlinePairLoss(nn.Module):
#     def __init__(self, margin, pair_selector):
#         super(OnlinePairLoss, self).__init__()
#         self.margin = margin
#         self.pair_selector = pair_selector
#
#     def forward(self, embeddings1, embeddings2, labels):
#         pairs, pair_labels = self.pair_selector.get_pairs(embeddings1, labels)
#
#         if embeddings1.is_cuda:
#             pairs = pairs.cuda()
#             pair_labels = pair_labels.cuda()
#
#         print("---------------------------- Pairs size:", pairs.size())
#         print("---------------------------- Embeddings1 size:", embeddings1.size())
#         print("---------------------------- Embeddings2 size:", embeddings2.size())
#
#         embeddings1 = embeddings1[pairs[:, 0]]
#         embeddings2 = embeddings2[pairs[:, 1]]
#         pair_labels = pair_labels.float()
#
#         euclidean_distances = (embeddings1 - embeddings2).pow(2).sum(1).sqrt()
#         loss = 0.5 * (pair_labels * euclidean_distances.pow(2) +
#                       (1 - pair_labels) * F.relu(self.margin - euclidean_distances).pow(2))
#
#         return loss.mean(), len(pairs)
#
