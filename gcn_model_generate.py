'''为20方初始化相同的模型并保存模型'''
'''只运行一次'''
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import citation_graph as citegrh
from dgl.data import RedditDataset
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
from random import seed
import sys
import pandas as pd


class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, test=False, concat=False):
        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.concat = concat
        self.test = test

    def forward(self, node):
        h = node.data['h']
        if self.test:
            h = h * node.data['norm']
        h = self.linear(h)
        # skip connection
        if self.concat:
            h = torch.cat((h, self.activation(h)), dim=1)
        elif self.activation:
            h = self.activation(h)
        return {'activation': h}


# class GCNSampling(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  n_hidden,
#                  n_classes,
#                  n_layers,
#                  activation,
#                  dropout):
#         super(GCNSampling, self).__init__()
#         self.n_layers = n_layers
#         if dropout != 0:
#             self.dropout = nn.Dropout(p=dropout)
#         else:
#             self.dropout = None
#         self.layers = nn.ModuleList()
#         # input layer
#         skip_start = (0 == n_layers - 1)
#         self.layers.append(NodeUpdate(in_feats, n_hidden, activation))
#         # hidden layers
#         for i in range(1, n_layers):
#             skip_start = (i == n_layers - 1)
#             self.layers.append(NodeUpdate(n_hidden, n_hidden, activation))
#         # output layer
#         self.layers.append(NodeUpdate(n_hidden, n_classes))
#
#     def forward(self, nf):
#         nf.layers[0].data['activation'] = nf.layers[0].data['features']
#
#         for i, layer in enumerate(self.layers):
#             h = nf.layers[i].data.pop('activation')
#             if self.dropout:
#                 h = self.dropout(h)
#             nf.layers[i].data['h'] = h
#             nf.block_compute(i, fn.copy_src(src='h', out='m'),
#                              lambda node: {'h': node.mailbox['m'].mean(dim=1)},
#                              layer)
#
#         h = nf.layers[-1].data.pop('activation')
#         return h


class GCNSampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCNSampling, self).__init__()
        self.n_layers = n_layers
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()
        # input layer
        skip_start = (0 == n_layers - 1)
        self.layers.append(NodeUpdate(in_feats, n_hidden, activation, concat=skip_start))
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers - 1)
            self.layers.append(NodeUpdate(n_hidden, n_hidden, activation, concat=skip_start))
        # output layer
        self.layers.append(NodeUpdate(2 * n_hidden, n_classes))

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i, fn.copy_src(src='h', out='m'),
                             lambda node: {'h': node.mailbox['m'].mean(dim=1)},
                             layer)

        h = nf.layers[-1].data.pop('activation')
        return h


'''parameter setting '''
dropout = 0.5
gpu = 0
lr = 3e-2
n_epochs = 100
batch_size = 10
test_batch_size = 100
num_neighbors = 25
n_hidden = 128
n_layers = 1
self_loop = 'store_true'
weight_decay = 5e-4

# load and preprocess dataset
data = RedditDataset(self_loop=True)
# data = citegrh.load_cora()
# data = citegrh.load_pubmed()

train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)

test_nid = np.nonzero(data.test_mask)[0].astype(np.int64)

features = torch.FloatTensor(data.features)
labels = torch.LongTensor(data.labels)
if hasattr(torch, 'BoolTensor'):
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)
else:
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
in_feats = features.shape[1]
n_classes = data.num_labels
n_edges = data.graph.number_of_edges()

n_train_samples = train_mask.int().sum().item()
n_val_samples = val_mask.int().sum().item()
n_test_samples = test_mask.int().sum().item()

# create GCN model
g = data.graph
g = DGLGraph(g)
g.readonly()
norm = 1. / g.in_degrees().float().unsqueeze(1)

if gpu < 0:
    cuda = False
else:
    cuda = True
    torch.cuda.set_device(gpu)
    features = features.cuda()
    labels = labels.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()
    norm = norm.cuda()

'''training'''
model = GCNSampling(in_feats, n_hidden, n_classes, n_layers, F.relu, dropout)
print(in_feats)
if cuda:
    model.cuda()
'''保存模型'''
eval('torch.save(model.state_dict(), \'model' + str(1) + '.pkl\')')
