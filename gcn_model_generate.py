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


class GCNInfer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(GCNInfer, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        skip_start = (0 == n_layers - 1)
        self.layers.append(NodeUpdate(in_feats, n_hidden, activation, test=True, concat=skip_start))
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers - 1)
            self.layers.append(NodeUpdate(n_hidden, n_hidden, activation, test=True, concat=skip_start))
        # output layer
        self.layers.append(NodeUpdate(2 * n_hidden, n_classes, test=True))

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
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
n_hidden = 16
n_layers = 1
self_loop = 'store_true'
weight_decay = 5e-4

# load and preprocess dataset
# data = RedditDataset(self_loop=True)
# data = citegrh.load_cora()
data = citegrh.load_pubmed()

train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
print('训练集：', train_nid)

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

# print("""----Data statistics------'
#   #Edges %d
#   #Classes %d
#   #Train samples %d
#   #Val samples %d
#   #Test samples %d""" %
#       (n_edges, n_classes,
#        n_train_samples,
#        n_val_samples,
#        n_test_samples))

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


'''子图划分'''
g.ndata['features'] = features
g.ndata['norm'] = norm

N = 20  # 30 parties
node_list = list(range(19717))  # 点集
sub_node_list = []  # 子点集
sub_labels = []  # 子label
sub_train_id = []  # 子点集上的子训练集
for i in range(N):
    sub_node_list.append(node_list[i::N])
    sub_labels.append(labels[i::N])

for i in range(N):
    sub_train_id.append([])

for i in range(N):
    for j in range(len(sub_node_list[i])):
        if sub_node_list[i][j] in train_nid:
            sub_train_id[i].append(sub_node_list[i][j])

'''##########     训练的总节点      ############'''
def local_train(b, send_index):
    new_list = []
    new_labels = []
    for i in range(len(send_index)):
        new_list.extend(sub_node_list[send_index[i]])
        new_labels.extend(sub_labels[send_index[i]].tolist())






# 子图所需列表
new_list = sub_node_list[0] + sub_node_list[1] + sub_node_list[2] + sub_node_list[3] + sub_node_list[4] + \
             sub_node_list[5]
# 子图所需label
new_labels = sub_labels[0].tolist() + sub_labels[1].tolist() + sub_labels[2].tolist() + sub_labels[3].tolist() + \
             sub_labels[4].tolist() + sub_labels[5].tolist()
new_labels = torch.LongTensor(new_labels)
new_labels.cuda()
# 生成子图
new_g = g.subgraph(new_list)
new_g.copy_from_parent()
new_g.readonly()

# 子点集上的子训练集映射到子图上，即确认子图上的训练节点
new_train_id = []
'''################  训练方  #############'''
for i in range(len(sub_train_id[0])):  # 训练方的train_id在子图上的标号
    new_train_id.append(int(new_g.map_to_subgraph_nid(sub_train_id[0][i])))
new_train_id = np.array(new_train_id)


'''training'''
model = GCNSampling(in_feats, n_hidden, n_classes, n_layers, F.relu, dropout)

if cuda:
    model.cuda()
'''保存模型'''
eval('torch.save(model.state_dict(), \'model'+str(1)+'.pkl\')')


loss_fcn = nn.CrossEntropyLoss()
infer_model = GCNInfer(in_feats, n_hidden, n_classes, n_layers, F.relu)

if cuda:
    infer_model.cuda()

# use optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)

# initialize graph
dur = []
for epoch in range(1):
    for nf in dgl.contrib.sampling.NeighborSampler(new_g, batch_size,
                                                   num_neighbors,
                                                   neighbor_type='in',
                                                   shuffle=True,
                                                   num_workers=32,
                                                   num_hops=n_layers + 1,
                                                   seed_nodes=new_train_id):
        nf.copy_from_parent()
        model.train()
        print('层数：', nf.num_layers)
        print('第一层节点数：', nf.layer_size(0))
        print('第二层节点数：', nf.layer_size(1))
        print('第三层节点数：', nf.layer_size(2))
        # forward
        pred = model(nf)
        batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)

        batch_labels = new_labels.cuda()[batch_nids]
        loss = loss_fcn(pred, batch_labels)
        print('loss:'+str(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for infer_param, param in zip(infer_model.parameters(), model.parameters()):
        infer_param.data.copy_(param.data)

    num_acc = 0.

    for nf in dgl.contrib.sampling.NeighborSampler(g, test_batch_size,
                                                   g.number_of_nodes(),
                                                   neighbor_type='in',
                                                   num_workers=32,
                                                   num_hops=n_layers + 1,
                                                   seed_nodes=test_nid):
        nf.copy_from_parent()
        infer_model.eval()

        with torch.no_grad():
            pred = infer_model(nf)
            batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
            batch_labels = labels[batch_nids]
            num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()
    print("Test Accuracy {:.4f}".format(num_acc / n_test_samples))

# '''保存模型'''
# eval('torch.save(model.state_dict(), \'model'+str(0)+'.pkl\')')
