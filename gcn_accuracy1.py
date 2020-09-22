
 
"""分成20方子图后，某一方的有标签点只在部分子图组成的大图上采样训练，相当于忽略部分子图的边连接"""
''''1. 导入自己模型
    2. 训练自己的模型
    3. 导入其他人模型并平均
    4. 保存自己的模型'''
'''求出准确率'''
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

'''create GCN model'''


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
        self.layers.append(NodeUpdate(in_feats, n_hidden, activation))
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers - 1)
            self.layers.append(NodeUpdate(n_hidden, n_hidden, activation))
        # output layer
        self.layers.append(NodeUpdate(n_hidden, n_classes))

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')

            if self.dropout:
                h = self.dropout(h)

            if i == 0:
                nf.layers[i].data['h'] = h

                sum_nf = []  # 替换
                self_nodes_nf = []  # 自己nf上的ID
                sum_nf1 = nf.layer_parent_nid(0).numpy().tolist()
                sum_nf2 = nf.layer_parent_nid(1).numpy().tolist()
                sum_nf3 = nf.layer_parent_nid(2).numpy().tolist()
                sum_nf = sum_nf + sum_nf1 + sum_nf2 + sum_nf3  # 总共用到的节点的ID, 是在new_g中的ID
                for index8 in range(len(sum_nf1)):
                    if sum_nf1[index8] in (sub_node_id[0]):  # 如果第一层的点是自己家的
                        self_nodes_nf.append(nf.map_from_parent_nid(0, sum_nf1[index8]).item())
                nf.copy_to_parent()  # 赋值到new_g中


                for index8 in range(len(sum_nf1)):
                    if sum_nf1[index8] not in (sub_node_id[0]):                         # 不是自己方
                        for n2 in range(len(send_index1)):
                            map_g = new_g.parent_nid[sum_nf1[index8]]  # 在g中的ID
                            if map_g in sub_node_list[send_index1[n2]]:  # send_index1[n2] 方
                                fc.weight = nn.Parameter(Model[send_index1[in2]]['layers.0.linear.weight'])
                                fc.bias = nn.Parameter(Model[send_index1[in2]]['layers.0.linear.bias'])
                                new_g.nodes[sum_nf1[index8]].data['activation'] = \
                                    fc(g.nodes[map_g].data['features'])

                nf.copy_from_parent()

                nf.apply_layer(0, layer, v=self_nodes_nf)  # layer0 降维

                print(i)
                print(layer)

            else:
                nf.layers[i].data['h'] = h
                nf.block_compute(i,
                                 fn.copy_src(src='h', out='m'),
                                 lambda node: {'h': node.mailbox['m'].mean(dim=1)},
                                 layer)
                print(i)
                print(layer)
            # print(i)
            # print(layer)
            # print(len(nf.layers[i].data['h'][0]))  # 该层的h
            # print(len(nf.layers[-1].data['activation'][0]))

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
        self.layers.append(NodeUpdate(in_feats, n_hidden, activation, test=True))
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers - 1)
            self.layers.append(NodeUpdate(n_hidden, n_hidden, activation, test=True))
        # output layer
        self.layers.append(NodeUpdate(n_hidden, n_classes, test=True))

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
batch_size = 1500
test_batch_size = 100
num_neighbors = 3
n_hidden = 128
n_layers = 1
self_loop = 'store_true'
weight_decay = 5e-4

# load and preprocess dataset
data = RedditDataset(self_loop=True)
# data = citegrh.load_cora()
# data = citegrh.load_pubmed()

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
node_list = list(range(232965))  # 点集
# node_list = list(range(19717))  # 点集
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

'''加载所有的初始化模型，作为各方model的全局模型'''
Model = []
for index7 in range(N):
    Model.append(torch.load('model1.pkl'))  # 全局模型包含每方的模型，且全都对他们进行相同的初始化

'''##########     训练的总节点      ############'''
send_index1 = [1, 2, ]
b = 0

fc = nn.Linear(in_features=in_feats, out_features=n_hidden , bias=True)
# fc.weight = nn.Parameter(Model[0]['layers.0.linear.weight'])
# fc.bias = nn.Parameter(Model[0]['layers.0.linear.bias'])
# fc(g.nodes[0].data['features'])

# 发送方的节点添加特征
# for in2 in range(len(send_index1)):  # 第 send_index[in2] 方
#     for in3 in range(len(sub_node_list[send_index1[in2]])):  # 子节点列表
#         fc.weight = nn.Parameter(Model[send_index1[in2]]['layers.0.linear.weight'])
#         fc.bias = nn.Parameter(Model[send_index1[in2]]['layers.0.linear.bias'])
#         g.nodes[sub_node_list[send_index1[in2]][in3]].data['fff'] = \
#             fc(g.nodes[sub_node_list[send_index1[in2]][in3]].data['features'])
#


send_index = send_index1
send_index.append(b)
new_list = []
new_labels = []
for in2 in range(len(send_index)):
    new_list.extend(sub_node_list[send_index[in2]])
    new_labels.extend(sub_labels[send_index[in2]].tolist())

new_labels = torch.LongTensor(new_labels)
new_labels.cuda()
# 生成子图
new_g = g.subgraph(new_list)
new_g.copy_from_parent()
new_g.readonly()

# 子列表在子图中的ID
sub_node_id = []
sub_node_id.append(new_g.map_to_subgraph_nid(sub_node_list[b]))  # 自己方节点在子图中的ID



# 子点集上的子训练集映射到子图上，即确认子图上的训练节点
new_train_id = []
'''################  训练方  #############'''
for in3 in range(len(sub_train_id[b])):  # 训练方的train_id在子图上的标号
    new_train_id.append(int(new_g.map_to_subgraph_nid(sub_train_id[b][in3])))
new_train_id = np.array(new_train_id)

# '''training'''
model = GCNSampling(in_feats, n_hidden, n_classes, n_layers, F.relu, dropout)
#
if cuda:
    model.cuda()
# '''加载模型'''
# model.load_state_dict(Model[b])

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
                                                   num_workers=64,
                                                   num_hops=n_layers + 1,
                                                   seed_nodes=new_train_id):
        nf.copy_from_parent()
        model.train()
        # forward
        print('-------------------')
        timea = time.time()
        pred = model(nf)

        batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
        batch_labels = new_labels.cuda()[batch_nids]
        loss = loss_fcn(pred, batch_labels)
        timeb = time.time()
        print(timeb - timea)
        timec = time.time()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        timed = time.time()
        print(timed-timec)
        print('####################')

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
#
# '''平均模型'''
# p1 = model.state_dict()
# p2 = Model[2]
# for key, value in p2.items():
#     for index8 in range(len(send_index1)):
#         p1[key] = p1[key] + Model[send_index1[index8]][key]
#
#     p1[key] = p1[key] / (len(send_index) + 1)
#
# '''保存模型'''
# Model[b] = p1


