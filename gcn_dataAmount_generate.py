'''生成20 方之间的数据交互，num_list,只包含一轮，以后每轮应当一样'''
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


def random_int_list1(start, stop, length, se):  # 生成随机整数数组，使用相同的seed则生成相同的数组
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        seed(se + i)  # 添加随机种子，保持每次生成的随机数不变
        random_list.append(random.randint(start, stop))
    return random_list


# data = citegrh.load_pubmed()
# data = citegrh.load_cora()
data = RedditDataset(self_loop = True)

features = torch.FloatTensor(data.features)
labels = torch.LongTensor(data.labels)

train_mask = torch.BoolTensor(data.train_mask)
test_mask = torch.BoolTensor(data.test_mask)

train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
test_nid = np.nonzero(data.test_mask)[0].astype(np.int64)

g = data.graph
g = DGLGraph(g)
n_classes = data.num_labels
g.readonly()

norm = 1. / g.in_degrees().float().unsqueeze(1)
g.ndata['features'] = features
g.ndata['norm'] = norm

N = 20 # 20 parties
Epo = 1  # ages

node_list = list(range(232965))  # 点集

sub_node_list = []
for i in range(N):
    sub_node_list.append(node_list[i::N])

sub_train_id = []
for i in range(N):
    sub_train_id.append([])

for i in range(N):
    for j in range(len(sub_node_list[i])):
        if sub_node_list[i][j] in train_nid:
            sub_train_id[i].append(sub_node_list[i][j])

'''create GCN model'''
# num_list = []
# for i in range(N):
#     num_list.append([])

# num_neighbors = random_int_list1(2, 30, Epo, 2)  # 5_10 要传输数据量的随机变化，20个随机数,相当于传输20次，每次数据量不一样
for n in range(N):
    locals()['num_list' + str(n)] = []
    for i in range(N):  # ch num_list is a two dimention list
        eval('num_list' + str(n) + '.append([])')
    print('Sample party:' + str(n))
    for epoch in range(Epo):
        sum_nf = []
        for nf in dgl.contrib.sampling.NeighborSampler(g, 10,
                                                       # num_neighbors[epoch],
                                                       25,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_workers=32,
                                                       num_hops=2,
                                                       # seed_nodes=np.array(sub_train_id[n])):
                                                       seed_nodes=np.array(sub_node_list[n])):
            nf.copy_from_parent()
            sum_nf1 = nf.layer_parent_nid(0).numpy().tolist()
            sum_nf2 = nf.layer_parent_nid(1).numpy().tolist()
            sum_nf3 = nf.layer_parent_nid(2).numpy().tolist()
            sum_nf = sum_nf + sum_nf1 + sum_nf2 + sum_nf3  # 总共用到的节点的ID

        count = np.zeros(N)
        for i in range(len(sum_nf)):
            for j in range(N):
                if sum_nf[i] in sub_node_list[j]:
                    count[j] = count[j] + 1

        print('The ' + str(epoch) + 'th epoch')
        for j in range(N):
            eval('num_list' + str(n) + '[j].append(count[j])/670')  # 点的个数

name = []
for i in range(Epo):
    name.append(i)
for i in range(N):
    save_data = eval('pd.DataFrame(columns = name, data = num_list' + str(i) + ')')
    eval('save_data.to_csv(\'./num_list' + str(i) + '.csv\')')
