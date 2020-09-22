import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
from random import seed
import numpy as np
import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# from gcn_sampling import *

N = 3
Epo = 30


def random_int_list(start, stop, length):  # 生成随机整数数组,每次调用该函数生成的数组不变
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        seed(i)  # 添加随机种子，保持每次生成的随机数不变
        random_list.append(random.randint(start, stop))
    return random_list


def random_int_list1(start, stop, length, se):  # 生成随机整数数组，使用相同的seed则生成相同的数组
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        seed(se + i)  # 添加随机种子，保持每次生成的随机数不变
        random_list.append(random.randint(start, stop))
    return random_list


'''###########################添加拓扑逻辑关系####################'''
'''构建拓扑图'''
data = pd.read_csv("50_topology.csv")
data = data.values.tolist()
edg1 = []
edg2 = []

# 读取数据
for n in range(N):
    locals()['num_list' + str(n)] = []
    for i in range(N):
        eval('num_list' + str(n) + '.append([])')
for i in range(N):
    mi = eval('pd.read_csv(\'./num_list' + str(i) + '.csv\')')
    mi = mi.values.tolist()
    mi = list(map(lambda x: x[1:], mi))
    for j in range(N):
        for j1 in range(len(mi[j])):
            eval('num_list' + str(i) + '[j].append(mi[j][j1])')

note_set = [82, 86, 10, 6, 28, 32, 24, 26, 14, 18, 13, 16, 17, 37, 35, 42, 89, 94, 91, 93, 90, 95,
            76, 83, 9, 7, 8, 0, 5, 30, 22, 3, 15, 36, 11, 33, 21, 92, 87]

# note_set = [82, 86,
#             10, 6, 4, 2, 5, 10,
#             28, 32, 24, 26, 27,
#             14, 18, 13, 16,
#             37, 35, 42, 40, 39,
#             89, 94, 91, 93,
#             81, 79, 77, 85, 80,
#             76, 83, 9, 7, 8, 0, 5, 30, 22, 3, 15, 36, 11, 33, 21, 92, 87]

for i in range(len(data)):
    a, b = data[i]
    if a in note_set and b in note_set:
        edg1.append([a, b])  # 选五十个点
#
for i in range(len(edg1)):  # 将五十个点赋值为0-49，并保存他们的关系
    a, b = edg1[i]
    for j in range(len(note_set)):
        if a == note_set[j]:
            edg1[i][0] = j
        if b == note_set[j]:
            edg1[i][1] = j

for i in range(len(edg1)):  # 单向连接改为双向连接
    a, b = edg1[i]
    edg2.append([b, a])

G = nx.MultiDiGraph()  # 有多重边有向图
G.add_edges_from(edg1)  # 添加边
G.add_edges_from(edg2)  # 添加边

# color_map = []
# for node in G:
#     if node < N:
#         color_map.append('blue')
#     else:
#         color_map.append('green')
# nx.draw(G, node_color=color_map, with_labels=True)

# nx.draw(G, with_labels=True)

'''设置路径限制带宽'''
paths = edg1 + edg2
times = np.zeros(len(paths))  # fair share 使每条路径被占用的次数

bandwidth = []
for i in range(100):  # 变动1000次，每次里面有80个路径的数值
    bandwidth.append(random_int_list(123, 124, (len(paths))))

for i in range(len(paths)):
    for j in range(100):
        a, b = paths[i]
        G[a][b][0]['Capacity'] = bandwidth[0][i]  # 规定每条子路径的带宽限制

'''设置各方传输路径和传输数据'''
short_paths = dict(nx.all_pairs_shortest_path(G))  # 计算graph两两节点之间的最短路径

for i in range(N):  # 定义路径动态变量
    for j in range(N):
        if i != j:
            globals()['paths_' + str(i) + '_' + str(j)] = []

for i in range(N):  # 每两个点之间的路径
    a = short_paths.get(i)
    for j in a.keys():
        if i != j & j < N:
            b = []
            for x in range(len(a[j]) - 1):
                b.append([a[j][x], a[j][x + 1]])
            for y in range(len(b)):
                eval('paths_' + str(i) + '_' + str(j) + '.append(b[y])')

for i in range(N):  # 定义流量变量
    for j in range(N):
        if i != j:
            globals()['D_' + str(i) + '_' + str(j)] = []

for i in range(N):  # 定义acc变量, 保存i方每轮的acc
    globals()['Acc' + str(i)] = []

for i in range(N):  # 定义Loss变量
    globals()['Loss' + str(i)] = []

for i in range(N):  # 定义time变量,保存i方每轮的time_cost
    globals()['Time_round' + str(i)] = []

Start_time = []
for i in range(N):  # 定义time变量,保存i方每轮的time_cost
    Start_time.append(0)

for i in range(N):  # 赋值流量变量
    a = eval('num_list' + str(i))
    for j in range(N):
        if i != j:
            b = []
            for x in range(len(a[j])):
                b.append(a[j][x])
            for y in range(Epo):  # 后面100个都和第一个同样
                eval('D_' + str(j) + '_' + str(i) + '.append(round(b[0]))')  # 变大100倍
            eval('D_' + str(j) + '_' + str(i))[0] = 0


def t_cost(b1, send_index):
    a = eval('num_list' + str(b1))
    sum_time = 0
    for in1 in range(len(send_index)):
        sum_time = sum_time + a[send_index[in1]][0]
    sum_time = round(sum_time * 10)  # 将计算时间和传输方传输的数据量相关联,是其10倍
    return sum_time

# t_cost = [3, 4, 3, 5, 6, 8, 9, 10, 3, 4, 8, 5, 3, 4, 2, 4, 6, 7, 8, 9]
