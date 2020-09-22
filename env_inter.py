import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
from random import seed
import numpy as np
from env_topology import *
# from gcn_accuracy import *


# 该路径最小子路径
def find_less_subpath(path):
    mx = 2000  # 取一个大数
    less_flow = [0, 0, 0]
    for i in range(len(path)):  # 所有的子路径
        capa = G[path[i][0]][path[i][1]][0]['Capacity']  # 第i个子路径带宽
        if capa < mx:
            less_flow[0] = path[i][0]  # 起点
            less_flow[1] = path[i][1]  # 终点
            less_flow[2] = capa  # 最小带宽
            mx = capa
    return less_flow[2]


# 该路径所有子路径capacity
def find_all_capacity(path):
    capac = []
    for i in range(len(path)):  # 所有的子路径
        capa = G[path[i][0]][path[i][1]][0]['Capacity']  # 第i个子路径带宽
        capac.append(capa)
    return capac


# 生成新的流的优先级和他们所在的层数
def regeneral(prio, state):
    new_prio = []
    new_state = []
    for index3 in range(len(prio)):  # 6个数据流，0，1，2，3，4，5
        flow = prio[index3].split('_')
        a = int(flow[1])  # 发送方
        b = int(flow[2])  # 接收方
        c = list(set(nodes).difference(set([a, b])))
        if eval('D_' + str(b) + '_' + str(a))[state[index3] - 1] != 0:
            print('现在优先的是数据流' + str(a) + '_' + str(b) + '的第' + str(state[index3]) + '层,' + \
                  '但是前一层的' + str(b) + '_' + str(a) + '的数据流没传完')
        elif eval('D_' + str(a) + '_' + str(b))[state[index3] - 1] != 0:
            print('现在优先的是数据流' + str(a) + '_' + str(b) + '的第' + str(state[index3]) + '层,' + \
                  '但是前一层的' + str(a) + '_' + str(b) + '的数据流没传完')
        else:
            for index2 in range(len(c)):
                cc = c[index2]
                if eval('D_' + str(cc) + '_' + str(a))[state[index3] - 1] != 0:
                    print('现在优先的是数据流' + str(a) + '_' + str(b) + '的第' + str(state[index3]) + '层,' + \
                          '但是前一层的' + str(cc) + '_' + str(a) + '的数据流没传完')
                    break
            if index2 == len(c) - 1:
                new_prio.append(prio[index3])
                new_state.append(state[index3])
    return new_prio, new_state


# 计算当前情况下各方时间花费和最小带宽限制
def gen_T(D_prio, new_state):
    '''计算预计的T[0]'''
    flow = D_prio[0].split('_')
    a = int(flow[1])  # 发送方
    b = int(flow[2])  # 接收方
    T = []  # T是按现在的优先度，和流的限制，各方预计传完的时间，目的是找出最小的那个
    capa = []  # capa 是按现在的有限度和流的限制，各方路径限制最小的带宽
    path = eval('paths_' + str(a) + '_' + str(b))  # 发送方和接收方之间的传播路径
    capa.append(find_less_subpath(path))  # 该数据流路径上有最小带宽限制的子路径
    if eval(D_prio[0])[new_state[0]] >= 0:  # 发送方到接收方的数据大于0
        T.append(round(eval(D_prio[0])[new_state[0]] / capa[0], 2))  # 最高优先流所花费时间
        # eval(D_prio[i])[state[i]] = 0  # 该数据流传输完毕

    '''计算预计的T[j]'''
    if len(D_prio) > 1:
        for j in range(1, len(D_prio)):
            flow1 = D_prio[j].split('_')
            a1 = int(flow1[1])  # 发送方
            b1 = int(flow1[2])  # 接收方
            path1 = eval('paths_' + str(a1) + '_' + str(b1))  # 第二个流的路径
            common_path = [val for val in path if val in path1]  # 两条路径的交集
            if common_path == []:  # 和上一个无交集
                capa.append(find_less_subpath(path1))  # 第二个流最小路径带宽
                T.append(round(eval(D_prio[j])[new_state[j]] / capa[j], 2))  # 完成时间
            else:  # 和上一个有交集
                capa1 = max(find_all_capacity(common_path))  # 交集路径中每条子路径的带宽 中最大的
                if capa1 > capa[j - 1]:
                    capa.append(capa1 - capa[j - 1])  # 因为前一个流每次只以最小的带宽来传数据，在其他路径可能会占不满
                    T.append(round(eval(D_prio[j])[new_state[j]] / (capa[j]), 2))  # 完成时间
                else:
                    T.append(1000000)  # 完全堵塞，完成时间取一个极大数
                    capa.append(0)
            c_path = path + path1
            path = [(c_path[i]) for i in range(0, len(c_path)) if c_path[i] not in c_path[:i]]  # 添加优先级
    return T, capa


# 传输
def trans(min_t, T, D_prio, new_state, capa, End_agent, times):
    t_index = []  # 把所有等于所花最短时间的流的下标保存（在D_prio中的下标）
    print('#各个流预计用时和其中最短用时：')
    print(T, min_t)

    for i1 in range(len(D_prio)):
        if T[i1] == min_t:
            t_index.append(i1)

    for i1 in range(len(D_prio)):
        eval(D_prio[i1])[new_state[i1]] = round(eval(D_prio[i1])[new_state[i1]] - round(
            capa[i1] * min_t), 2)  # 最先传完的流传完成，更新各个流的状态

    for i1 in range(len(t_index)):  # 所有耗时最短的流,都有可能结束了接收方当前的轮
        t_ind = t_index[i1]  # 流的下标,在D_prio中的下标
        flow = D_prio[t_ind].split('_')  # 具体的第t_ind个流
        a = int(flow[1])  # 发送方
        b = int(flow[2])  # 接收方
        c = list(set(nodes).difference(set([b])))  # 其他发送方
        send_index = []  # 发送给b，已经发送完的参与方
        for index2 in range(len(c)):
            if eval('D_' + str(c[index2]) + '_' + str(b))[new_state[t_ind]] == 0:  # 所有发送方在某层已经传完
                send_index.append(c[index2])
        if len(send_index) == len(c) - Ignore:  # 终点b结束当前轮，只要有16方发送完就算b结束
            print('**************')
            # print('到' + str(b) + '的流的第' + str(new_state[t_ind]) + '层传了一部分：' + 'D_' + str(a) + '_' + str(b) + '[' + str(
            #     new_state[t_ind]) + ']')
            print(str(b) + '的第' + str(new_state[t_ind]) + '层传完了')
            print('用时:' + str(min_t))
            main_T.append(min_t)  # 累计用时

            '''重新采样'''
            resample(D_prio, new_state, t_index, End_agent, times)  # 只要有一个终点完成当前轮，直接调用函数返回状态
            return
        else:
            print('======到' + str(b) + '的流的第' + str(new_state[t_ind]) + '层传了一部分：' + 'D_' + str(a) + '_' + str(b) + '[' + str(
                new_state[t_ind]) + ']' + '用时:' + str(min_t))
            main_T.append(min_t)  # 累计用时
            D_prio.pop(t_ind)  # 从流集合里去掉传完的流
            new_state.pop(t_ind)
            t_index = [x - 1 for x in t_index]  # 减去了元素，所以下标要完成的流的下标要减一

    if i1 == len(t_index) - 1:  # 没传完的流重新按优先级和带宽情况传（即上面运行完了，但是没有完成方）
        print("#该子流完成后各流及状态：")
        if D_prio == []:  # 传完了
            return
        else:
            print(D_prio)
            print(new_state)
            DD = []
            for i in range(len(D_prio)):
                DD.append(eval(D_prio[i])[new_state[i]])
            print(DD)
            print("#####重新调度")
            interact_ev(D_prio, new_state, End_agent, times)


# 一方结束当前层，计算和重新采样时环境的变换
def resample(D_prio, new_state, t_index, End_agent, times):  # 流结束的话重新采样,t_index是传完这一层的流的下标的集合
    com_tim = []  # 所有完成当前轮的点的计算加下一轮采样时间
    end_point = []  # 有哪些点完成了当前论
    for i2 in range(len(t_index)):  # 所有耗时最短的流
        print('#当前有' + str(len(t_index)) + '个数据流传完')
        t_ind = t_index[i2]  # 流的下标
        flow = D_prio[t_ind].split('_')  # 具体的第t_ind个流
        a = int(flow[1])  # 发送方
        b = int(flow[2])  # 接收方
        c = list(set(nodes).difference(set([b])))
        send_index = []  # 发送给b，已经发送完的参与方
        for index2 in range(len(c)):
            if eval('D_' + str(c[index2]) + '_' + str(b))[new_state[t_ind]] == 0:
                send_index.append(c[index2])

        if len(send_index) == len(c) - Ignore:  # 终点b结束当前层
            print('===花费时间----' + str(sum(main_T)))
            if new_state[t_ind] % 3 == 0: # 终点b结束当前轮
                End_agent.append(b)
                acc, loss = local_train(b, send_index)
                eval('Acc' + str(b) + '.append(acc)')  # 保存b方此轮的acc
                eval('Loss' + str(b) + '.append(loss)')  # 保存b方此轮的loss
                round_time_cost = sum(main_T) - Start_time[b]
                eval('Time_round' + str(b) + '.append(round_time_cost)')  # 保存b方此轮的time_cost
                Start_time[b] = sum(main_T)
                times = 1000000
                com_tim.append((1.5+1.82+0.664))  # 聚合时间，前向时间
                parent_t = 1.5+1.82+0.664
                print('===============第' + str(b) + '方完成了一轮，重新更新参数' + '需要' + \
                      str(parent_t) + '的重新时间')

            else:
                com_tim.append((1.5+1.82))  # 计算时间
                parent_t = 1.5+1.82
                print('=====到' + str(b) + '的' + str(new_state[t_ind]) + '层数据流传完，' + str(b) + '需要' + \
                      str(parent_t) + '的计算和重新采样时间')
            end_point.append(b)
            D_prio.pop(t_ind)  # 从流集合里去掉传完的流
            new_state.pop(t_ind)
            t_index = [x - 1 for x in t_index]  # 减去了元素，所以下标要完成的流的下标要减一

    for index4 in range(len(end_point)):  # 终点是结束阶段的流，进入下一层
        for index5 in range(len(priop)):  # 所有的流
            flow = priop[index5].split('_')  # 具体的第j个流
            a = int(flow[1])  # 发送方
            b = int(flow[2])  # 接收方
            if b == end_point[index4]:
                eval('D_' + str(a) + '_' + str(b))[main_state[priop.index('D_' + str(a) + '_' + str(b))]] = 0
                if main_state[priop.index('D_' + str(a) + '_' + str(b))] < Epo - 1:
                    main_state[priop.index('D_' + str(a) + '_' + str(b))] = main_state[priop.index(
                        'D_' + str(a) + '_' + str(b))] + 1

    for index4 in range(len(end_point)):  # 终点是结束阶段的流
        for index5 in range(len(D_prio)):  # 该回合的流
            if index5 < len(D_prio):
                flow = D_prio[index5].split('_')  # 具体的第j个流
                a = int(flow[1])  # 发送方
                b = int(flow[2])  # 接收方
                if b == end_point[index4]:
                    D_prio.pop(index5)  # 从流集合里去掉传完的流
                    new_state.pop(index5)

    if D_prio == []:  # 传完了
        return

    t_max = max(com_tim)  # 最长的计算时间
    T, capa = gen_T(D_prio, new_state)

    min_t = min(T)  # 预计传完最小的时间
    if min_t > t_max:  # 各方在完成方计算时间内无法传完
        # main_T.pop(len(main_T) - 1)  # 把上一个循环里的传输时间去掉，因为后面的采样和计算时间会覆盖掉它
        main_T.append(parent_t)  # 最里层的采样时间
        for i1 in range(len(D_prio)):
            eval(D_prio[i1])[new_state[i1]] = round(eval(D_prio[i1])[new_state[i1]] - round(
                capa[i1] * t_max), 2)  # 最先传完的流传完成，更新各个流的状态
            print('#完成采样和计算后各个流的状态')
            print(D_prio)
            print(new_state)
            dd = []
            for i in range(len(D_prio)):
                dd.append(eval(D_prio[i])[new_state[i]])
            print(dd)
            return
    else:
        print('#在采样和计算时间内又有其他的流完成传输')
        trans(min_t, T, D_prio, new_state, capa, End_agent, times)


# 给定优先级，按当前带宽，数据量调度，带宽在这期间内不变， 数据传完自动到下一个优先级数据，某一方全部传完则该轮结束，返回状态
def interact_ev(prio, state, End_agent, times):  # prio，优先级，里面是打乱顺序的1-6，state[0]对应prio[0]流的所在层数
    print('#限制前的优先级,层数，待传数据量')
    print(prio)
    print(state)
    dd = []
    for i in range(len(prio)):
        dd.append(eval(prio[i])[state[i]])
    print(dd)

    """获取当前状态每条流的带宽条件"""
    for i in range(len(paths)):
        a, b = paths[i]  # 连接边的两个点
        G[a][b][0]['Capacity'] = bandwidth[times][i]  # 获取当前状态每条流的带宽条件

    '''根据限制去除无效的流'''
    D_prio, new_state = regeneral(prio, state)
    print('#限制后的优先级,层数，待传数据量')
    print(D_prio)
    print(new_state)
    dd = []
    for i in range(len(D_prio)):
        dd.append(eval(D_prio[i])[new_state[i]])
    print(dd)

    '''当前情况下每个流预计传完时间，及最小的capabilities'''
    T, capa = gen_T(D_prio, new_state)

    '''判断是否有流传完'''
    min_t = min(T)  # 预计传完最小的时间

    trans(min_t, T, D_prio, new_state, capa, End_agent, times)


# if __name__ == '__main__':
#     nodes = []
#     for index1 in range(N):
#         nodes.append(index1)
#     main_T = []
#     main_state = []  # 输入状态,各个流所在层数
#     for i in range(N * (N - 1)):
#         main_state.append(1)
#     times = 1  # interaction times
#     D = 'D_'
#     D1 = '_'
#     priop = []
#     for i1 in range(N):
#         for j1 in range(N):
#             if i1 != j1:
#                 eval('priop.append(D+str(i1)+D1+str(j1))')
#
#     for ii in range(N):
#         for jj in range(N):
#             if ii != jj:
#                 while eval('D_' + str(ii) + '_' + str(jj) + '[Epo-1] != 0'):
#                     print('####################第' + str(times) + '次交互####################')
#                     priopp = []
#                     Main_state = []
#                     for j2 in range(len(priop)):
#                         if eval(priop[j2])[main_state[j2]] != 0:
#                             priopp.append(priop[j2])
#                             Main_state.append(main_state[j2])
#                     interact_ev(priopp, Main_state)
#                     times = times + 1
#     print(sum(main_T))
Ignore = 0
nodes = []
for index1 in range(N):
    nodes.append(index1)
main_T = []
main_state = []  # 输入状态,各个流所在层数
for i in range(N * (N - 1)):
    main_state.append(1)

D = 'D_'
D1 = '_'
priop = []
for i1 in range(N):
    for j1 in range(N):
        if i1 != j1:
            eval('priop.append(D+str(i1)+D1+str(j1))')


def in_out(priop, main_state, End_agent, times):
    End_agent = []
    for ii in range(N):
        for jj in range(N):
            if ii != jj:
                while eval('D_' + str(ii) + '_' + str(jj) + '[Epo-1] != 0'):
                    print('###############################第' + str(times) + '次交互##################################')
                    priopp = []
                    Main_state = []
                    for j2 in range(len(priop)):
                        if eval(priop[j2])[main_state[j2]] != 0:
                            priopp.append(priop[j2])
                            Main_state.append(main_state[j2])
                    interact_ev(priopp, Main_state, End_agent, times)
                    times = times + 1
                    if times == 1000000:
                        return End_agent


times = 1  # interaction times
End_agent = []
aa = in_out(priop, main_state, End_agent, times)
print('结束方：')
print(aa)
print('结束方此轮训练')
