import numpy as np
import os
import json
import math
from itertools import combinations, permutations
import data_provider
import random
import matplotlib.pyplot as plt
import time

'''
计算样本的覆盖率
'''


# 前面取m个神经元，后面取n个神经元
# active_datas1 表示num个样本的对应前面一层的神经元们的激活情况
# active_datas2 表示num个样本的对应后面一层的神经元们的激活情况
# 后面估计有Type属性？计算的是哪种覆盖？
# 缺点不难看出这是一个指数级增长的情况
def two_layer(m, n, active_datas1, active_datas2, type=None):
    # 记录过程的数据
    # process_data = np.zeros(shape=(len(layer1_combination) * len(layer2_combination), nums))
    process_data = []
    # 首先样本数据量需要一致对吧
    if len(active_datas1) != len(active_datas2):
        raise RuntimeError("the dimension of two layer active datas are not compatible")
    # 获取一些维度数据如样本数量，每层神经元的个数
    nums = np.shape(active_datas1)[0]
    layer1_neuron_nums = np.shape(active_datas1)[1]
    layer2_neuron_nums = np.shape(active_datas2)[1]
    # 直接使用python库自带的组合情况计算的工具类来产生组合的情况
    # 获取的是itertools的combinations的object,可以list转,但是本身是可以迭代的。
    layer1_combination = list(combinations(range(layer1_neuron_nums), m))
    layer2_combination = list(combinations(range(layer2_neuron_nums), n))
    # m+n个情况里面的全覆盖的总数是 2^(m+n)
    condition_nums = 2 ** (m + n)
    for comb1 in layer1_combination:
        for comb2 in layer2_combination:
            # 记录一个由所有情况排列的数据列表 用于设置覆盖情况
            temp_data = list(range(condition_nums))
            temp_cover = []
            for i in range(nums):
                # 取出两层的数据
                data1 = [active_datas1[i][index] for index in comb1]
                data2 = [active_datas2[i][index] for index in comb2]
                data1.extend(data2)
                value = cal_2_to_10_value(data1)
                temp_data[int(value)] = -1
                temp_cover.append(temp_data.count(-1))
            process_data.append(temp_cover)
    # 统计覆盖情况 每一列的数据的加和情况
    print(np.shape(process_data))
    return np.shape(process_data)[0], np.sum(process_data, axis=0)


# 二进制转十进制
# 计算的是unsigned 非负数
def cal_2_to_10_value(datas):
    # 反序容易计算
    datas.reverse()
    result = 0
    for i in range(len(datas)):
        result += (datas[i] * (2 ** i))
    return result


if __name__ == '__main__':
    # combination = combinations(range(50), 2)
    # print(combination)
    # # print(len(list(combination)))
    # for item in combination:
    #     print(item)
    # data = np.array(range(10))
    # print(data)
    # print(cal_2_to_10_value([1, 1, 0, 1]))
    # data = [[1, 2, 3], [2, 3, 4]]
    # print(np.sum(data, axis=0))
    # print(np.sum(data, axis=1))

    neurons = [128, 64, 32]
    m = 1
    n = 1
    sample_nums = 2000
    labels = list(range(1, sample_nums + 1))
    for i in [3, 5, 10]:
        now = time.time()
        print(i)
        plt.figure()
        datas = data_provider.get_all_active_data('../data/mnist/', 'mnist', i)
        datas = data_provider.shuffle_mnist_data(datas)
        datas = np.array(random.sample(list(datas), sample_nums))
        total_data = []
        process_data = []
        for j in range(i - 1):
            datas1 = np.array([list(item) for item in datas[:, j]])
            datas2 = np.array([list(item) for item in datas[:, j + 1]])
            print(datas1.shape)
            print(datas2.shape)
            datas1[datas1 > 0] = 1
            datas2[datas2 > 0] = 1
            datas1[datas1 <= 0] = 0
            datas2[datas2 <= 0] = 0
            total_comb_num, coverage_data = two_layer(m, n, datas1, datas2)
            total_data.append(total_comb_num * (2 ** (m + n)))
            process_data.append(coverage_data)
            # 添加j,j+1层激活数据的情况
            plt.plot(labels, coverage_data / (total_comb_num * (2 ** (m + n))), label=str(j) + '_' + str(j + 1))
        plt.plot(labels, np.sum(process_data, axis=0) / np.sum(total_data), label='total')
        # 消耗的时间
        print(str(time.time() - now) + ' s')
        plt.legend(loc='best')
        plt.savefig(str(i) + '_hidden_layer_mnist_model.jpg')
        plt.ion()
        # plt.pause(1)
        plt.close()
