import numpy as np
import os
import json
from itertools import combinations, permutations
import data_provider
import random
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import gc

# 默认的模式只能在windows下使用,改成通用的Agg
# mpl.use('Agg')
'''
计算样本的覆盖率
试验1代码 测试all wrong 和attack
'''


# 前面取m个神经元，后面取n个神经元
# active_datas1 表示num个样本的对应前面一层的神经元们的激活情况
# active_datas2 表示num个样本的对应后面一层的神经元们的激活情况
# 后面估计有Type属性？计算的是哪种覆盖？ 0-4种全覆盖 1-输出端激活 2-全部激活
# 缺点不难看出这是一个指数级增长的情况
def two_layer(m, n, k, active_datas1, active_datas2, active_datas3, type=None):
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
    layer3_neuron_nums = np.shape(active_datas3)[1]
    # 直接使用python库自带的组合情况计算的工具类来产生组合的情况
    # 获取的是itertools的combinations的object,可以list转,但是本身是可以迭代的。
    layer1_combination = list(combinations(range(layer1_neuron_nums), m))
    layer2_combination = list(combinations(range(layer2_neuron_nums), n))
    layer3_combination = list(combinations(range(layer3_neuron_nums), k))
    # m+n个情况里面的全覆盖的总数是 2^(m+n) 全覆盖默认是type=0
    condition_nums = 2 ** (m + n + k)
    if type == 1:
        condition_nums = 2 ** (m + n)
    for comb1 in layer1_combination:
        for comb2 in layer2_combination:
            for comb3 in layer3_combination:
                # 记录一个由所有情况排列的数据列表 用于设置覆盖情况
                temp_data = list(range(condition_nums))
                temp_cover = []
                # 取出两层的数据
                for i in range(nums):
                    data1 = [active_datas1[i][index] for index in comb1]
                    data2 = [active_datas2[i][index] for index in comb2]
                    data3 = [active_datas3[i][index] for index in comb3]
                    # 全覆盖
                    if type == 0:
                        data1.extend(data2)
                        data1.extend(data3)
                        value = cal_2_to_10_value(data1)
                        temp_data[int(value)] = -1
                        temp_cover.append(temp_data.count(-1))
                    elif type == 1:
                        data3 = np.array(data3)
                        # 如果输出端全激活 输出端是指data3
                        if len(data3) == len(data3[data3 == 1]):
                            data1.extend(data2)
                            value = cal_2_to_10_value(data1)
                            temp_data[int(value)] = -1
                            temp_cover.append(temp_data.count(-1))
                        else:
                            if len(temp_cover) == 0:
                                temp_cover.append(0)
                            else:
                                temp_cover.append(temp_cover[-1])
                    else:
                        data1.extend(data2)
                        data1.extend(data3)
                        data1 = np.array(data1)
                        if len(data1) == len(data1[data1 == 1]):
                            temp_cover.append(1)
                        else:
                            # 如果没覆盖就把之前的累计覆盖填上去
                            if len(temp_cover) == 0:
                                temp_cover.append(0)
                            else:
                                temp_cover.append(temp_cover[-1])
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


def get_total_size(combine_nums, type, m, n, k):
    if type == 1:
        return combine_nums * (2 ** (m + n))
    elif type == 2:
        return combine_nums
    else:
        return combine_nums * (2 ** (m + n + k))


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == '__main__':
    # threshold = 0
    percentile = 0
    m = 1
    n = 1
    k = 1
    # 将几种阈值都尝试一遍
    for percentile in [25, 50, 75]:
        for type in [0, 1, 2]:
            for layer_num in [3, 5, 10]:
                result = {"model": str(layer_num) + '_hidden_layers_model.hdf5', "type": type, 'data_type': 'right'}
                result_coverage_data = {}
                result_process_data = {}
                # 隔离出来的保存rq1的目录
                result_path = '../result/rq1/right/' + str(layer_num) + '_hidden_layers_model/adjacent_' + str(
                    m) + '_' + str(n) + '_' + str(k) + '_percentile_' + str(
                    percentile) + '/type' + str(type)
                check_dir(result_path)
                now = time.time()
                print(str(layer_num) + ' hidden layers model process type' + str(type) + '...right...')
                plt.figure
                datas = data_provider.get_right_active_data('../data/mnist/', 'mnist', layer_num)
                datas = data_provider.shuffle_mnist_data(datas)
                total_data = []
                process_data = []
                layers = []
                for j in range(layer_num - 2):
                    datas1 = np.array([list(item) for item in datas[:, j]])
                    datas2 = np.array([list(item) for item in datas[:, j + 1]])
                    datas3 = np.array([list(item) for item in datas[:, j + 2]])
                    # print(datas1.shape)
                    # print(datas2.shape)
                    threshold1 = np.percentile(datas1, percentile)
                    threshold2 = np.percentile(datas2, percentile)
                    threshold3 = np.percentile(datas3, percentile)
                    datas1[datas1 > threshold1] = 1
                    datas2[datas2 > threshold2] = 1
                    datas3[datas3 > threshold3] = 1
                    datas1[datas1 <= threshold1] = 0
                    datas2[datas2 <= threshold2] = 0
                    datas3[datas3 <= threshold3] = 0
                    # print(datas1.shape)
                    # print(datas2.shape)
                    total_comb_num, coverage_data = two_layer(m, n, k, datas1, datas2, datas3, type=type)
                    total_size = get_total_size(total_comb_num, type, m, n, k)
                    total_data.append(total_size)
                    process_data.append(coverage_data)
                    layers.append([j, j + 1, j + 2])
                    # 添加j,j+1层激活数据的情况
                    coverage_rate = coverage_data / total_size
                    plt.plot(range(1, len(datas) + 1), coverage_data / total_size,
                             label=str(j) + '_' + str(j + 1) + '_' + str(j + 2))
                    print('第' + str(j) + '-' + str(j + 2) + '层最终覆盖率:' + str(coverage_rate[-1]))
                    result_coverage_data[str(j) + '_' + str(j + 2)] = coverage_rate[-1]
                    result_process_data[str(j) + '_' + str(j + 2)] = coverage_rate.tolist()
                    del datas1
                    del datas2
                    del datas3
                    gc.collect()
                total_rate = np.sum(process_data, axis=0) / np.sum(total_data)
                plt.plot(range(1, len(datas) + 1), total_rate, label='total')
                print('综合的边覆盖是:' + str(total_rate[-1]))
                result_coverage_data['total'] = total_rate[-1]
                result_process_data['total'] = total_rate.tolist()
                result['coverage_result'] = result_coverage_data
                # 过程数据 两千列, layer数据 目前是相邻两层 , total_data在这两层对应的覆盖情况总数
                data_path = '../data/mnist/coverage/' + 'adjacent_' + str(
                    m) + '_' + str(n) + '_' + str(k) + '_percentile' + str(percentile) + '/type' + str(
                    type) + '/right/'
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                np.save(os.path.join(data_path, str(layer_num) + '_hidden_layers_coverage_data.npy'),
                        [process_data, layers, total_data])
                # 消耗的时间
                print(str(time.time() - now) + ' s')
                plt.legend(loc='best')
                plt.savefig(os.path.join(data_path, str(layer_num) + '_hidden_layer_mnist_model_for_right.jpg'))
                plt.savefig(os.path.join(result_path, str(layer_num) + '_hidden_layer_mnist_model_for_right.jpg'))
                plt.ion()
                # plt.pause(1)
                plt.close()
                del datas
                del process_data
                del total_data
                gc.collect()
                with open(os.path.join(result_path, 'result.json'), 'w') as file:
                    json.dump(result, file)
                with open(os.path.join(result_path, 'process.json'), 'w') as file:
                    json.dump(result_process_data, file)
