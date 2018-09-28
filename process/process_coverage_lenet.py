import os
import numpy as np
import data_provider
import gc
import time
from itertools import combinations, permutations

'''
本脚本是將样本的激活数据的边覆盖情况具现化保存
数据结构边写边考虑
'''

'''
目前只考虑1-1的三种情况的覆盖
写尽可能考虑可拓展性
'''


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# 获取两层之间的每个数据的对应的
def process_two_layer(m, n, active_datas1, active_datas2, type=None):
    # 首先样本数据量需要一致对吧
    if len(active_datas1) != len(active_datas2):
        raise RuntimeError("the dimension of two layer active datas are not compatible")
    if type is None:
        type = 0
    nums = np.shape(active_datas1)[0]
    layer1_neuron_nums = np.shape(active_datas1)[1]
    layer2_neuron_nums = np.shape(active_datas2)[1]
    # 直接使用python库自带的组合情况计算的工具类来产生组合的情况
    # 获取的是itertools的combinations的object,可以list转,但是本身是可以迭代的。
    layer1_combination = list(combinations(range(layer1_neuron_nums), m))
    layer2_combination = list(combinations(range(layer2_neuron_nums), n))
    # m+n个情况里面的全覆盖的总数是 2^(m+n)
    condition_nums = 2 ** (m + n)
    if type == 1:
        condition_nums = 2 ** m
    # i am not sure this should be write down , it could be a specific condition~ maybe~
    elif type == 2:
        condition_nums = 1

    # process_data 是指对应的样本在每个组合中对应的覆盖情况
    process_data = []
    for comb1 in layer1_combination:
        for comb2 in layer2_combination:
            cover_data = []
            # 取出两层的数据
            for i in range(nums):
                temp_data = np.zeros((condition_nums,))
                data1 = [active_datas1[i][index] for index in comb1]
                data2 = [active_datas2[i][index] for index in comb2]
                # 全覆盖
                if type == 0:
                    data1.extend(data2)
                    value = cal_2_to_10_value(data1)
                    temp_data[int(value)] = 1
                elif type == 1:
                    data2 = np.array(data2)
                    # 如果输出端全激活
                    if len(data2) == len(data2[data2 == 1]):
                        value = cal_2_to_10_value(data1)
                        temp_data[int(value)] = 1
                else:
                    data1.extend(data2)
                    data1 = np.array(data1)
                    if len(data1) == len(data1[data1 == 1]):
                        temp_data[0] = 1
                cover_data.append(cal_2_to_10_value(list(temp_data)))
            process_data.append(cover_data)
    process_data = np.array(process_data)
    process_data = process_data.transpose((1, 0))
    print(process_data[0])
    print(process_data.shape)
    return process_data


# 二进制转十进制
# 计算的是unsigned 非负数
def cal_2_to_10_value(datas):
    # 反序容易计算
    datas.reverse()
    result = 0
    for i in range(len(datas)):
        result += (datas[i] * (2 ** i))
    return result


m = 1
n = 1
# 以0为激活的阈值
threshold = 0
# 数据集是mnist
dataset = 'mnist'

# 0-全覆盖
# 1-输出端激活覆盖
# 2-输入输出端均激活的覆盖
type = 0

if __name__ == '__main__':
    for threshold in [0, 0.25, 0.5, 0.75, 1]:
        for type in [0, 1, 2]:
            now = time.time()
            right_datas = np.load('../data/mnist/mnist_right_active_data/lenet5_active_data.npy')
            layer_nums = 4
            # process_right_datas and save coverage data
            # 默认我们用相邻两层
            result_for_right = np.empty((len(right_datas), 0))
            for layer_num in range(layer_nums - 1):
                layer1 = layer_num
                layer2 = layer_num + 1
                datas1 = np.array([list(item) for item in right_datas[:, layer1]])
                datas2 = np.array([list(item) for item in right_datas[:, layer2]])
                print(datas1.shape)
                print(datas2.shape)
                datas1[datas1 > threshold] = 1
                datas2[datas2 > threshold] = 1
                datas1[datas1 <= threshold] = 0
                datas2[datas2 <= threshold] = 0
                process_data_for_right = process_two_layer(m, n, datas1, datas2, type)
                result_for_right = np.hstack((result_for_right, process_data_for_right))
            # process结束后
            # 保存激活数据信息
            print(result_for_right.shape)
            save_path = '../data/mnist/mnist_right_active_data/coverage/threshold' + str(
                threshold) + '/adjacent_' + str(
                m) + '_' + str(n) + '/type' + str(type)
            check_dir(save_path)
            np.save(os.path.join(save_path, 'lenet5_coverage.npy'), result_for_right)
            del right_datas
            del result_for_right

            gc.collect()
            # 错误数据的激活信息保存
            for attack_type in ['fgsm', 'gaussian_noise', 'saliency_map', 'uniform_noise']:
                wrong_datas = np.load(
                    '../data/mnist/mnist_wrong_active_data/' + attack_type + '/lenet5_active_data.npy')
                # process_wrong_datas and save coverage data
                # 默认我们用相邻两层
                result_for_wrong = np.empty((len(wrong_datas), 0))
                for layer_num in range(layer_nums - 1):
                    layer1 = layer_num
                    layer2 = layer_num + 1
                    datas1 = np.array([list(item) for item in wrong_datas[:, layer1]])
                    datas2 = np.array([list(item) for item in wrong_datas[:, layer2]])
                    print(datas1.shape)
                    print(datas2.shape)
                    datas1[datas1 > threshold] = 1
                    datas2[datas2 > threshold] = 1
                    datas1[datas1 <= threshold] = 0
                    datas2[datas2 <= threshold] = 0
                    process_data_for_wrong = process_two_layer(m, n, datas1, datas2, type)
                    result_for_wrong = np.hstack((result_for_wrong, process_data_for_wrong))
                # process结束后
                # 保存激活数据的信息
                print(result_for_wrong.shape)
                save_path = '../data/mnist/mnist_wrong_active_data/coverage/' + attack_type + '/threshold' + str(
                    threshold) + '/adjacent_' + str(m) + '_' + str(n) + '/type' + str(type)
                check_dir(save_path)
                np.save(os.path.join(save_path, 'lenet5_coverage.npy'), result_for_wrong)
                del wrong_datas
                del result_for_wrong
                gc.collect()
