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

batch_size = 30000


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def bitwise_or(array_data):
    result = array_data[0]
    for data in array_data:
        result = np.bitwise_or(result, data)
    print(result.shape)
    return result


# 获取两层之间的每个数据的对应的
def process_three_layer(layer_nums, m, n, k, active_datas1, active_datas2, active_datas3, save_paths, layer_index,
                        type=None):
    # 首先样本数据量需要一致对吧
    if len(active_datas1) != len(active_datas2):
        raise RuntimeError("the dimension of two layer active datas are not compatible")
    if type is None:
        type = 0
    layer1_neuron_nums = np.shape(active_datas1[0])[1]
    layer2_neuron_nums = np.shape(active_datas2[0])[1]
    layer3_neuron_nums = np.shape(active_datas3[0])[1]
    print('layer1 neuron nums is ', layer1_neuron_nums)
    print('layer2 neuron nums is ', layer2_neuron_nums)
    print('layer3 neuron nums is ', layer3_neuron_nums)
    # 直接使用python库自带的组合情况计算的工具类来产生组合的情况
    # 获取的是itertools的combinations的object,可以list转,但是本身是可以迭代的。
    layer1_combination = list(combinations(range(layer1_neuron_nums), m))
    layer2_combination = list(combinations(range(layer2_neuron_nums), n))
    layer3_combination = list(combinations(range(layer3_neuron_nums), k))
    # m+n个情况里面的全覆盖的总数是 2^(m+n)
    condition_nums = 2 ** (m + n + k)
    if type == 1:
        condition_nums = 2 ** (m + n)
    # i am not sure this should be write down , it could be a specific condition~ maybe~
    elif type == 2:
        condition_nums = 1

    # process_data 是指对应的样本在每个组合中对应的覆盖情况
    process_data = []
    batch_index = 0
    # 记录condition的index
    condition_index = 0

    result = np.zeros((len(save_paths),), dtype=np.int)
    combination_nums = len(layer1_combination) * len(layer2_combination) * len(layer3_combination)
    print('codition_nums is ', len(layer1_combination) * len(layer2_combination) * len(layer3_combination))
    process_data = [[], [], [], [], []]
    for comb1 in layer1_combination:
        for comb2 in layer2_combination:
            for comb3 in layer3_combination:
                condition_index += 1
                print(condition_index)
                # 取出3层的数据
                for i in range(len(save_paths)):
                    # print(i, end=' ')
                    save_path = save_paths[i]
                    # print(save_path)
                    datas1 = active_datas1[i]
                    datas2 = active_datas2[i]
                    datas3 = active_datas3[i]
                    temp_data = np.zeros((condition_nums,))
                    sample_nums = np.shape(datas1)[0]
                    print('sample_nums ', sample_nums)
                    for j in range(sample_nums):
                        cover_data = []
                        data1 = [datas1[j][index] for index in comb1]
                        data2 = [datas2[j][index] for index in comb2]
                        data3 = [datas3[j][index] for index in comb3]
                        # 全覆盖
                        if type == 0:
                            data1.extend(data2)
                            data1.extend(data3)
                            value = cal_2_to_10_value(data1)
                            temp_data[int(value)] = 1
                        elif type == 1:
                            data3 = np.array(data3)
                            # 如果输出端全激活
                            if len(data3) == len(data3[data3 == 1]):
                                data1.extend(data2)
                                value = cal_2_to_10_value(data1)
                                temp_data[int(value)] = 1
                        else:
                            data1.extend(data2)
                            data1.extend(data3)
                            data1 = np.array(data1)
                            if len(data1) == len(data1[data1 == 1]):
                                temp_data[0] = 1
                        # cover_data.append(cal_2_to_10_value(list(temp_data)))
                        # print(cal_2_to_10_value(list(temp_data)))
                        result[i] = np.bitwise_or(result[i], int(cal_2_to_10_value(list(temp_data))))

                    # process_data.append(cover_data)
                    process_data[i].append(result[i])
                    if len(process_data[i]) == batch_size or condition_index == combination_nums:
                        # process_data = np.transpose(process_data, (1, 0))
                        # result = bitwise_or(process_data)
                        np.save(os.path.join(save_path,
                                             str(layer_nums) + '_hidden_layers_coverage_total_layer_index_' + str(
                                                 layer_index) + '_batch_' + str(
                                                 batch_index) + '.npy'), process_data[i])
                        batch_index += 1
                        result[i] = 0
                        process_data[i] = []
    # process_data = np.array(process_data)
    # process_data = process_data.transpose((1, 0))
    # print(process_data[0])
    # print(process_data.shape)
    # return process_data


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
k = 1
# 以0为激活的阈值
threshold = 0
# 数据集是mnist
dataset = 'mnist'

# 0-全覆盖
# 1-输出端激活覆盖
# 2-输入输出端均激活的覆盖
type = 2

if __name__ == '__main__':
    for threshold in [0.5]:
        for type in [0, 1, 2]:
            for layer_nums in [3]:
                if type == 0 and layer_nums == 3:
                    continue
                print('threshold is ', threshold)
                print('layer_num is ', layer_nums)
                print('type is ', type)
                now = time.time()
                right_datas = data_provider.get_right_active_data('../data/mnist', dataset, layer_nums)
                # process_right_datas and save coverage data
                # 默认我们用相邻两层
                result_for_right = np.empty((len(right_datas), 0))
                for layer_num in range(layer_nums - 2):
                    save_paths = []
                    datas_1_list = []
                    datas_2_list = []
                    datas_3_list = []
                    layer1 = layer_num
                    layer2 = layer_num + 1
                    layer3 = layer_num + 2
                    datas1 = np.array([list(item) for item in right_datas[:, layer1]])
                    datas2 = np.array([list(item) for item in right_datas[:, layer2]])
                    datas3 = np.array([list(item) for item in right_datas[:, layer3]])
                    print(datas1.shape)
                    print(datas2.shape)
                    datas1[datas1 > threshold] = 1
                    datas2[datas2 > threshold] = 1
                    datas3[datas3 > threshold] = 1
                    datas1[datas1 <= threshold] = 0
                    datas2[datas2 <= threshold] = 0
                    datas3[datas3 <= threshold] = 0
                    save_path = '../data/mnist/mnist_right_active_data/coverage/threshold' + str(
                        threshold) + '/adjacent_' + str(
                        m) + '_' + str(n) + '_' + str(k) + '/type' + str(type)
                    check_dir(save_path)
                    save_paths.append(save_path)
                    datas_1_list.append(datas1)
                    datas_2_list.append(datas2)
                    datas_3_list.append(datas3)

                    # 错误数据的激活信息保存
                    for attack_type in ['fgsm', 'gaussian_noise', 'saliency_map', 'uniform_noise']:
                        wrong_datas = data_provider.get_wrong_active_data_with_attack_type('../data/mnist', dataset,
                                                                                           layer_nums,
                                                                                           attack_type)
                        # process_wrong_datas and save coverage data
                        layer1 = layer_num
                        layer2 = layer_num + 1
                        layer3 = layer_num + 2
                        datas1 = np.array([list(item) for item in wrong_datas[:, layer1]])
                        datas2 = np.array([list(item) for item in wrong_datas[:, layer2]])
                        datas3 = np.array([list(item) for item in wrong_datas[:, layer3]])
                        print(datas1.shape)
                        print(datas2.shape)
                        print(datas3.shape)
                        datas1[datas1 > threshold] = 1
                        datas2[datas2 > threshold] = 1
                        datas3[datas3 > threshold] = 1
                        datas1[datas1 <= threshold] = 0
                        datas2[datas2 <= threshold] = 0
                        datas3[datas3 <= threshold] = 0

                        save_path = '../data/mnist/mnist_wrong_active_data/coverage/' + attack_type + '/threshold' + str(
                            threshold) + '/adjacent_' + str(m) + '_' + str(n) + '_' + str(k) + '/type' + str(type)
                        check_dir(save_path)
                        save_paths.append(save_path)
                        datas_1_list.append(datas1)
                        datas_2_list.append(datas2)
                        datas_3_list.append(datas3)

                    process_three_layer(layer_nums, m, n, k, datas_1_list, datas_2_list, datas_3_list, save_paths,
                                        layer_num, type)
