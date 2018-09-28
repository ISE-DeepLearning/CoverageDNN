import os
import numpy as np
import gc
import json


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def gc_collect(datas):
    del datas
    gc.collect()


# 接下来这两个方法，硬编码，不是很好，有机会改一下……
def get_value_for_two_layer_data(type=0):
    active_value = []
    unative_value = []
    if type == 0:
        active_value = [2, 4]
        unative_value = [1, 1]
    elif type == 1:
        active_value = [2, 1]
        unative_value = [1, 0]
    elif type == 2:
        active_value = [1, 1]
        unative_value = [0, 0]
    return active_value, unative_value


def get_condition_nums_for_two_layer_data(type=0):
    if type == 0:
        return 4
    elif type == 1:
        return 2
    elif type == 2:
        return 1


# ----------------- end -------------------------


def bitwise_or(data_array):
    result = data_array[0]
    for data in data_array:
        result = np.bitwise_or(result, data)
    return result


def calculate_coverage(result, nums):
    result = [bin(n).count('1') for n in result]
    return np.sum(result) / (len(result) * nums)


def process_active_data(active_data, layer_num, threshold, is_percentile=False, type=0):
    active_values, unactive_values = get_value_for_two_layer_data(type)
    # print(active_data.shape)

    total_result = []
    for layer_index in range(layer_num - 1):
        layer1 = layer_index
        layer2 = layer_index + 1
        # print(layer1)
        # print(layer2)
        thresholds = [threshold, threshold]

        datas1 = np.array([list(item) for item in active_data[:, layer1]])
        datas2 = np.array([list(item) for item in active_data[:, layer2]])
        print(datas1.shape)
        print(datas2.shape)
        #  如果是寻找分位值的话，需要重新设置threshold
        if is_percentile:
            # flatten 与否结果都是一样
            # thresholds = [np.percentile(datas1.flatten(), threshold), np.percentile(datas2.flatten(), threshold)]
            thresholds = [np.percentile(datas1, threshold), np.percentile(datas2, threshold)]
        datas1[datas1 > thresholds[0]] = active_values[0]
        datas2[datas2 > thresholds[1]] = active_values[1]
        datas1[datas1 <= thresholds[0]] = unactive_values[0]
        datas2[datas2 <= thresholds[1]] = unactive_values[1]

        result = np.zeros((datas1.shape[1] * datas2.shape[1],), np.int)
        for i in range(len(datas1)):
            # print(i)
            # print(datas1[i].shape)
            # print(datas2[i][:, np.newaxis].shape)
            # data = np.multiply(datas1[i], datas2[i][:, np.newaxis])
            # print(data.shape)
            result = np.bitwise_or(result, np.multiply(datas1[i], datas2[i][:, np.newaxis]).flatten().astype(np.int))
        total_result.append(result)
        # print(result.shape)
    total_result = np.concatenate(total_result)
    print(total_result.shape)
    return total_result


m = n = 1

if __name__ == '__main__':
    for type in [0, 1, 2]:
        active_values, unactive_values = get_value_for_two_layer_data(type)
        for threshold in [0, 0.25, 0.5, 0.75, 1]:
            # 保存文件的目录
            result_path = '../../result/optimizer_new/rq1/adjacent1_1/threshold' + str(threshold) + '/type' + str(
                type)
            check_dir(result_path)
            result_json = {'model': 'lenet5_model', 'type': type, 'threshold': threshold}
            print(result_json)
            # right
            right_active_data = np.load(
                '../../data/mnist/mnist_right_active_data/' + 'lenet5_full_active_data.npy')
            right_result = process_active_data(right_active_data, 4, threshold, False, type)
            # 保留正确数据的总的覆盖信息数据
            np.save(os.path.join(result_path, 'lenet5_full_right_result.npy'), right_result)
            coverage = calculate_coverage(right_result, get_condition_nums_for_two_layer_data(type))
            result_json['right'] = coverage
            gc_collect(right_active_data)

            # wrong
            wrong_results = []
            for attack_type in ['fgsm', 'gaussian_noise', 'saliency_map', 'uniform_noise']:
                wrong_active_data = np.load(
                    '../../data/mnist/mnist_wrong_active_data/' + attack_type + '/' + 'lenet5_full_active_data.npy')
                wrong_result = process_active_data(wrong_active_data, 4, threshold, False, type)
                wrong_results.append(wrong_result)
                # 保留错误数据的总的覆盖信息数据
                np.save(os.path.join(result_path, 'lenet5_full_' + attack_type + '_result.npy'), wrong_result)
                gc_collect(wrong_active_data)
                coverage = calculate_coverage(wrong_result, get_condition_nums_for_two_layer_data(type))
                result_json['wrong_' + attack_type] = coverage
                # all
                all_result = np.bitwise_or(right_result, wrong_result)
                # 保留all数据的总的覆盖信息数据
                np.save(os.path.join(result_path, 'lenet5_full_all_' + attack_type + '_result.npy'), all_result)
                coverage = calculate_coverage(all_result, get_condition_nums_for_two_layer_data(type))
                result_json['all_' + attack_type] = coverage
            # total
            all_wrong_result = bitwise_or(wrong_results)
            total_result = np.bitwise_or(all_wrong_result, right_result)
            np.save(os.path.join(result_path, 'lenet5_full_total_result.npy'), total_result)
            coverage = calculate_coverage(total_result, get_condition_nums_for_two_layer_data(type))
            result_json['total'] = coverage
            with open(os.path.join(result_path, 'lenet5_full_model_result.json'),
                      'w') as file:
                json.dump(result_json, file)

        for percentile in [25, 50, 75]:
            result_path = '../../result/optimizer_new/rq1/adjacent1_1/percentile' + str(percentile) + '/type' + str(
                type)
            check_dir(result_path)
            result_json = {'model': 'lenet5_model', 'type': type, 'percentile': percentile}
            print(result_json)
            # right
            right_active_data = np.load(
                '../../data/mnist/mnist_right_active_data/' + 'lenet5_full_active_data.npy')
            right_result = process_active_data(right_active_data, 4, percentile, True, type)
            # 保留正确数据的总的覆盖信息数据
            np.save(os.path.join(result_path, 'lenet5_full_right_result.npy'), right_result)
            coverage = calculate_coverage(right_result, get_condition_nums_for_two_layer_data(type))
            result_json['right'] = coverage
            gc_collect(right_active_data)
            # wrong
            for attack_type in ['fgsm', 'gaussian_noise', 'saliency_map', 'uniform_noise']:
                wrong_active_data = np.load(
                    '../../data/mnist/mnist_wrong_active_data/' + attack_type + '/lenet5_full_active_data.npy')
                wrong_result = process_active_data(wrong_active_data, 4, percentile, True, type)
                wrong_results.append(wrong_result)
                # 保留错误数据的总的覆盖信息数据
                np.save(os.path.join(result_path, 'lenet5_full_' + attack_type + '_result.npy'), wrong_result)
                gc_collect(wrong_active_data)
                coverage = calculate_coverage(wrong_result, get_condition_nums_for_two_layer_data(type))
                result_json['wrong_' + attack_type] = coverage
                # all
                all_result = np.bitwise_or(right_result, wrong_result)
                # 保留all数据的总的覆盖信息数据
                np.save(os.path.join(result_path, 'lenet5_full_all_' + attack_type + '_result.npy'), all_result)
                coverage = calculate_coverage(all_result, get_condition_nums_for_two_layer_data(type))
                result_json['all_' + attack_type] = coverage
            # total
            all_wrong_result = bitwise_or(wrong_results)
            total_result = np.bitwise_or(all_wrong_result, right_result)
            np.save(os.path.join(result_path, 'lenet5_full_total_result.npy'), total_result)
            coverage = calculate_coverage(total_result, get_condition_nums_for_two_layer_data(type))
            result_json['total'] = coverage
            with open(os.path.join(result_path, 'lenet5_full_model_result.json'),
                      'w') as file:
                json.dump(result_json, file)
