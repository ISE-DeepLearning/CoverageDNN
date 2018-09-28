import os
import numpy as np
import gc
import json


def bitwise_or(data_array):
    result = data_array[0]
    for data in data_array:
        result = np.bitwise_or(result, data)
    return result


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


m = n = 1


def condition_number(type):
    if type == 0 or type is None:
        return 2 ** (m + n)
    elif type == 1:
        return 2 ** m
    elif type == 2:
        return 1


def calculate_coverage(result, nums):
    result = [bin(n).count('1') for n in result]
    return np.sum(result) / (len(result) * nums)


if __name__ == '__main__':
    for layer_num in [3, 5, 10]:
        for type in [0, 1, 2]:
            conditions = condition_number(type)
            for threshold in [0, 0.25, 0.5, 0.75]:
                result_path = '../../result/optimizer/rq1/threshold' + str(threshold) + '/type' + str(type)
                check_dir(result_path)
                result_json = {'model': str(layer_num) + '_hidden_layers_model', 'type': type, 'threshold': threshold}
                print(result_json)
                # right
                right_active_data = np.load('../../data/mnist/mnist_right_active_data/coverage/threshold' + str(
                    threshold) + '/adjacent_1_1/type' + str(type) + '/' + str(
                    layer_num) + '_hidden_layers_coverage.npy')
                right_active_data = right_active_data.astype(np.int)
                right_result = bitwise_or(right_active_data)
                del right_active_data
                gc.collect()
                coverage = calculate_coverage(right_result, conditions)
                result_json['right'] = coverage

                # wrong
                wrong_results = []
                for attack_type in ['fgsm', 'gaussian_noise', 'saliency_map', 'uniform_noise']:
                    wrong_active_data = np.load(
                        '../../data/mnist/mnist_wrong_active_data/coverage/' + attack_type + '/threshold' + str(
                            threshold) + '/adjacent_1_1/type' + str(type) + '/' + str(
                            layer_num) + '_hidden_layers_coverage.npy')
                    wrong_active_data = wrong_active_data.astype(np.int)
                    wrong_result = bitwise_or(wrong_active_data)
                    del wrong_active_data
                    gc.collect()
                    coverage = calculate_coverage(wrong_result, conditions)
                    result_json['wrong_' + attack_type] = coverage
                    wrong_results.append(wrong_result)
                    # all
                    all_result = np.bitwise_or(right_result, wrong_result)
                    coverage = calculate_coverage(all_result, conditions)
                    result_json['all_' + attack_type] = coverage

                # total
                all_wrong_result = bitwise_or(wrong_results)
                total_result = np.bitwise_or(all_wrong_result, right_result)
                coverage = calculate_coverage(total_result, conditions)
                result_json['total'] = coverage
                with open(os.path.join(result_path, str(layer_num) + '_hidden_layers_model_result.json'), 'w') as file:
                    json.dump(result_json, file)

            for percentile in [25, 50, 75]:
                result_path = '../../result/optimizer/rq1/percentile' + str(percentile) + '/type' + str(type)
                check_dir(result_path)
                result_json = {'model': str(layer_num) + '_hidden_layers_model', 'type': type, 'percentile': percentile}
                print(result_json)
                # right
                right_active_data = np.load('../../data/mnist/mnist_right_active_data/coverage/percentitle' + str(
                    percentile) + '/adjacent_1_1/type' + str(type) + '/' + str(
                    layer_num) + '_hidden_layers_coverage.npy')
                right_active_data = right_active_data.astype(np.int)
                right_result = bitwise_or(right_active_data)
                del right_active_data
                gc.collect()
                coverage = calculate_coverage(right_result, conditions)
                result_json['right'] = coverage
                # wrong
                for attack_type in ['fgsm', 'gaussian_noise', 'saliency_map', 'uniform_noise']:
                    wrong_active_data = np.load(
                        '../../data/mnist/mnist_wrong_active_data/coverage/' + attack_type + '/percentitle' + str(
                            percentile) + '/adjacent_1_1/type' + str(type) + '/' + str(
                            layer_num) + '_hidden_layers_coverage.npy')
                    wrong_active_data = wrong_active_data.astype(np.int)
                    wrong_result = bitwise_or(wrong_active_data)
                    del wrong_active_data
                    gc.collect()
                    coverage = calculate_coverage(wrong_result, conditions)
                    result_json['wrong_' + attack_type] = coverage
                    wrong_results.append(wrong_result)
                    # all
                    all_result = np.bitwise_or(right_result, wrong_result)
                    coverage = calculate_coverage(all_result, conditions)
                    result_json['all_' + attack_type] = coverage

                    # total
                all_wrong_result = bitwise_or(wrong_results)
                total_result = np.bitwise_or(all_wrong_result, right_result)
                coverage = calculate_coverage(total_result, conditions)
                result_json['total'] = coverage
                with open(os.path.join(result_path, str(layer_num) + '_hidden_layers_model_result.json'), 'w') as file:
                    json.dump(result_json, file)
