import os
import numpy as np


def shuffle_mnist_data(datas):
    # np.random.shuffle 是inplace的修改，没有返回值 是对原值的修改
    # np.random.permutation 是基于shuffle的实现，但是是有返回值，是copy了一个新数组
    return np.random.permutation(datas)


def get_class_test_data(base_dir, dataset, i):
    return np.load(os.path.join(base_dir, dataset + '_test_data', str(i) + '.npy'))


def get_all_active_data(base_dir, dataset, hidden_layer_num):
    test_active_path = os.path.join(base_dir, dataset + '_test_active_data',
                                    str(hidden_layer_num) + '_hidden_layers_active_data.npy')
    attack_active_path = os.path.join(base_dir, dataset + '_attack_active_data',
                                      str(hidden_layer_num) + '_hidden_layers_active_data.npy')
    test_active_data = np.load(test_active_path)
    attack_active_data = np.load(attack_active_path)

    return np.concatenate((test_active_data, attack_active_data), axis=0)


def get_all_active_data_with_attack_type(base_dir, dataset, hidden_layer_num, type='fgsm'):
    right_active_path = os.path.join(base_dir, dataset + '_right_active_data',
                                     str(hidden_layer_num) + '_hidden_layers_active_data.npy')
    wrong_active_path = os.path.join(base_dir, dataset + '_wrong_active_data', type,
                                     str(hidden_layer_num) + '_hidden_layers_active_data.npy')
    right_active_data = np.load(right_active_path)
    if type == 'all':
        return get_all_active_data_with_attack_types(base_dir, dataset, hidden_layer_num)
    wrong_active_data = np.load(wrong_active_path)

    return np.concatenate((right_active_data, wrong_active_data), axis=0)


def get_all_active_data_with_attack_types(base_dir, dataset, hidden_layer_num,
                                          types=['fgsm', 'gaussian_noise', 'saliency_map', 'uniform_noise']):
    right_active_path = os.path.join(base_dir, dataset + '_right_active_data',
                                     str(hidden_layer_num) + '_hidden_layers_active_data.npy')
    right_active_data = np.load(right_active_path)
    for type in types:
        wrong_active_path = os.path.join(base_dir, dataset + '_wrong_active_data', type,
                                         str(hidden_layer_num) + '_hidden_layers_active_data.npy')
        wrong_active_data = np.load(wrong_active_path)
        right_active_data = np.concatenate((right_active_data, wrong_active_data), axis=0)
    return right_active_data


def get_right_active_data(base_dir, dataset, hidden_layer_num):
    right_active_path = os.path.join(base_dir, dataset + '_right_active_data',
                                     str(hidden_layer_num) + '_hidden_layers_active_data.npy')
    right_active_data = np.load(right_active_path)
    return right_active_data


def get_wrong_active_data(base_dir, dataset, hidden_layer_num):
    wrong_active_path = os.path.join(base_dir, dataset + '_wrong_active_data',
                                     str(hidden_layer_num) + '_hidden_layers_active_data.npy')
    wrong_active_data = np.load(wrong_active_path)
    return wrong_active_data


def get_wrong_active_data_with_attack_type(base_dir, dataset, hidden_layer_num, type='fgsm'):
    wrong_active_path = os.path.join(base_dir, dataset + '_wrong_active_data', type,
                                     str(hidden_layer_num) + '_hidden_layers_active_data.npy')
    wrong_active_data = np.load(wrong_active_path)
    return wrong_active_data


if __name__ == '__main__':
    data = get_all_active_data('../data/mnist', 'mnist', 3)
    data = shuffle_mnist_data(data)
    print(data.shape)
