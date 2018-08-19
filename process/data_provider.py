import os
import numpy as np


def shuffle_mnist_data(datas):
    # np.random.shuffle 是inplace的修改，没有返回值 是对原值的修改
    # np.random.permutation 是基于shuffle的实现，但是是有返回值，是copy了一个新数组
    return np.random.permutation(datas)


def get_all_active_data(base_dir, dataset, hidden_layer_num):
    test_active_path = os.path.join(base_dir, dataset + '_test_active_data',
                                    str(hidden_layer_num) + '_hidden_layers_active_data.npy')
    attack_active_path = os.path.join(base_dir, dataset + '_attack_active_data',
                                      str(hidden_layer_num) + '_hidden_layers_active_data.npy')
    test_active_data = np.load(test_active_path)
    attack_active_data = np.load(attack_active_path)

    return np.concatenate((test_active_data, attack_active_data), axis=0)


if __name__ == '__main__':
    data = get_all_active_data('../data/mnist','mnist',3)
    data = shuffle_mnist_data(data)
    print(data.shape)
