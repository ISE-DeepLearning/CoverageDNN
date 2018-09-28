import os
import numpy as np

import gc

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt

'''
实验三 selection的试验
分为三种selection
1. random selection 无需做，由概率获得 k= wrong/all
2. additional selection 取数据里面跟已经选择的样本们的交集最大
3. total selection 直接取得数据中单一最大的覆盖情况
'''

attack_type = 'gaussian_noise'
threshold = 0
type = 1
m = n = 1
base_dir_path = '../data/mnist'

# additional
if __name__ == '__main__':
    for layer_num in [3, 5, 10]:
        wrong_coverage_data_path = os.path.join(base_dir_path,'mnist_wrong_active_data/coverage/' + attack_type + '/threshold' + str(threshold) + '/adjacent_' + str(m) + '_' + str(n) + '/type' + str(type))
        print(wrong_coverage_data_path)
        # str(layer_num)_hidden_layers_coverage.npy
        right_coverage_data_path = os.path.join(base_dir_path,'mnist_right_active_data/coverage/threshold' + str(threshold) + '/adjacent_' + str(m) + '_' + str(n) + '/type' + str(type))
        print(right_coverage_data_path)
        # str(layer_num)_hidden_layers_coverage.npy

        pass
