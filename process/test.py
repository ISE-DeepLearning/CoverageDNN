import numpy as np

import os

if __name__ == '__main__':
    # data_path = '../data/mnist/mnist_right_active_data/coverage/threshold0/adjacent_1_1/type0'
    # save_path = '../data/mnist/mnist_right_active_data/coverage/threshold0/adjacent_1_1/type0'
    # for i in [3, 5, 10]:
    #     datas = np.load(os.path.join(data_path, str(i) + '_hidden_layers_coverage.npy'))
    #     print(datas.shape)
    #     datas = datas[:100]
    #     np.save(os.path.join(data_path, 'temp_' + str(i) + '_hidden_layers_coverage.npy'), datas)
    # a = np.array([
    #     [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # ])
    # print(np.sum(a, axis=0))
    # print(np.sum(a, axis=0).shape)
    # print(np.sum(a, axis=1))
    # print(np.sum(a, axis=1).shape)

    # init = 0.1
    # data = [init + i * 0.1 for i in range(10)]
    # data.append(1095.2**97)
    # if all([np.isfinite(d).all() for d in data]):
    #     print('false')
    # else:
    #     print('true')

    print(np.bitwise_or(8,9))