import os
import numpy as np
from train import config


# 训练数据 测试数据提供
# classes 分类数量
# input_size 图像维度/输入维度
# max_pixel 最大数值 0-1/0-255
def get_mnist_data():
    classes = 10
    input_size = 784
    max_pixel = 1
    train_dir_path = os.path.join(config.data_save_path, config.mnist_train_data)
    test_dir_path = os.path.join(config.data_save_path, config.mnist_test_data)
    train_datas = np.empty(shape=(0, input_size))
    train_labels = np.empty(shape=(0, classes))
    test_datas = np.empty(shape=(0, input_size))
    test_labels = np.empty(shape=(0, classes))
    for i in range(classes):
        temp_train_datas = np.load(os.path.join(train_dir_path, str(i) + '.npy'))
        temp_train_labels = np.zeros(shape=(temp_train_datas.shape[0], classes))
        temp_train_labels[:, i] = 1
        temp_test_datas = np.load(os.path.join(test_dir_path, str(i) + '.npy'))
        temp_test_labels = np.zeros(shape=(temp_test_datas.shape[0], classes))
        temp_test_labels[:, i] = 1

        train_datas = np.concatenate((train_datas, temp_train_datas))
        train_labels = np.concatenate((train_labels, temp_train_labels))
        test_datas = np.concatenate((test_datas, temp_test_datas))
        test_labels = np.concatenate((test_labels, temp_test_labels))
    return train_datas / max_pixel, train_labels, test_datas / max_pixel, test_labels


def get_cifar_10_data():
    classes = 10
    input_size = 1024
    max_pixel = 255
    train_dir_path = os.path.join(config.data_save_path, config.cifar_10_train_L_data)
    test_dir_path = os.path.join(config.data_save_path, config.cifar_10_test_L_data)
    train_datas = np.empty(shape=(0, input_size))
    train_labels = np.empty(shape=(0, classes))
    test_datas = np.empty(shape=(0, input_size))
    test_labels = np.empty(shape=(0, classes))
    for i in range(classes):
        temp_train_datas = np.load(os.path.join(train_dir_path, str(i) + '.npy'))
        temp_train_labels = np.zeros(shape=(temp_train_datas.shape[0], classes))
        temp_train_labels[:, i] = 1
        temp_test_datas = np.load(os.path.join(test_dir_path, str(i) + '.npy'))
        temp_test_labels = np.zeros(shape=(temp_test_datas.shape[0], classes))
        temp_test_labels[:, i] = 1

        train_datas = np.concatenate((train_datas, temp_train_datas))
        train_labels = np.concatenate((train_labels, temp_train_labels))
        test_datas = np.concatenate((test_datas, temp_test_datas))
        test_labels = np.concatenate((test_labels, temp_test_labels))
    return train_datas / max_pixel, train_labels, test_datas / max_pixel, test_labels


def get_cifar_100_data():
    classes = 100
    input_size = 1024
    max_pixel = 255
    train_dir_path = os.path.join(config.data_save_path, config.cifar_100_train_L_data)
    test_dir_path = os.path.join(config.data_save_path, config.cifar_100_test_L_data)
    train_datas = np.empty(shape=(0, input_size))
    train_labels = np.empty(shape=(0, classes))
    test_datas = np.empty(shape=(0, input_size))
    test_labels = np.empty(shape=(0, classes))
    for i in range(classes):
        temp_train_datas = np.load(os.path.join(train_dir_path, str(i) + '.npy'))
        temp_train_labels = np.zeros(shape=(temp_train_datas.shape[0], classes))
        temp_train_labels[:, i] = 1
        temp_test_datas = np.load(os.path.join(test_dir_path, str(i) + '.npy'))
        temp_test_labels = np.zeros(shape=(temp_test_datas.shape[0], classes))
        temp_test_labels[:, i] = 1

        train_datas = np.concatenate((train_datas, temp_train_datas))
        train_labels = np.concatenate((train_labels, temp_train_labels))
        test_datas = np.concatenate((test_datas, temp_test_datas))
        test_labels = np.concatenate((test_labels, temp_test_labels))
    return train_datas / max_pixel, train_labels, test_datas / max_pixel, test_labels


if __name__ == '__main__':
    # data = np.zeros(shape=(5, 10))
    # print(data)
    # # 任意行 的第一个元素设置为1  也就是将第一列设置为1
    # data[:, 1] = 1
    # print(data)
    # train_data, train_label, test_data, test_label = get_mnist_data()
    train_data, train_label, test_data, test_label = get_cifar_10_data()
    # train_data, train_label, test_data, test_label = get_cifar_100_data()
    print(train_data.shape)
    print(test_data.shape)
    print(train_label.shape)
    print(test_label.shape)
