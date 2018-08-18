# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import config
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

# 数据集分类数据存储位置
mnist_train_data = config.mnist_train_data
mnist_test_data = config.mnist_test_data
cifar_10_train_data = config.cifar_10_train_data
cifar_10_test_data = config.cifar_10_test_data
cifar_100_train_data = config.cifar_100_train_data
cifar_100_test_data = config.cifar_100_test_data
# 灰度图存储数据存储位置
cifar_10_train_L_data = config.cifar_10_train_L_data
cifar_10_test_L_data = config.cifar_10_test_L_data
cifar_100_train_L_data = config.cifar_100_train_L_data
cifar_100_test_L_data = config.cifar_100_test_L_data


# 检查目录是否存在 存在就保存
def check_path(paths):
    for item in paths:
        path_str = os.path.join(config.data_save_path, item)
        if not os.path.exists(path_str):
            os.makedirs(path_str)


# 分类保存mnist的train和test数据
def save_mnist():
    # mnist 数据集
    mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)
    classes = np.shape(mnist.train.labels)[1]  # 10类
    train_datas = []
    test_datas = []
    # 构造[[],[],[],[],[],[],[],[],[],[]]
    for _ in range(classes):
        train_datas.append([])
        test_datas.append([])

    # 训练集合的数据对应label
    train_labels = np.argmax(mnist.train.labels, axis=1)
    # 测试集合的数据对应label
    test_labels = np.argmax(mnist.test.labels, axis=1)

    print(len(train_labels))
    print(len(test_labels))
    # 分类
    for i in range(len(train_labels)):
        train_datas[train_labels[i]].append(mnist.train.images[i])
    for i in range(len(test_labels)):
        test_datas[test_labels[i]].append(mnist.test.images[i])
    for i in range(len(train_datas)):
        save_path = os.path.join(config.data_save_path, mnist_train_data, str(i) + '.npy')
        np.save(save_path, train_datas[i])
    for i in range(len(test_datas)):
        save_path = os.path.join(config.data_save_path, mnist_test_data, str(i) + '.npy')
        np.save(save_path, test_datas[i])


def save_cifar_10_train():
    # 训练集文件有5个
    batch_file_size = 5
    train_datas = []
    for i in range(1, batch_file_size + 1):
        with open('../data/cifar_data/cifar-10/data_batch_' + str(i), 'rb') as train_data_file:
            train_data = pickle.load(train_data_file, encoding='bytes')
            print(train_data.keys())
            datas = train_data[b'data']
            labels = train_data[b'labels']
            classes = np.max(labels) + 1
            # 只有读第一个batch的时候才新建，其余都是添加操作
            if i == 1:
                for _ in range(classes):
                    train_datas.append([])
            for i in range(len(labels)):
                train_datas[labels[i]].append(datas[i])
    for i in range(classes):
        save_path = os.path.join(config.data_save_path, cifar_10_train_data, str(i) + '.npy')
        np.save(save_path, train_datas[i])


def save_cifar_10_test():
    # cifar 100 数据集
    with open('../data/cifar_data/cifar-10/test_batch', 'rb') as test_data_file:
        test_data = pickle.load(test_data_file, encoding='bytes')
        print(test_data.keys())
        datas = test_data[b'data']
        labels = test_data[b'labels']
        classes = np.max(labels) + 1
        test_datas = []
        for _ in range(classes):
            test_datas.append([])
        for i in range(len(labels)):
            test_datas[labels[i]].append(datas[i])
        for i in range(classes):
            save_path = os.path.join(config.data_save_path, cifar_10_test_data, str(i) + '.npy')
            np.save(save_path, test_datas[i])


def save_cifar_100_train():
    # cifar 100 数据集
    with open('../data/cifar_data/cifar-100/train', 'rb') as train_data_file:
        train_data = pickle.load(train_data_file, encoding='bytes')
        print(train_data.keys())
        datas = train_data[b'data']
        labels = train_data[b'fine_labels']
        classes = np.max(labels) + 1
        train_datas = []
        for _ in range(classes):
            train_datas.append([])

        for i in range(len(labels)):
            train_datas[labels[i]].append(datas[i])
        for i in range(classes):
            save_path = os.path.join(config.data_save_path, cifar_100_train_data, str(i) + '.npy')
            np.save(save_path, train_datas[i])


def save_cifar_100_test():
    # cifar 100 数据集
    with open('../data/cifar_data/cifar-100/test', 'rb') as test_data_file:
        test_data = pickle.load(test_data_file, encoding='bytes')
        print(test_data.keys())
        datas = test_data[b'data']
        labels = test_data[b'fine_labels']
        classes = np.max(labels) + 1
        test_datas = []
        for _ in range(classes):
            test_datas.append([])

        for i in range(len(labels)):
            test_datas[labels[i]].append(datas[i])
        for i in range(classes):
            save_path = os.path.join(config.data_save_path, cifar_100_test_data, str(i) + '.npy')
            np.save(save_path, test_datas[i])


def transfer_to_L(data, shape, transpose_axies):
    data = np.reshape(data, shape)
    data = data.transpose(transpose_axies)
    image = Image.fromarray(data)
    image = image.convert('L')
    # image.save("test.png")
    return np.asarray(image).flatten()


def save_cifar_10_L_data():
    for i in range(10):
        print('cifar-10', i)
        train_path = os.path.join(config.data_save_path, cifar_10_train_data, str(i) + '.npy')
        test_path = os.path.join(config.data_save_path, cifar_10_test_data, str(i) + '.npy')
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        new_train_data = []
        new_test_data = []
        for data in train_data:
            data = transfer_to_L(data, (3, 32, 32), (1, 2, 0))
            new_train_data.append(data)
        for data in test_data:
            data = transfer_to_L(data, (3, 32, 32), (1, 2, 0))
            new_test_data.append(data)
        np.save(os.path.join(config.data_save_path, cifar_10_train_L_data, str(i) + '.npy'), new_train_data)
        np.save(os.path.join(config.data_save_path, cifar_10_test_L_data, str(i) + '.npy'), new_test_data)


def save_cifar_100_L_data():
    for i in range(100):
        print('cifar-100', i)
        train_path = os.path.join(config.data_save_path, cifar_100_train_data, str(i) + '.npy')
        test_path = os.path.join(config.data_save_path, cifar_100_test_data, str(i) + '.npy')
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        new_train_data = []
        new_test_data = []
        for data in train_data:
            data = transfer_to_L(data, (3, 32, 32), (1, 2, 0))
            new_train_data.append(data)
        for data in test_data:
            data = transfer_to_L(data, (3, 32, 32), (1, 2, 0))
            new_test_data.append(data)
        np.save(os.path.join(config.data_save_path, cifar_100_train_L_data, str(i) + '.npy'), new_train_data)
        np.save(os.path.join(config.data_save_path, cifar_100_test_L_data, str(i) + '.npy'), new_test_data)


if __name__ == '__main__':
    check_path([mnist_train_data, mnist_test_data,
                cifar_10_train_data, cifar_100_train_data,
                cifar_10_test_data, cifar_100_test_data,
                cifar_10_train_L_data, cifar_10_test_L_data,
                cifar_100_train_L_data, cifar_100_test_L_data])
    # 存储数据 分类存储 mnist cifar-10 cifar-100
    save_mnist()
    save_cifar_100_train()
    save_cifar_100_test()
    save_cifar_10_train()
    save_cifar_10_test()
    # 根据彩色图片将cifar存为灰度图
    save_cifar_10_L_data()
    save_cifar_100_L_data()

    # 检查数据完整性 粗略检查数据的数量总和是不是跟当时读取的时候一致
    # mnist_test_count = 0
    # mnist_train_count = 0
    # for i in range(10):
    #     train_data_path = os.path.join(config.data_save_path,cifar_10_train_L_data,str(i)+'.npy')
    #     test_data_path = os.path.join(config.data_save_path,cifar_10_test_L_data,str(i)+'.npy')
    #     train_size = np.shape(np.load(train_data_path))[0]
    #     test_size = np.shape(np.load(test_data_path))[0]
    #     print(train_size)
    #     print(test_size)
    #     mnist_train_count += train_size
    #     mnist_test_count += test_size
    # print('mnist_train_size is '+str(mnist_train_count))
    # print('mnist_test_size is '+str(mnist_test_count))

    # 彩色转灰度
    # mnist单个图片的数据之前做过,生成图片的确是数字图片
    # 从cifar图片中取出一个数据查看是否能生成对应的图片
    # datas = np.load(os.path.join(config.data_save_path,cifar_10_train_data,'0.npy'))
    # data = np.reshape(datas[1], (3,32,32))
    # data = data.transpose((1,2,0))
    # image = Image.fromarray(data)
    # image = image.convert('L')
    # # image.save("test.png")
    # np.asarray(image)

    # 查看灰度图保存的数据是否能够输出为对应的图片
    # datas = np.load(os.path.join(config.data_save_path,cifar_10_train_L_data,'0.npy'))
    # data = np.reshape(datas[1], (32,32))
    # image = Image.fromarray(data)
    # image = image.convert('L')
    # image.save('test2.png')
