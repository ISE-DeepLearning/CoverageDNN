import os
import numpy as np
from keras.models import load_model
import gc


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def split_data(dataset, layer_nums, classes=10):
    model = load_model(os.path.join('../model', dataset, dataset + '_' + str(layer_nums) + '_hidden_layers_model.hdf5'))
    attack_datas = np.load(os.path.join('../data', dataset, dataset + '_attack_data',
                                        str(layer_nums) + '_hidden_layers_model_attack_datas.npy'))
    origin_test_datas = []
    origin_test_labels = []
    for i in range(classes):
        datas = np.load(os.path.join('../data/', dataset, dataset + '_test_data', str(i) + '.npy'))
        labels = np.zeros(shape=(len(datas),))
        labels[:] = i
        origin_test_datas.extend(list(datas))
        origin_test_labels.extend(list(labels))
    origin_test_datas = np.array(origin_test_datas)
    predict_result = model.predict(origin_test_datas)
    predict_labels = np.argmax(predict_result, axis=1)
    # print(np.shape(origin_test_labels))
    # print(np.shape(predict_labels))
    # print(np.where(origin_test_labels != predict_labels))
    print(len(attack_datas) + len(origin_test_datas))
    # 返回识别正确的数据和所有识别错误的数据(原始测试集中识别错误的数据集以及)
    right_datas = origin_test_datas[origin_test_labels == predict_labels]
    wrong_datas = np.concatenate((attack_datas, origin_test_datas[origin_test_labels != predict_labels]))
    return right_datas, wrong_datas


def split_data_by_attack_type(dataset, layer_nums, attack_type, classes=10):
    model = load_model(os.path.join('../model', dataset, dataset + '_' + str(layer_nums) + '_hidden_layers_model.hdf5'))
    attack_datas = []
    for i in range(classes):
        attack_temp_datas = np.load(os.path.join('../data', dataset, dataset + '_attack_data', attack_type,
                                                 str(layer_nums) + '_hidden_layers_model_attack_' + str(
                                                     i) + '_datas.npy'))
        attack_datas.extend(list(attack_temp_datas))
    attack_datas = np.array(attack_datas)
    origin_test_datas = []
    origin_test_labels = []
    for i in range(classes):
        datas = np.load(os.path.join('../data/', dataset, dataset + '_test_data', str(i) + '.npy'))
        labels = np.zeros(shape=(len(datas),))
        labels[:] = i
        origin_test_datas.extend(list(datas))
        origin_test_labels.extend(list(labels))
    origin_test_datas = np.array(origin_test_datas)
    predict_result = model.predict(origin_test_datas)
    predict_labels = np.argmax(predict_result, axis=1)
    # print(np.shape(origin_test_labels))
    # print(np.shape(predict_labels))
    # print(np.where(origin_test_labels != predict_labels))
    print(len(attack_datas) + len(origin_test_datas))
    # 返回识别正确的数据和所有识别错误的数据(原始测试集中识别错误的数据集以及)
    right_datas = origin_test_datas[origin_test_labels == predict_labels]
    wrong_datas = np.concatenate((attack_datas, origin_test_datas[origin_test_labels != predict_labels]))
    return right_datas, wrong_datas


def split_data_lenet5(dataset, layer_nums, classes=10):
    model = load_model(os.path.join('../model', dataset, 'lenet5.hdf5'))
    attack_datas = np.load(os.path.join('../data', dataset, dataset + '_attack_data',
                                        'lenet5_attack_datas.npy'))
    origin_test_datas = []
    origin_test_labels = []
    for i in range(classes):
        datas = np.load(os.path.join('../data/', dataset, dataset + '_test_data', str(i) + '.npy'))
        labels = np.zeros(shape=(len(datas),))
        labels[:] = i
        origin_test_datas.extend(list(datas))
        origin_test_labels.extend(list(labels))
    origin_test_datas = np.array(origin_test_datas)
    predict_result = model.predict(origin_test_datas)
    predict_labels = np.argmax(predict_result, axis=1)
    # print(np.shape(origin_test_labels))
    # print(np.shape(predict_labels))
    # print(np.where(origin_test_labels != predict_labels))
    print(len(attack_datas) + len(origin_test_datas))
    # 返回识别正确的数据和所有识别错误的数据(原始测试集中识别错误的数据集以及)
    right_datas = origin_test_datas[origin_test_labels == predict_labels]
    wrong_datas = np.concatenate((attack_datas, origin_test_datas[origin_test_labels != predict_labels]))
    return right_datas, wrong_datas


def split_data_by_attack_type_lenet5(dataset, attack_type, classes=10):
    model = load_model(os.path.join('../model', dataset, 'lenet5.hdf5'))
    attack_datas = []
    for i in range(classes):
        attack_temp_datas = np.load(os.path.join('../data', dataset, dataset + '_attack_data', attack_type,
                                                 'lenet5_attack_' + str(i) + '_datas.npy'))
        print(attack_temp_datas.shape)
        attack_datas.extend(list(attack_temp_datas))
    attack_datas = np.array(attack_datas)
    attack_datas = np.reshape(attack_datas,(-1,784))
    print(attack_datas.shape)
    origin_test_datas = []
    origin_test_labels = []
    for i in range(classes):
        datas = np.load(os.path.join('../data/', dataset, dataset + '_test_data', str(i) + '.npy'))
        labels = np.zeros(shape=(len(datas),))
        labels[:] = i
        origin_test_datas.extend(list(datas))
        origin_test_labels.extend(list(labels))

    origin_test_datas = np.array(origin_test_datas)
    print(origin_test_datas.shape)
    predict_result = model.predict(np.reshape(origin_test_datas, (-1, 28, 28, 1)))
    predict_labels = np.argmax(predict_result, axis=1)
    # print(np.shape(origin_test_labels))
    # print(np.shape(predict_labels))
    # print(np.where(origin_test_labels != predict_labels))
    print(len(attack_datas) + len(origin_test_datas))
    # 返回识别正确的数据和所有识别错误的数据(原始测试集中识别错误的数据集以及)
    right_datas = origin_test_datas[origin_test_labels == predict_labels]
    wrong_datas = np.concatenate((attack_datas, origin_test_datas[origin_test_labels != predict_labels]))
    print(len(right_datas))
    print(len(wrong_datas))
    return right_datas, wrong_datas


# 清洗数据将识别正确的数据和识别错误的数据分离出来
if __name__ == '__main__':
    # for layer_nums in [3, 5, 10]:
    #     right_datas, wrong_datas = split_data('mnist', layer_nums)
    #     # save right and wrong data
    #     save_right_path = os.path.join('../data/mnist', 'mnist_right_data')
    #     save_wrong_path = os.path.join('../data/mnist', 'mnist_wrong_data')
    #     check_dir(save_right_path)
    #     check_dir(save_wrong_path)
    #     np.save(os.path.join(save_right_path, str(layer_nums) + '_hidden_layers_right_data.npy'), right_datas)
    #     np.save(os.path.join(save_wrong_path, str(layer_nums) + '_hidden_layers_wrong_data.npy'), wrong_datas)
    #     print(len(right_datas))
    #     print(len(wrong_datas))

    # for layer_nums in [3, 5, 10]:
    #     for attack_type in ['fgsm', 'gaussian_noise', 'saliency_map', 'uniform_noise']:
    #         right_datas, wrong_datas = split_data_by_attack_type('mnist', layer_nums, attack_type)
    #         # save right and wrong data
    #         save_right_path = os.path.join('../data/mnist', 'mnist_right_data')
    #         save_wrong_path = os.path.join('../data/mnist', 'mnist_wrong_data', attack_type)
    #         check_dir(save_right_path)
    #         check_dir(save_wrong_path)
    #         np.save(os.path.join(save_right_path, str(layer_nums) + '_hidden_layers_right_data.npy'), right_datas)
    #         np.save(os.path.join(save_wrong_path, str(layer_nums) + '_hidden_layers_wrong_data.npy'), wrong_datas)
    #         print(len(right_datas))
    #         print(len(wrong_datas))
    #         del right_datas
    #         del wrong_datas
    #         gc.collect()

    for attack_type in ['fgsm', 'gaussian_noise', 'saliency_map', 'uniform_noise']:
        right_datas, wrong_datas = split_data_by_attack_type_lenet5('mnist', attack_type)
        # save right and wrong data
        save_right_path = os.path.join('../data/mnist', 'mnist_right_data')
        save_wrong_path = os.path.join('../data/mnist', 'mnist_wrong_data', attack_type)
        check_dir(save_right_path)
        check_dir(save_wrong_path)
        np.save(os.path.join(save_right_path, 'lenet5_right_data.npy'), right_datas)
        np.save(os.path.join(save_wrong_path, 'lenet5_wrong_data.npy'), wrong_datas)
        print(len(right_datas))
        print(len(wrong_datas))
        print('----------------------------------------------------------------------')
        del right_datas
        del wrong_datas
        gc.collect()
