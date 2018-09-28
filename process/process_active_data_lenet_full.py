import os
import json
import numpy as np
from keras import Model
from keras.models import load_model
import sys

sys.path.append('../')
from train import data_provider
from train import config


# 获取模型的激活层模型[softmax层除外]
def get_activation_layers(model):
    layers = []
    model_detail = json.loads(model.to_json())
    layer_detials = model_detail['config']['layers']
    for layer in layer_detials:
        # print(layer)
        if 'activation' in layer['config'].keys() and layer['config']['activation'] == 'relu':
            print(layer['name'])
            layer_model = Model(inputs=model.input, outputs=model.get_layer(layer['name']).output)
            layers.append(layer_model)
    # 默认只筛选了relu的层
    # 删除最后一层的softmax/sigmod之类的分类激活函数
    # layers = layers[:-1]
    return layers


def process_active_data(model_path, data_path, save_path):
    model = load_model(model_path)
    layers = get_activation_layers(model)
    datas = []
    datas_by_sample = []
    # train_datas, train_labels, test_datas, test_labels = data_provider.get_mnist_data()
    attack_datas = np.load(data_path)
    for layer_model in layers:
        print(layer_model.to_json())
        model_arch = json.loads(layer_model.to_json())
        active_datas = layer_model.predict(attack_datas.reshape((-1, 28, 28, 1)))
        print(active_datas.shape)
        # print(model_arch['config']['layers'])
        if model_arch['config']['layers'][-1]['class_name'] == 'Conv2D':
            # # 对于卷积层 每层取平均值
            # active_datas = active_datas.mean(axis=(1, 2))
            active_datas = active_datas.reshape((len(attack_datas), -1))
            # print(active_datas.shape)
            # # print(active_datas.sum(axis=0))
            # print(active_datas.mean(axis=(1, 2)).shape)
            # # print(active_datas.sum(axis=1))
            # # print(active_datas.sum(axis=1).shape)
        print(active_datas.shape)
        datas.append(active_datas)
        # print(active_datas)
    print(len(datas[0]))
    print(len(datas))
    for i in range(len(datas[0])):
        data = []
        for j in range(len(datas)):
            data.append(datas[j][i])
        data = np.array(data)
        datas_by_sample.append(data)
        # print(data.shape)
        # print(data)
    print(np.shape(datas_by_sample))
    save_dir_path = os.path.join(config.data_save_path, save_path)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    np.save(os.path.join(save_dir_path, 'lenet5_full_active_data.npy'), datas_by_sample)


# def process_active_data_with_label(model_path, data_path, save_path, label=0):
#     model = load_model(model_path)
#     layers = get_activation_layers(model)
#     datas = []
#     datas_by_sample = []
#     # train_datas, train_labels, test_datas, test_labels = data_provider.get_mnist_data()
#     attack_datas = np.load(data_path)
#     print('label ' + str(label) + ' data nums are ' + str(len(attack_datas)))
#     for layer_model in layers:
#         active_datas = layer_model.predict(attack_datas)
#         print(active_datas.shape)
#         datas.append(active_datas)
#         # print(active_datas)
#     print(len(datas[0]))
#     print(len(datas))
#     for i in range(len(datas[0])):
#         data = []
#         for j in range(len(datas)):
#             data.append(datas[j][i])
#         data = np.array(data)
#         datas_by_sample.append(data)
#         # print(data.shape)
#         # print(data)
#     print(np.shape(datas_by_sample))
#     save_dir_path = os.path.join(config.data_save_path, save_path)
#     if not os.path.exists(save_dir_path):
#         os.makedirs(save_dir_path)
#     np.save(os.path.join(save_dir_path, str(layer_num) + '_hidden_layers_active_' + str(label) + '_data.npy'),
#             datas_by_sample)


if __name__ == '__main__':
    # process save attack active datas
    # for layer_num in [3, 5, 10]:
    #     model_path = '../model/mnist/mnist_' + str(layer_num) + '_hidden_layers_model.hdf5'
    #     data_path = '../data/mnist/mnist_attack_data/' + str(layer_num) + '_hidden_layers_model_attack_datas.npy'
    #     save_path = 'mnist/mnist_attack_active_data'
    #     process_active_data(model_path, data_path, save_path)

    # process different kind of active data
    # classes = 10
    # for layer_num in [3, 5, 10]:
    #     print(str(layer_num) + ' layers')
    #     for attack_type in ['fgsm', 'gaussian_noise', 'saliency_map', 'uniform_noise']:
    #         print('attack type is ' + attack_type)
    #         model_path = '../model/mnist/mnist_' + str(layer_num) + '_hidden_layers_model.hdf5'
    #         for i in range(classes):
    #             data_path = '../data/mnist/mnist_attack_data/' + attack_type + '/' + str(
    #                 layer_num) + '_hidden_layers_model_attack_' + str(i) + '_datas.npy'
    #             save_path = 'mnist/mnist_attack_active_data/' + attack_type
    #             process_active_data_with_label(model_path, data_path, save_path, i)

    # process right and wrong active data
    # for layer_num in [3, 5, 10]:
    #     model_path = '../model/mnist/mnist_' + str(layer_num) + '_hidden_layers_model.hdf5'
    #     data_path = '../data/mnist/mnist_right_data/' + str(layer_num) + '_hidden_layers_right_data.npy'
    #     save_path = 'mnist/mnist_right_active_data/'
    #     process_active_data(model_path, data_path, save_path)
    #     data_path = '../data/mnist/mnist_wrong_data/' + str(layer_num) + '_hidden_layers_wrong_data.npy'
    #     save_path = 'mnist/mnist_wrong_active_data/'
    #     process_active_data(model_path, data_path, save_path)

    # process wrong active data from different attack
    for attack_type in ['fgsm', 'gaussian_noise', 'saliency_map', 'uniform_noise']:
        model_path = '../model/mnist/lenet5.hdf5'
        data_path = '../data/mnist/mnist_wrong_data/' + attack_type + '/lenet5_wrong_data.npy'
        save_path = 'mnist/mnist_wrong_active_data/' + attack_type
        process_active_data(model_path, data_path, save_path)

    # process right active data
    model_path = '../model/mnist/lenet5.hdf5'
    data_path = '../data/mnist/mnist_right_data/lenet5_right_data.npy'
    save_path = 'mnist/mnist_right_active_data/'
    process_active_data(model_path, data_path, save_path)
