import os
import json
import numpy as np
from keras import Model
from keras.models import load_model
from train import data_provider
from train import config


# 获取模型的激活层模型[softmax层除外]
def get_activation_layers(model):
    layers = []
    model_detail = json.loads(model.to_json())
    layer_detials = model_detail['config']['layers']
    for layer in layer_detials:
        print(layer)
        if layer['class_name'] == 'Activation':
            layer_model = Model(inputs=model.input, outputs=model.get_layer(layer['name']).output)
            layers.append(layer_model)
    # 删除最后一层的softmax/sigmod之类的分类激活函数
    layers = layers[:-1]
    return layers


def process_active_data(model_path, data_path, save_path):
    model = load_model(model_path)
    layers = get_activation_layers(model)
    datas = []
    datas_by_sample = []
    # train_datas, train_labels, test_datas, test_labels = data_provider.get_mnist_data()
    attack_datas = np.load(data_path)
    for layer_model in layers:
        active_datas = layer_model.predict(attack_datas)
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
    np.save(os.path.join(save_dir_path, str(layer_num) + '_hidden_layers_active_data.npy'), datas_by_sample)


if __name__ == '__main__':
    for layer_num in [3, 5, 10]:
        model_path = '../model/mnist/mnist_' + str(layer_num) + '_hidden_layers_model.hdf5'
        data_path = '../data/mnist/mnist_attack_data/' + str(layer_num) + '_hidden_layers_model_attack_datas.npy'
        save_path = 'mnist/mnist_attack_active_data'
        process_active_data(model_path, data_path, save_path)
