# -*- coding: utf-8 -*-
import numpy as np
import foolbox
import keras
from keras.models import load_model
import math
from train import data_provider
import os

# import matplotlib
# import math
#
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt


mnist_attack_data_base_path = '../data/mnist/'
cifar_attack_data_base_path = '../data/cifar/'

threshold = 85


# 计算成绩
def cal_score(original_data, compose_data):
    test_count = 0
    # 28*28 / 32*32 维度 计算得分
    pixels = np.array(original_data).flatten().shape[0]
    for i in range(0, pixels):
        x = (original_data.reshape(pixels)[i] * 255).astype(np.uint8)
        y = (compose_data.reshape(pixels)[i] * 255).astype(np.uint8)
        temp = (int(x) - int(y)) ** 2
        test_count = test_count + temp
    # print(test_count)
    mse_pow = float(test_count) / float(pixels)
    mse = math.sqrt(mse_pow)
    # print(mse)
    # div = mse / 70
    score = 100 / (1 + math.pow(math.e, (mse - 70) / 15))
    # count = np.sum((original_data.reshape(28, 28) - compose_data.reshape(28, 28)) ** 2)
    # # 平方和 / 784
    # mse_pow = float(count) / float(len(original_data.flatten()))
    # mse = math.sqrt(mse_pow)
    # score = 100 / (1 + math.pow(math.e, (mse - 50) / 15))
    return score


# 攻击函数
def start_attack(foolmodel, image, label):
    attack = foolbox.attacks.SaliencyMapAttack(foolmodel)
    image = np.array(image).flatten()
    for theta in np.arange(0.1, 1, 0.1):
        adv = attack(image, label, unpack=True, theta=theta)
        if adv is not None and cal_score(image, adv) >= threshold:
            return adv
        # 返回生成的攻击样本，用于保存


if __name__ == '__main__':
    keras.backend.set_learning_phase(0)
    train_datas, train_labels, test_datas, test_labels = data_provider.get_mnist_data()
    for i in [3, 5, 10]:
        advs = []
        model = load_model('../model/mnist/mnist_' + str(i) + '_hidden_layers_model.hdf5')
        foolmodel = foolbox.models.KerasModel(model, bounds=(0, 1), preprocessing=(0, 1))
        predict_labels = model.predict(test_datas)
        # 找到模型预测的结果
        predict_result = np.argmax(predict_labels, axis=1)
        # 找到测试集预测的结果
        test_result = np.argmax(test_labels, axis=1)
        for j in range(len(test_datas)):
            print(j)
            # 对那些成功的样本进行对抗样本的生成
            if predict_result[j] == test_result[j]:
                adv = start_attack(foolmodel, test_datas[j], test_result[j])
                if adv is not None:
                    advs.append(adv)
        save_path = os.path.join(mnist_attack_data_base_path, 'mnist_attack_data/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, str(i) + '_hidden_layers_model_attack_datas.npy'), advs)
