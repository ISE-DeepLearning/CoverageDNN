from keras import Model, Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model
from keras.optimizers import SGD, Adam
from tensorflow.examples.tutorials.mnist import input_data as inputData
import numpy as np
import json
import os
from model_arch import ModelArch
import data_provider

import config


def model_train_process(model_arch, train_images, train_labels, test_images, test_labels, batch_size=256,
                        epochs=10):
    input_data = Input(model_arch.input_size)
    temp_data = input_data
    # 输入层dropout 0.2
    temp_data = Dropout(0.2)(temp_data)
    for item in model_arch.model_arch:
        if item['name'] == 'dense':
            temp_data = Dense(item['size'])(temp_data)
        elif item['name'] == 'activation':
            temp_data = Activation(item['type'])(temp_data)
            # dropout
            # temp_data = Dropout(0.2)(temp_data)
        else:
            raise RuntimeError("暂时不支持该name建层:" + item['name'])
    output_data = temp_data
    model = Model(inputs=[input_data], outputs=[output_data])
    modelcheck = ModelCheckpoint(model_arch.save_path, monitor='loss', verbose=1, save_best_only=True)
    sgd = SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # significant params [batch_size,epochs]
    hist = model.fit([train_images], [train_labels], batch_size=batch_size, epochs=epochs, callbacks=[modelcheck],
                     validation_data=(test_images, test_labels))

    train_score = model.evaluate(train_images, train_labels, verbose=0)
    score = model.evaluate(test_images, test_labels, verbose=0)
    print(train_score[0], train_score[1])
    print('测试集 score(val_loss): %.4f' % score[0])
    print('测试集 accuracy: %.4f' % score[1])
    # result.append({'path': model_save_path, 'val_loss': score[0], 'acc': score[1]})
    print("train DNN done, save at " + model_arch.save_path)

    # 存储目录拆分
    (filepath, tempfilename) = os.path.split(model_arch.save_path)
    (filename, extension) = os.path.splitext(tempfilename)
    # accuracy的history存储目录生成
    hist_save_path = os.path.join(filepath, filename + '.json')
    # 存储history
    with open(hist_save_path, 'w') as outfile:
        json.dump(hist.history, outfile, ensure_ascii=False)
        outfile.write('\n')
    return hist


# 检查目录是否存在 存在就保存
def check_path(paths):
    for item in paths:
        path_str = os.path.join(config.data_save_path, item)
        if not os.path.exists(path_str):
            os.makedirs(path_str)


if __name__ == '__main__':
    # model = load_model('model.hdf5')
    # model_dict = json.loads(model.to_json())
    # print(type(model_dict))
    # print(type(model_dict['config']['layers']))
    # for i in range(len(model_dict['config']['layers'])):
    #     print(model_dict['config']['layers'][i])
    # mnist = inputData.read_data_sets("MNIST_data/", one_hot=True)
    # # 模型应该存储的位置
    # model_save_dir = os.path.join(config.model_save_path, config.dataset_name)
    # if not os.path.exists(model_save_dir):
    #     os.makedirs(model_save_dir)
    # hist = model_train_process(os.path.join(model_save_dir, 'model_dnn_3_hidden_layers.hdf5'),
    #                            mnist.train.images, mnist.train.labels, mnist.test.images,
    #                            mnist.test.labels)
    #
    # print(np.shape(mnist.train.images))  # 55000,784
    # print(np.shape(mnist.train.labels))  # 55500,10
    # print(np.shape(mnist.test.images))  # 10000,784
    # print(np.shape(mnist.test.labels))  # 10000,10
    # print(str(hist))
    # # print(dict(hist))
    # print(str(hist.history))
    # print(dict(hist.history))

    mnist_model_path = os.path.join(config.model_save_path, "mnist")
    cifar10_model_path = os.path.join(config.model_save_path, "cifar-10")
    cifar100_model_path = os.path.join(config.model_save_path, "cifar-100")
    check_path([mnist_model_path, cifar10_model_path, cifar100_model_path])
    # 组合几组要训练的模型
    # model architecture
    model_arch_01 = [
        {'name': 'dense', 'size': 128}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 64}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 32}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 10}, {'name': 'activation', 'type': 'softmax'}]
    model_arch_02 = [
        {'name': 'dense', 'size': 128}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 96}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 64}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 32}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 24}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 10}, {'name': 'activation', 'type': 'softmax'}]
    model_arch_03 = [
        {'name': 'dense', 'size': 256}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 128}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 64}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 64}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 64}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 64}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 64}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 56}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 32}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 24}, {'name': 'activation', 'type': 'relu'},
        {'name': 'dense', 'size': 10}, {'name': 'activation', 'type': 'softmax'}]

    mnist_3_hidden_layers_model = ModelArch('mnist', (28 * 28,), model_arch_01,
                                            os.path.join(mnist_model_path, "mnist_3_hidden_layers_model.hdf5"))
    mnist_5_hidden_layers_model = ModelArch('mnist', (28 * 28,), model_arch_02,
                                            os.path.join(mnist_model_path, "mnist_5_hidden_layers_model.hdf5"))
    mnist_10_hidden_layers_model = ModelArch('mnist', (28 * 28,), model_arch_03,
                                             os.path.join(mnist_model_path, "mnist_10_hidden_layers_model.hdf5"))
    cifar_10_3_hidden_layers_model = ModelArch('cifar-10', (32 * 32,), model_arch_01,
                                               os.path.join(cifar10_model_path, 'cifar_10_3_hidden_layers_model.hdf5'))
    cifar_10_5_hidden_layers_model = ModelArch('cifar-10', (32 * 32,), model_arch_01,
                                               os.path.join(cifar10_model_path, 'cifar_10_5_hidden_layers_model.hdf5'))
    cifar_10_10_hidden_layers_model = ModelArch('cifar-10', (32 * 32,), model_arch_01,
                                                os.path.join(cifar10_model_path,
                                                             'cifar_10_10_hidden_layers_model.hdf5'))

    print("preparing data ...")
    # 获取mnist数据
    mnist_train_datas, mnist_train_labels, mnist_test_datas, mnist_test_labels = data_provider.get_mnist_data()
    # 获取cifar数据
    cifar_10_train_datas, cifar_10_train_labels, cifar_10_test_datas, cifar_10_test_labels = data_provider.get_cifar_10_data()
    print("training start ...")

    # 训练mnist
    # model_train_process(mnist_3_hidden_layers_model, mnist_train_datas, mnist_train_labels, mnist_test_datas,
    #                     mnist_test_labels)
    # model_train_process(mnist_5_hidden_layers_model, mnist_train_datas, mnist_train_labels, mnist_test_datas,
    #                     mnist_test_labels)
    # model_train_process(mnist_10_hidden_layers_model, mnist_train_datas, mnist_train_labels, mnist_test_datas,
    #                     mnist_test_labels)

    # 训练cifar-10
    model_train_process(cifar_10_3_hidden_layers_model, cifar_10_train_datas, cifar_10_train_labels,
                        cifar_10_test_datas, cifar_10_test_labels, batch_size=128, epochs=300)
    model_train_process(cifar_10_5_hidden_layers_model, cifar_10_train_datas, cifar_10_train_labels,
                        cifar_10_test_datas, cifar_10_test_labels, batch_size=128, epochs=300)
    model_train_process(cifar_10_10_hidden_layers_model, cifar_10_train_datas, cifar_10_train_labels,
                        cifar_10_test_datas, cifar_10_test_labels, batch_size=128, epochs=300)

    # cifar_100_model_arch_01 = [
    #               {'name': 'dense', 'size': 128}, {'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 64}, {'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 32}, {'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 10}, {'name': 'activation', 'type': 'softmax'}]
    # cifar_100_model_arch_02 = [
    #               {'name': 'dense', 'size': 128},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 96},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 64},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 32},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 24},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 10},{'name': 'activation', 'type': 'softmax'}]
    # cifar_100_model_arch_03 = [
    #               {'name': 'dense', 'size': 256},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 128}, {'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 64}, {'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 64}, {'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 64},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 64},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 64},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 56},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 32},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 24},{'name': 'activation', 'type': 'relu'},
    #               {'name': 'dense', 'size': 10},{'name': 'activation', 'type': 'softmax'}]
