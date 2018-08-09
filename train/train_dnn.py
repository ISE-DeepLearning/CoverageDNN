from keras import Model, Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation
from keras.models import load_model
from tensorflow.examples.tutorials.mnist import input_data as inputData
import numpy as np
import json
import os

import config


def model_train_process(model_save_path, train_images, train_labels, test_images, test_labels, batch_size=256,
                        epochs=10):
    input_data = Input(config.input_size)
    temp_data = input_data
    for item in config.model_arch:
        if item['name'] == 'dense':
            temp_data = Dense(item['size'])(temp_data)
        elif item['name'] == 'activation':
            temp_data = Activation(item['type'])(temp_data)
        else:
            raise RuntimeError("暂时不支持该name建层:" + item['name'])
    output_data = temp_data
    model = Model(inputs=[input_data], outputs=[output_data])
    modelcheck = ModelCheckpoint(model_save_path, monitor='loss', verbose=1, save_best_only=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # significant params [batch_size,epochs]
    hist = model.fit([train_images], [train_labels], batch_size=batch_size, epochs=epochs, callbacks=[modelcheck],
                     validation_data=(test_images, test_labels))

    train_score = model.evaluate(train_images, train_labels, verbose=0)
    score = model.evaluate(test_images, test_labels, verbose=0)
    print(train_score[0], train_score[1])
    print('测试集 score(val_loss): %.4f' % score[0])
    print('测试集 accuracy: %.4f' % score[1])
    # result.append({'path': model_save_path, 'val_loss': score[0], 'acc': score[1]})
    print("train DNN done, save at " + model_save_path)
    # return hist


if __name__ == '__main__':
    # model = load_model('model.hdf5')
    # model_dict = json.loads(model.to_json())
    # print(type(model_dict))
    # print(type(model_dict['config']['layers']))
    # for i in range(len(model_dict['config']['layers'])):
    #     print(model_dict['config']['layers'][i])
    mnist = inputData.read_data_sets("MNIST_data/", one_hot=True)
    # 模型应该存储的位置
    model_save_dir = os.path.join(config.model_save_path, config.dataset_name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    hist = model_train_process(os.path.join(model_save_dir, 'model_dnn_3_hidden_layers.hdf5'),
                               mnist.train.images, mnist.train.labels, mnist.test.images,
                               mnist.test.labels)

    print(np.shape(mnist.train.images))  # 55000,784
    print(np.shape(mnist.train.labels))  # 55500,10
    print(np.shape(mnist.test.images))  # 10000,784
    print(np.shape(mnist.test.labels))  # 10000,10
    print(str(hist))
    # print(dict(hist))
    print(str(hist.history))
    print(dict(hist.history))
