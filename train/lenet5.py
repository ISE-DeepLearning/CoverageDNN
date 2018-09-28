from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np
from train import data_provider
import json


def create_lenet():
    inputs = Input((28, 28, 1))
    conv1 = Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='valid', activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(conv1)

    conv2 = Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='valid', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(conv2)

    full1 = Flatten()(pool2)  # 卷积层是(batch,rows,cols,channel),为了全连接需要将其展开得到(batch,rows*cols*channel)的平铺层
    full2 = Dense(units=120, activation='relu')(full1)
    full3 = Dense(units=84, activation='relu')(full2)
    outputs = Dense(units=10, activation='softmax')(full3)

    model = Model(inputs=[inputs], outputs=[outputs])
    sgd = SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


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
    # 删除最后一层的softmax/sigmod之类的分类激活函数
    layers = layers[:-1]
    return layers


if __name__ == '__main__':
    # # 基于mnist数据集训练lenet模型
    # lenet5 = create_lenet()
    # print("preparing data ...")
    # # 获取mnist数据
    # mnist_train_datas, mnist_train_labels, mnist_test_datas, mnist_test_labels = data_provider.get_mnist_data()
    # mnist_train_datas = np.reshape(mnist_train_datas, (-1, 28, 28, 1))
    # mnist_test_datas = np.reshape(mnist_test_datas, (-1, 28, 28, 1))
    # print("training start ...")
    # lenet5.fit(mnist_train_datas, mnist_train_labels, batch_size=32, epochs=10, verbose=1, shuffle=True,
    #            validation_data=(mnist_train_datas, mnist_train_labels))
    # # 保存训练完的模型
    # lenet5.save("./test/lenet5.hdf5")

    # 加载训练完的模型
    lenet5_model = load_model('./test/lenet5.hdf5')
    # print(lenet5_model.to_json())
    # print(lenet5_model.trainable_weights)
    # print(lenet5_model.get_weights()[0])
    #
    # print(lenet5_model.get_weights()[1])
    get_activation_layers(lenet5_model)
