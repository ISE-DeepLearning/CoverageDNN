# mnist 数据集
dataset_name = 'mnist'
# input size
input_size = (28 * 28,)
# output_size = 10
# model architecture
model_arch = [{'name': 'dense', 'size': 128},
              {'name': 'activation', 'type': 'relu'},
              {'name': 'dense', 'size': 64},
              {'name': 'activation', 'type': 'relu'},
              {'name': 'dense', 'size': 32},
              {'name': 'activation', 'type': 'relu'},
              {'name': 'dense', 'size': 10},
              {'name': 'activation', 'type': 'softmax'}
              ]
# base save model path
model_save_path = '../model/'
data_save_path = '../data/'
