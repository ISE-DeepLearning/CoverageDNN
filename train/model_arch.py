# -*- coding:UTF-8 -*-
import config
import os

class ModelArch:
    dataset_name = 'mnist'
    input_size = (28 * 28,)
    model_arch = []
    save_path = os.path.join(config.model_save_path,"model.hdf5")

    def __init__(self, dataset_name='mnist', input_size=(28 * 28,), model_arch=[],model_save_path='model.hdf5'):
        self.dataset_name = dataset_name
        self.input_size = input_size
        self.model_arch = model_arch
        self.save_path = model_save_path

    # 添加层
    def add_layer(self, layer_dict):
        self.model_arch.append(layer_dict)

    def pop_layer(self):
        self.model_arch.pop()

    def change_save_path(self,model_save_path):
        self.save_path = model_save_path

