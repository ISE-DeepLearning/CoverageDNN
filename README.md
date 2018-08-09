# CoverageDNN
explore the new criteria of coverage in DNN


## 1 数据准备工作 data preparation
首先将mnist数据集(tensorflow版)和cifar数据集(python版) 准备到data文件夹中
(没有找我要,or 百度网盘链接: (链接)[https://pan.baidu.com])
结构如下:
|CoverageDNN
|
|---data
|
|--------cifar_data
|
|------------------cifar-10
|---------------------------data_batch_1
|---------------------------data_batch_2
|---------------------------data_batch_3
|---------------------------data_batch_4
|---------------------------data_batch_5
|---------------------------test_batch
|
|------------------cifar-100
|---------------------------test
|---------------------------train
|
|--------MNIST_data
|------------------t10k-images-idx3-ubyte.gz
|------------------t10k-labels-idx1-ubyte.gz
|------------------train-images-idx3-ubyte.gz
|------------------train-labels-idx1-ubyte.gz
