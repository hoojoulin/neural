# coding=utf8
from neuralnetwork import neuralnetwork
from matplotlib import pyplot
import numpy


# 创建一个神经网络，输入层784个节点，隐含层300个节点，输出层十个节点，训练速率0.2
# network_1 = neuralnetwork(784, 300, 10, 0.2)


# 保存网络参数到配置文件
def save_net(network_1):
    wih_csv = open('conf/wih.csv', 'w')
    for i in network_1.wih:
        for j in i:
            wih_csv.write(str(j) + ' ')
        wih_csv.write('\n')
    wih_csv.close()

    who_csv = open('conf/who.csv', 'w')
    for i in network_1.who:
        for j in i:
            who_csv.write(str(j) + ' ')
        who_csv.write('\n')
    who_csv.close()


# 训练网络
def train_net(data):
    network_1 = neuralnetwork(784, 300, 10, 0.3)
    set_net(network_1)
    ans = int(data[0])  # 正确答案
    input_list = numpy.array(map(float, data[1:]))  # 输入数据
    input_list = input_list / 255.0 * 0.99 + 0.01
    target = numpy.zeros(10) + 0.01
    target[ans] = 0.99
    network_1.train(input_list, target)  # 训练
    save_net(network_1)
    pass


# 将配置文件参数设置到网络
def set_net(network_1):
    wih_csv = open('conf/wih.csv', 'r')
    wih_data = wih_csv.readlines()
    wih_csv.close()
    r = 0
    for w in wih_data:
        w = w.split()
        w = map(float, w)
        network_1.wih[r] = numpy.array(w)
        r += 1

    who_csv = open('conf/who.csv', 'r')
    who_data = who_csv.readlines()
    who_csv.close()
    r = 0
    for w in who_data:
        w = w.split()
        w = map(float, w)
        network_1.who[r] = numpy.array(w)
        r += 1
    pass


# 询问答案
def query(input_list):
    network_1 = neuralnetwork(784, 300, 10, 0.1)
    set_net(network_1)
    network_1.query(input_list)
    output_list = list(network_1.query(input_list))
    return output_list.index(max(output_list))
