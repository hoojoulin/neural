# coding=utf8
import time
from matplotlib import pyplot
import numpy
import scipy.special


class neuralnetwork:
    # 初始化神经网络,输入层，隐含层，输出层，学习速率
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.wih = numpy.random.rand(hidden_nodes, input_nodes) - 0.5
        self.who = numpy.random.rand(output_nodes, hidden_nodes) - 0.5
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, input_list, target_list):  # 训练，输入，目标输出
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        targets = numpy.array(target_list, ndmin=2).T
        output_error = targets - final_outputs
        hidden_error = numpy.dot(self.who.T, output_error)
        self.who += self.learning_rate * numpy.dot((output_error * final_outputs) * (1.0 - final_outputs),
                                                   numpy.transpose((hidden_outputs)))
        self.wih += self.learning_rate * numpy.dot((hidden_error * hidden_outputs) * (1.0 - hidden_outputs),
                                                   numpy.transpose(inputs))
        pass

    def query(self, input_list):  # 输入得到答案
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass


#
# if __name__ == '__main__':
#     t0 = time.time()
#     # train
#     for i in range(30):
#         training_file = csv_file = open('conf/mnist_train.csv', 'r')
#         training_data_list = training_file.readlines()
#         training_file.close()
#         for data in training_data_list:
#             data = data.split(',')
#             train_net(data)
#         save_net()
#
#         training_file = csv_file = open('conf/mnist_train_100.csv', 'r')
#         training_data_list = training_file.readlines()
#         training_file.close()
#         for data in training_data_list:
#             data = data.split(',')
#             train_net(data)
#
#         training_file = csv_file = open('conf/mnist_test.csv', 'r')
#         training_data_list = training_file.readlines()
#         training_file.close()
#         for data in training_data_list:
#             data = data.split(',')
#             train_net(data)
#         save_net()
#     t1 = time.time()
#     print(t1 - t0)
#
# network_1 = neuralnetwork(784, 300, 10, 0.1)
# set_net(network_1)
# #     # test
# #
# test_file = open('conf/mnist_test_10.csv', 'r')
# test_data_list = test_file.readlines()
# test_file.close()
# cnt = 0
# total = 0
# wrong = []
# for data in test_data_list:
#     total += 1
#     data0 = data.split(',')
#     input_list = numpy.array(map(float, data0[1:]))
#     input_list = input_list / 255.0 * 0.99 + 0.01
#     output_list = list(network_1.query(input_list))
#     if int(data0[0]) != output_list.index(max(output_list)):
#         wrong.append([int(data0[0]), output_list.index(max(output_list))])
#         cnt += 1
#     pyplot.imshow(numpy.asfarray(input_list.reshape(28, 28)), cmap='Greys')
#     pyplot.show()
#     train_net(network_1, data0)
# save_net(network_1)
# for i in wrong:
#     print(i)
# print(cnt, total, 1.0 - cnt * 1.0 / total)
# t2 = time.time()
# print(t2 - t0)
