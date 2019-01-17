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
