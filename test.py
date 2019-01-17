# coding=utf8
from PIL import Image
from matplotlib import pyplot
import numpy, cv2
import network

for i in range(20):
    img = cv2.imread('picture/{}.png'.format(i), 0)
    img = cv2.resize(img, (28, 28))
    img = numpy.array(img)
    # img = img.dot((numpy.array([1.0 / 3, 1.0 / 3, 1.0 / 3])).T)
    img = img / 255.0 * 0.99 + 0.01
    img = 1.0 - img
    input_list = img
    pyplot.imshow(input_list, cmap='Greys')  # 显示处理后的测试图片
    pyplot.show()
    input_list = numpy.asfarray(input_list.reshape(784))  # 处理成1*784
    print(i, network.query(input_list))  # 询问答案
    # net.train_net([i % 10] + list((input_list - 0.01) * 255 / 0.9))
