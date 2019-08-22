#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
from xzc_tools import tools
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

@tools.funcRunTime
def load_data(file_path):
    try:
        feature_data = []
        label_data = []
        with open(file_path) as f:
            for line in f:
                feature_temp = [1,]
                line = [float(i) for i in line.strip().split('\t')]
                feature_temp.extend(line[0:2])
                label_data.append(int(line[-1]))
                feature_data.append(feature_temp)
        return np.mat(feature_data), np.mat(label_data).T, len(set(label_data))
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

@tools.funcRunTime
def tensorflow_train(feature, label, maxCycle, alpha):
    try:
        # 定义神经网络的参数
        w = tf.Variable(tf.random_normal([3, 4]))

        # 存放训练数据的位置
        x = tf.placeholder(tf.float32, shape=(None, 3), name='x-input')
        y_real = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

        # 定义神经网络的传播过程
        y_predict = tf.matmul(x, w)
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict, labels=[int(i[0]) for i in label.tolist()]))
        train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

        # 创建会话
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            for i in range(maxCycle):
                sess.run(train_step, feed_dict={x:feature, y_real:label})
                if i % 200 == 0:
                    total_cross_entropy = sess.run(cross_entropy, feed_dict={x: feature, y_real: label})
                    print(i, total_cross_entropy)
            return sess.run(w)
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

@tools.funcRunTime
def show_model(feature, w, label):
    try:
        w_size = np.shape(w)
        feature_x = [float(i[0]) for i in feature[:, 1]]
        plt.title('logistic regression')
        plt.xlabel('x')
        plt.ylabel('y')

        point_type = ['o', '*', '+', '.']

        for i in range(w_size[1]):
            w_temp = w[:,i]
            w_temp = w_temp.tolist()
            print(w_temp)
            point_x = []
            point_y = []
            for j in range(len(label)):
                if label[j] == i:
                    point_x.append(float(feature[:,1][j]))
                    point_y.append(float(feature[:,2][j]))
            feature_y = [-(w_temp[0] + w_temp[1] * k) / w_temp[2] for k in feature_x]
            plt.plot(point_x, point_y, point_type[i])
            plt.plot(feature_x, feature_y, 'r')
        plt.show()

    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

@tools.funcRunTime
def save_model(w):
    try:
        if os.path.exists('w_info.txt'):
            os.remove('w_info.txt')
        for i in range(np.shape(w)[1]):
            w_temp = w[:,i]
            w_temp = w_temp.tolist()
            w_temp = [str(w_temp[0]), str(w_temp[1]), str(w_temp[2])]
            tools.writeFile(1, 'w_info.txt', '\t'.join(w_temp))
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

if __name__ == '__main__':
    # 导入训练数据
    tools.printInfo(1,'导入训练数据')
    data_path = os.path.abspath(sys.argv[1])
    feature, label, label_num = load_data(data_path)

    # 训练数据
    maxCycle = int(input('请输入最大循环次数(比0大):\n'))
    alpha = float(input('请输入学习率(0, 1):\n'))
    if maxCycle <= 0:
        tools.printInfo(3, '最大循环次数数值错误，请重新运行并输入正确的值!')
        sys.exit()
    if alpha <= 0 or alpha >= 1:
        tools.printInfo(3, '学习率数值错误，请重新运行并输入正确的值!')
        sys.exit()
    tools.printInfo(1, '最大循环次数和学习率符合范围，开始训练数据:')
    w = tensorflow_train(feature, label, maxCycle, alpha)

    save_model(w)

    show_model(feature, w, label)