#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
from xzc_tools import tools
import numpy as np
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
def train_data(feature, label, label_num, maxCycle, alpha):
    try:
        data_num, feature_num = np.shape(feature)
        w = np.mat(np.ones((feature_num, label_num)))
        i = 0
        while i < maxCycle:
            i += 1
            err = np.exp(feature * w)
            if i % 500 == 0:
                tools.printInfo(1, '训练次数:{0},损失函数值:{1}'.format(str(i),str(cost(err, label))))
            rowSum = -err.sum(axis=1)
            rowSum = rowSum.repeat(label_num, axis=1)
            err = err / rowSum
            for x in range(data_num):
                err[x, label[x, 0]] += 1
            w = w + (alpha / data_num) * feature.T * err

        return w
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

@tools.funcRunTime
def save_model(w):
    try:
        if os.path.exists('w_info.txt'):
            os.remove('w_info.txt')
        w = w.tolist()
        for i in w:
            i = [str(j) for j in i]
            tools.writeFile(1, 'w_info.txt', '\t'.join(i))
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()

def cost(err, label):
    try:
        m = np.shape(err)[0]
        sum_cost = 0.0
        for i in range(m):
            if err[i, label[i, 0]] / np.sum(err[i,:]) > 0:
                sum_cost -= np.log(err[i, label[i, 0]] / np.sum(err[i,:]))
            else:
                sum_cost -= 0
        return sum_cost / m
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
    w = train_data(feature, label, label_num, maxCycle, alpha)

    # 保存模型
    save_model(w)