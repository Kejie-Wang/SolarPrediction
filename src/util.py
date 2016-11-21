# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '21/11/2016'

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def MSE_And_MAE(targets, results):
    diff = results - targets
    mse = np.sum(diff * diff) / diff.size
    mae = np.sum(np.abs(diff)) / diff.size

    return mse, mae


# def figurePlot(y_train, y_test, y_result, index):
#     train_len = len(y_train)
#     test_len = len(y_test)

#     x_train = range(-train_len,0)
#     x_test = range(0, test_len)

#     plt.figure(index)
#     plt.title("Solar Irradiance Prediction with Deep Learning Model",fontsize=20)
#     plt.xlabel('Day',fontsize=15)
#     plt.ylabel('Avg Global CMP22 (vent/cor) [W/m^2]',fontsize=15)

#     f1 = interpolate.interp1d(x_train+x_test, y_train+y_test, kind='cubic')
#     xnew = np.arange(-train_len, test_len-1, 0.01)
#     ynew = f1(xnew)
#     #plt.plot(x_train+x_test, y_train+y_test, 'o', xnew, ynew, '-', color='blue')
#     plt.plot(xnew, ynew, color='blue')

#     f2 = interpolate.interp1d(x_test, y_result, kind='cubic')
#     xnew = np.arange(0, test_len-1, 0.01)
#     ynew = f2(xnew)
#     #plt.plot(x_test, y_result, 'o', xnew, ynew, '-', color='red')
#     plt.plot(xnew, ynew, color='red')
#     plt.show()