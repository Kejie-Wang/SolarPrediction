import numpy as np
import matplotlib.pyplot as pyt
import threading
import time

HOUR_IN_A_DAY = 24

sim_day_fea_train_data_path = './train/sim_day_fea_train_data.csv'
sim_day_fea_validation_data_path = './validation/sim_day_fea_validation_data.csv'
sim_day_fea_test_data_path = './test/sim_day_fea_test_data.csv'

hist_fea_data_path = './train/sim_day_fea_train_data.csv'
hist_ir_data_path = './train/target_train_data.csv'

def _DTW_distance(s1, s2, window_size):
    window_size = int(max(window_size, abs(len(s1)-len(s2))))
    Gamma={}
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            Gamma[(i, j)] = float('inf')
    Gamma[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i-window_size), min(len(s2), i+window_size+1)):
            dist= (s1[i]-s2[j])**2
            Gamma[(i, j)] = dist + min(Gamma[(i-1, j)],Gamma[(i, j-1)], Gamma[(i-1, j-1)])
    return (Gamma[len(s1)-1, len(s2)-1]) ** 0.5

def _sim_day_fea_comp(hist_fea_data, hist_ir_data, input_fea):
    '''
    hist_fea_data: [hist_day_num, HOUR_IN_A_DAY]
    hist_ir_data: [hist_day_num, HOUR_IN_A_DAY]
    input_data: [input_day_num, HOUR_IN_A_DAY]
    '''
    hist_day_num = hist_fea_data.shape[0]
    input_day_num = input_fea.shape[0]

    dist = np.zeros(shape=[input_day_num, hist_day_num])
    start = time.time()
    for i in range(input_day_num):
        print i, '/', input_day_num
        for j in range(0, hist_day_num):
            dist[i][j] = _DTW_distance(input_fea[i], hist_fea_data[j], 1)

    index = np.argsort(dist, axis=1)
    return index

sim_day_fea_train_data = np.loadtxt(sim_day_fea_train_data_path, delimiter=',')
sim_day_fea_validation_data = np.loadtxt(sim_day_fea_validation_data_path, delimiter=',')
sim_day_fea_test_data = np.loadtxt(sim_day_fea_test_data_path, delimiter=',')
hist_fea_data = np.loadtxt(hist_fea_data_path, delimiter=',')
hist_ir_data = np.loadtxt(hist_ir_data_path, delimiter=',')

sim_day_fea_train_data = np.reshape(sim_day_fea_train_data, [-1, HOUR_IN_A_DAY])
sim_day_fea_validation_data = np.reshape(sim_day_fea_validation_data, [-1, HOUR_IN_A_DAY])
sim_day_fea_test_data = np.reshape(sim_day_fea_test_data, [-1, HOUR_IN_A_DAY])
hist_fea_data = np.reshape(hist_fea_data, [-1, HOUR_IN_A_DAY])
hist_ir_data = np.reshape(hist_ir_data, [-1, HOUR_IN_A_DAY])


sim_day_validation_ir = _sim_day_fea_comp(hist_fea_data, hist_ir_data, sim_day_fea_validation_data)
np.savetxt('./validation/similar_day_validation_data_1.csv', sim_day_validation_ir, delimiter=',', fmt='%d')

sim_day_test_ir = _sim_day_fea_comp(hist_fea_data, hist_ir_data, sim_day_fea_test_data)
np.savetxt('./test/similar_day_test_data_1.csv', sim_day_test_ir, delimiter=',', fmt='%d')

sim_day_train_ir = _sim_day_fea_comp(hist_fea_data, hist_ir_data, sim_day_fea_train_data)
np.savetxt('./train/similar_day_train_data_1.csv', sim_day_train_ir, delimiter=',', fmt='%d')
