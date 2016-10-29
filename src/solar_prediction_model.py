# -*- coding: utf-8 -*-

import tensorflow as tf
import solar_prediction_reader
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import json
from collections import namedtuple


class SolarPredictionModel:
    def __init__(self, data, target, config, n_model):
        #load the 
        self.n_hidden_solar = config.n_hidden_solar
        self.n_hidden_temp = config.n_hidden_temp
        self.n_hidden_level2 = config.n_hidden_level2
        self.solar_data = data[0]
        self.solar_temp = data[1]
        self.target = target

        self._prediction = None
        self._optimize = None
        self._loss = None

        self.n_model = n_model


    @property
    def prediction(self):
        if self._prediction is None: 

            #build the graph
            #solar rnn lstm
            # with tf.variable_scope("solar_level1"):
            #     cell_solar = tf.nn.rnn_cell.LSTMCell(self.n_hidden_solar, state_is_tuple=True)
            #     outputs_solar, state_solar = tf.nn.dynamic_rnn(cell_solar, self.solar_data, dtype=tf.float32)

            # #temp rnn lstm
            # with tf.variable_scope("temp_level1"):
            #     cell_temp = tf.nn.rnn_cell.LSTMCell(self.n_hidden_temp, state_is_tuple=True)
            #     outputs_temp, state_temp = tf.nn.dynamic_rnn(cell_temp, self.solar_temp, dtype=tf.float32)

            # #concat two feature into a feature
            # data_level2 = tf.concat(1, [outputs_solar, outputs_temp])

            # #2nd level lstm
            # with tf.variable_scope("level2"):
            #     cell_level2 = tf.nn.rnn_cell.LSTMCell(self.n_hidden_level2, state_is_tuple=True)
            #     outputs, state_level2 = tf.nn.dynamic_rnn(cell_level2, data_level2, dtype=tf.float32)

            with tf.variable_scope("solar_test"+str(self.n_model)):
                cell_level2 = tf.nn.rnn_cell.LSTMCell(self.n_hidden_level2, state_is_tuple=True)
                outputs, state_level2 = tf.nn.dynamic_rnn(cell_level2, self.solar_data, dtype=tf.float32)

            #outputs: [batch_size, n_step, n_hidden] -->> [n_step, batch_size, n_hidden]
            #output: [batch_size, n_hidden]
            outputs = tf.transpose(outputs, [1, 0, 2])
            output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

            #regression
            weight1 = tf.Variable(tf.truncated_normal([self.n_hidden_level2, 64]), dtype=tf.float32)
            bias1 = tf.Variable(tf.constant(0.1, shape=[64]), dtype=tf.float32)
            out1 = tf.nn.relu(tf.matmul(output, weight1) + bias1)

            weight2 = tf.Variable(tf.truncated_normal([64, 1]), dtype=tf.float32)
            bias2 = tf.Variable(tf.constant(0.1, shape=[1]), dtype=tf.float32)
            self._prediction = tf.matmul(out1, weight2) + bias2

        return self._prediction


    @property
    def optimize(self):
    	# print "optimize"
        if self._optimize is None:
            optimizer = tf.train.AdamOptimizer(0.0001)
            self._optimize = optimizer.minimize(self.loss)
        return self._optimize


    @property
    def loss(self):
    	# print "loss"
        if self._loss is None:
            self._loss = tf.reduce_mean(tf.square(self.prediction-self.target))
        return self._loss


def figurePlot(y_train, y_test, y_result, index):
    train_len = len(y_train)
    test_len = len(y_test)

    x_train = range(-train_len,0)
    x_test = range(0, test_len)

    plt.figure(index)
    plt.title("Solar Prediction")
    plt.xlabel('Time/hour')
    plt.ylabel('Avg Global CMP22 (vent/cor) [W/m^2]')

    f1 = interpolate.interp1d(x_train+x_test, y_train+y_test, kind='cubic')
    xnew = np.arange(-train_len, test_len-1, 0.01)
    ynew = f1(xnew)
    #plt.plot(x_train+x_test, y_train+y_test, 'o', xnew, ynew, '-', color='blue')
    plt.plot(xnew, ynew, color='blue')

    f2 = interpolate.interp1d(x_test, y_result, kind='cubic')
    xnew = np.arange(0, test_len-1, 0.01)
    ynew = f2(xnew)
    #plt.plot(x_test, y_result, 'o', xnew, ynew, '-', color='red')
    plt.plot(xnew, ynew, color='red')
    plt.show()


def main(_):
    #get the config
    fp = open('config.json')
    config = json.load(fp, object_hook=lambda d:namedtuple('X', d.keys())(*d.values()))
    fp.close()

    n_step = config.n_step
    n_target = config.n_target
    n_input_solar = len(config.input_group_solar)
    n_input_temp = len(config.input_group_temp)
    epoch_size = config.epoch_size
    print_step = config.print_step

    n_model = config.n_model

    x_solar = []
    x_temp = []
    y_ = []
    for i in range(n_model):
        x_solar.append(tf.placeholder(tf.float32, [None, n_step, n_input_solar]))
        x_temp.append(tf.placeholder(tf.float32, [None, n_step, n_input_temp]))
        y_.append(tf.placeholder(tf.float32, [None, 1]))

    reader = solar_prediction_reader.Reader(config.data_path, config)
    models = []
    for i in range(n_model):
        models.append(SolarPredictionModel([x_solar[i], x_temp[i]], y_[i], config, i))

    predictions = []
    losses = []
    optimizes = []
    for i in range(n_model):
        predictions.append(models[i].prediction)
        losses.append(models[i].loss)
        optimizes.append(models[i].optimize)

    #new a saver to save the model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        save_path = saver.restore(sess, config.model_path)
        
        #train
        for i in range(epoch_size+1):
            batch = reader.next_batch()
            for j in range(n_model):
                if i%print_step == 0:
                    l = sess.run(losses[j], feed_dict={x_solar[j]:batch[0], x_temp[j]:batch[1], y_[j]:batch[2][j]})
                    print "model", j, "training loss:", l

                sess.run(optimizes[j], feed_dict={x_solar[j]:batch[0], x_temp[j]:batch[1], y_[j]:batch[2][j]})


            # test
            if i%config.test_step == 0:
                test_results = []
                solar_test_input, temp_test_input, test_targets = reader.get_test_set(5)
                for k in range(n_model):
                    test_result = sess.run(predictions[k], feed_dict={x_solar[k]:solar_test_input, x_temp[k]:temp_test_input})
                    test_results.append(test_result)
                

<<<<<<< HEAD
=======
                #first test result
>>>>>>> 67745d36d816e116247b39c3eebcf0bcd429b1c7
                test_target_all = []
                test_result_all = []

                for i in range(5):
                    test_target_all = test_target_all + test_targets[i]
                    test_result_all = test_result_all + list(zip(*test_results)[i])

                #reshape the list by zip all value of the same target dimension into a tuple
                #[test_size, n_target]
                #[n_target, test_size]
                test_target_all = zip(*test_target_all)
                test_result_all = zip(*test_result_all)

                #get the target real data from the reader
                #use to plot in the figure
                target_before_test_all = reader.get_target_before_test(120)
                target_before_test_all = zip(*target_before_test_all)

                #plot each target dimension result in a figure
                #now the target is one  dim, so there is only one figure
                for i in range(n_target):
                    figurePlot(list(target_before_test_all[i]), list(test_target_all[i]), list(test_result_all[i]), i)

                    
        

        save_path = saver.save(sess, config.model_path)

if __name__ == "__main__":
    tf.app.run()

