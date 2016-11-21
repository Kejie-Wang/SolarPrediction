# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '12/10/2016'

import tensorflow as tf

class Model:
    def __init__(self, data, target, keep_prob, config):
        #load the config
        #the input data
        self.solar_data = data[0]
        self.solar_temp = data[1]
        self.target = target
        self.keep_prob = keep_prob

        #the network parameters
        self.n_hidden_solar = config.n_hidden_solar
        self.n_hidden_temp = config.n_hidden_temp
        self.n_hidden_level2 = config.n_hidden_level2
        self.n_fully_connect_hidden = config.n_fully_connect_hidden
        self.n_target = config.n_target

        #train param
        self.lr = config.lr

        #loss param
        self.epsilon = config.epsilon
        self.C = config.C

        self._prediction = None
        self._optimize = None
        self._loss = None

    @property
    def prediction(self):
        if self._prediction is None: 

            # build the graph
            # solar rnn lstm
            with tf.variable_scope("solar_level1"):
                cell_solar = tf.nn.rnn_cell.LSTMCell(self.n_hidden_solar, state_is_tuple=True)
                outputs_solar, state_solar = tf.nn.dynamic_rnn(cell_solar, self.solar_data, dtype=tf.float32)

            # temp rnn lstm
            with tf.variable_scope("temp_level1"):
                cell_temp = tf.nn.rnn_cell.LSTMCell(self.n_hidden_temp, state_is_tuple=True)
                outputs_temp, state_temp = tf.nn.dynamic_rnn(cell_temp, self.solar_temp, dtype=tf.float32)

            # concat two features into a feature
            data_level2 = tf.concat(1, [outputs_solar, outputs_temp])

            #2nd level lstm
            with tf.variable_scope("level2"):
                cell_level2 = tf.nn.rnn_cell.LSTMCell(self.n_hidden_level2, state_is_tuple=True)
                outputs, state_level2 = tf.nn.dynamic_rnn(cell_level2, data_level2, dtype=tf.float32)

            #outputs: [batch_size, n_step, n_hidden] -->> [n_step, batch_size, n_hidden]
            #output: [batch_size, n_hidden]
            outputs = tf.transpose(outputs, [1, 0, 2])
            output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

            # regression
            w_fc = tf.Variable(tf.truncated_normal([self.n_hidden_level2, self.n_fully_connect_hidden]), dtype=tf.float32)
            b_fc = tf.Variable(tf.constant(0.1, shape=[self.n_fully_connect_hidden]), dtype=tf.float32)
            h_fc = tf.nn.relu(tf.matmul(output, w_fc) + b_fc)

            # h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

            self.weight = tf.Variable(tf.truncated_normal([self.n_fully_connect_hidden, self.n_target]), dtype=tf.float32)
            bias = tf.Variable(tf.constant(0.1, shape=[self.n_target]), dtype=tf.float32)

            self._prediction = tf.matmul(h_fc, self.weight) + bias

        return self._prediction

    @property
    def loss(self):
        if self._loss is None:
            m = tf.matmul(tf.transpose(self.weight,[1,0]), self.weight)
            diag = tf.matrix_diag_part(m)
            w_sqrt_sum = tf.reduce_sum(tf.sqrt(diag))

            diff = self.prediction - self.target
            err = tf.reduce_sum(tf.square(diff), reduction_indices=1) - self.epsilon
            err_greater_than_espilon = tf.cast(err > 0, tf.float32)
            total_err = tf.reduce_sum(tf.mul(err, err_greater_than_espilon))

            self._loss = 0.5 * w_sqrt_sum + self.C * total_err
        return self._loss


    @property
    def optimize(self):
        if self._optimize is None:
            optimizer = tf.train.AdamOptimizer(self.lr)
            self._optimize = optimizer.minimize(self.loss)
        return self._optimize

