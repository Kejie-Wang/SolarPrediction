# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '12/10/2016'

import tensorflow as tf

class Model:
    def __init__(self, data, target, keep_prob, config):
        #load the config
        #the input data
        self.solar_data = data[0]
        self.temp_data = data[1]
        self.target = target
        self.keep_prob = keep_prob

        #the network parameters
        self.n_hidden_solar = config.n_hidden_solar
        self.n_hidden_temp = config.n_hidden_temp
        self.n_hidden_level2 = config.n_hidden_level2
        self.n_fully_connect_hidden = config.n_fully_connect_hidden
        self.n_target = config.n_target

        #train params
        self.lr = config.lr

        #loss params (svr params)
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
                outputs_temp, state_temp = tf.nn.dynamic_rnn(cell_temp, self.temp_data, dtype=tf.float32)

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
            w_fc1 = tf.Variable(tf.truncated_normal([self.n_hidden_level2, self.n_fully_connect_hidden]), dtype=tf.float32)
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[self.n_fully_connect_hidden]), dtype=tf.float32)
            h_fc1 = tf.nn.relu(tf.matmul(output, w_fc1) + b_fc1)

            # h_fc_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            #multi-support vector regresiion
            self.weight = tf.Variable(tf.truncated_normal([self.n_fully_connect_hidden, self.n_target]), dtype=tf.float32)
            bias = tf.Variable(tf.constant(0.1, shape=[self.n_target]), dtype=tf.float32)

            self._prediction = tf.matmul(h_fc1, self.weight) + bias

        return self._prediction

    @property
    def loss(self):
        if self._loss is None:
            #compute the ||w||2
            #use the w^T * W to compute and the sum the diag to get the result
            m = tf.matmul(tf.transpose(self.weight,[1,0]), self.weight)
            diag = tf.matrix_diag_part(m)
            w_sqrt_sum = tf.reduce_sum(diag)

            #the loss of the trian set
            diff = self.prediction - self.target
            err = tf.sqrt(tf.reduce_sum(tf.square(diff), reduction_indices=1)) - self.epsilon
            err_greater_than_espilon = tf.cast(err > 0, tf.float32)
            total_err = tf.reduce_sum(tf.mul(tf.square(err), err_greater_than_espilon))

            # self._loss = 0.5 * w_sqrt_sum + self.C * total_err
            self._loss = total_err
        return self._loss


    @property
    def optimize(self):
        if self._optimize is None:
            optimizer = tf.train.AdamOptimizer(self.lr)
            self._optimize = optimizer.minimize(self.loss)
        return self._optimize
