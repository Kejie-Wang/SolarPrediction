# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '12/10/2016'

import tensorflow as tf

class Model:
    """
    This is an multi-modal (three modality) neutral network model used as point/probabilistic forecast
        model digram:
             ---------
        ----|  lstm   |--------- |
             ---------           |
                                 |
             ---------           |           ----------------           ------------
        ----|  lstm   |--------- |----------| lstm(optional) | --------| regression |
             ---------           |           ----------------           ------------
                                 |
             -----    ------     |
        ----| CNN |--| lstm |----|
             -----    ------
        The model focuses on the multi-modality and this model contain three modalities each of which is lstm, lstm and CNN.
        e.g. This model used to predict solar irradiance and the first and second modalities are the irradiance and meteorological data
            and the third modality is an image dataset and use an CNN to extract the feature. And then concatenating all features into a
            feature as the input of the lstm of the second level. Then we use the output of the last cell as the regressor input to predict
            the value.
        regressor: there are lots of regressors can be used for different purpose. (specific in the config in )
            e.g. linear regression, a fully connected NN with linear regression, support vector regression,
                multi-support vector regression (considering the time dependency)
                quantile regression (used as probabilistic regression)
    """
    def __init__(self, data, target, keep_prob, config):
        """
        @brief The constructor of the model
        @param data: the input the data of the model (features) data[0], data[1], ... for multi-modality
               target: the groundtruth of the model
               config: the configuration of the model and it may contains following values:
                    lr: learning rate
                    regressor: the regressor type chosen from {"lin", "msvr", "prob"}
                    epsilon, C: params for epsilon-insensitive multi-support regression
                    quantile: param for the quantile regression
        """
        #load the config
        #the input data
        self.data = data
        self.target = target
        self.keep_prob = keep_prob

        #the network parameters
        self.n_first_hidden = config.n_first_hidden
        self.n_second_hidden = config.n_second_hidden
        self.n_third_hidden = config.n_third_hidden
        self.n_hidden_level2 = config.n_hidden_level2
        self.n_target = config.n_target

        #modality configuration
        self.modality = config.modality

        #regressor type {'lin', 'msvr', 'prob'}
        self.regressor = config.regressor

        #train params
        self.lr = config.lr

        #loss params (svr params)
        if self.regressor == "msvr":
            self.epsilon = config.epsilon
            self.C = config.C

        #quantile regression params
        if self.regressor == "quantile":
            self.quantile_rate = config.quantile_rate

        self._prediction = None
        self._optimize = None
        self._loss = None


        self.n_step_1 = config.n_step_1
        self.n_step_2 = config.n_step_2
        self.n_step_3 = config.n_step_3

        self.input_width = config.width
        self.input_height = config.height
        self.cnn_feat_size = config.cnn_feat_size

        self.prediction
        self.optimize
        self.loss

    def weight_varible(self, shape):
        # initial = tf.truncated_normal(shape, stddev=0.1)
        weights = tf.get_variable('weights', shape, initializer=tf.random_normal_initializer(0, stddev=0.1))
        return weights

    def bias_variable(self, shape):
        # initial = tf.constant(0.1, shape=shape)
        bias = tf.get_variable('bias', shape, initializer=tf.constant_initializer(0.1))
        return bias

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def cnn_model(self, x):
        x_image = tf.reshape(x, [-1, self.input_height, self.input_width, 1])
        with tf.variable_scope('conv_1') as scope:
            W_conv1 = self.weight_varible([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])

            # conv layer-1
            # x = tf.placeholder(tf.float32, [None, 784])
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)

        with tf.variable_scope('conv_2') as scope:
            # conv layer-2
            W_conv2 = self.weight_varible([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])

            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self.max_pool_2x2(h_conv2)


        with tf.variable_scope('fc_1') as scope:
            # full connection
            s = h_pool2.get_shape().as_list()
            # print 'X1,X2,X3,X4', x1,x2,x3, x4
            W_fc1 = self.weight_varible([s[1]*s[2]*s[3], self.cnn_feat_size])
            b_fc1 = self.bias_variable([self.cnn_feat_size])

            h_pool2_flat = tf.reshape(h_pool2, [-1, s[1]*s[2]*s[3]])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            return h_fc1;

    def _get_last_out(self, outputs):
        #outputs: [batch_size, n_step, n_hidden] -->> [n_step, batch_size, n_hidden]
        #output: [batch_size, n_hidden]
        #get the last output of the lstm as the input of the regressor
        outputs = tf.transpose(outputs, [1, 0, 2])
        output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

        return output

    @property
    def prediction(self):
        """
        Build the graph of the model and return the prediction value of the model
        NOTE: You can easily treat it as an member of this class (model.prediction to refer)
        """
        if self._prediction is None:

            # build the graph
            output_first_level = []
            output_first_level_size = 0

            # The first modality
            # irradiance rnn lstm
            if self.modality[0] == 1:
                print "The irradiance modality is used"
                output_first_level_size += self.n_first_hidden
                with tf.variable_scope("irradiance"):
                    cell_1 = tf.nn.rnn_cell.LSTMCell(self.n_first_hidden, state_is_tuple=True)
                    cell_1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell_1, output_keep_prob=self.keep_prob)
                    cell_1 = tf.nn.rnn_cell.MultiRNNCell(cells=[cell_1] * 2, state_is_tuple=True)
                    outputs_1, state_1 = tf.nn.dynamic_rnn(cell_1, self.data[0], dtype=tf.float32)
                    output_first_level.append(self._get_last_out(outputs_1))

            # The second modality
            # meteorological rnn lstm
            if self.modality[1] == 1:
                print "The meteorological modality is used"
                output_first_level_size += self.n_second_hidden
                with tf.variable_scope("meteorological"):
                    cell_2 = tf.nn.rnn_cell.LSTMCell(self.n_second_hidden, state_is_tuple=True)
                    cell_2 = tf.nn.rnn_cell.DropoutWrapper(cell=cell_2, output_keep_prob=self.keep_prob)
                    cell_2 = tf.nn.rnn_cell.MultiRNNCell(cells=[cell_2] * 2, state_is_tuple=True)
                    outputs_2, state_2 = tf.nn.dynamic_rnn(cell_2, self.data[1], dtype=tf.float32)
                    output_first_level.append(self._get_last_out(outputs_2))

            # The third modality
            if self.modality[2] == 1:
                print "The sky camera image modality is used"
                output_first_level_size += self.n_third_hidden
                cnn_out = None
                with tf.variable_scope('sky_cam_image_cnn') as scope:
                    for i in range(self.n_step_3):
                        tmp_out = self.cnn_model(self.data[2][:, i, :, :])
                        tmp_out = tf.reshape(tmp_out, [-1, 1, self.cnn_feat_size])
                        if cnn_out is None:
                            cnn_out = tmp_out
                        else:
                            cnn_out = tf.concat(1, [cnn_out,tmp_out])
                        scope.reuse_variables()

                #image rnn lstm
                with tf.variable_scope("sky_cam_image_lstm"):
                    cell_3 = tf.nn.rnn_cell.LSTMCell(self.n_third_hidden, state_is_tuple=True)
                    outputs_3, state3 = tf.nn.dynamic_rnn(cell_3, cnn_out, dtype=tf.float32)
                    output_first_level.append(self._get_last_out(outputs_3))

            # concat two features into a feature
            output = tf.concat(1, output_first_level)

            # 2nd level lstm
            # optional
            # with tf.variable_scope("second_level"):
            #     output_first_level = []
            #     if self.modality[0] == 1:
            #         output_first_level.append(outputs_1)
            #     if self.modality[1] == 1:
            #         output_first_level.append(outputs_2)
            #     if self.modality[2] == 1:
            #         output_first_level.append(outputs_3)
            #     output_first_level = tf.concat(2, output_first_level)
            #     cell_level2 = tf.nn.rnn_cell.LSTMCell(self.n_hidden_level2, state_is_tuple=True)
            #     outputs, state_level2 = tf.nn.dynamic_rnn(cell_level2, output_first_level, dtype=tf.float32)
            #     output = self._get_last_out(outputs)
            #     output_first_level_size = self.n_hidden_level2

            # regression
            with tf.variable_scope("regression"):
                weight = tf.Variable(tf.truncated_normal(shape=[output_first_level_size, self.n_target], stddev=2.0), dtype=tf.float32)
                bias = tf.Variable(tf.constant(0.0, shape=[self.n_target]), dtype=tf.float32)
                self._prediction = tf.matmul(output, weight) + bias

                self.w_sum = tf.reduce_sum(tf.square(weight))
                self.b_sum = tf.reduce_sum(tf.square(bias))
                self.weight = weight
                self.bias = bias

        return self._prediction

    @property
    def loss(self):
        """
        Define the loss of the model and you can modify this section by using different regressor
        """
        if self._loss is None:
            if self.regressor == "lin": #only work on the target is one-dim
                self._loss = tf.reduce_mean(tf.square(self.prediction - self.target)) + (self.w_sum + self.b_sum) * self.C
            elif self.regressor == "msvr":
                #the loss of the train set
                diff = self.prediction - self.target
                err = tf.sqrt(tf.reduce_sum(tf.square(diff), reduction_indices=1)) - self.epsilon
                err_greater_than_espilon = tf.cast(err > 0, tf.float32)
                total_err = tf.reduce_mean(tf.mul(tf.square(err), err_greater_than_espilon))

                self._loss = self.C * self.w_sum + total_err
            elif self.regressor == "quantile":
                diff = self.prediction - self.target
                coeff = tf.cast(diff>0, tf.float32) - self.quantile_rate
                self._loss = tf.reduce_sum(tf.mul(diff, coeff)) + (self.w_sum + self.b_sum) * self.C

        return self._loss


    @property
    def optimize(self):
        """
        Define the optimizer of the model used to train the model
        """
        if self._optimize is None:
            optimizer = tf.train.AdamOptimizer(self.lr)
            self._optimize = optimizer.minimize(self.loss)
        return self._optimize

    @property
    def mae(self):
        return tf.reduce_mean(tf.abs(self.prediction - self.target))

    @property
    def rmse(self):
        return tf.sqrt(tf.reduce_mean(tf.square(self.prediction - self.target)))

    @property
    def coverage_rate(self):
        return tf.reduce_mean(tf.cast(self.prediction - self.target > 0, tf.float32))
