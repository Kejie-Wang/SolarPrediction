# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '12/10/2016'

import tensorflow as tf
import pywt

class Model:
    """
    This is an multi-modal (three modality) neutral network model used as point/probabilistic forecast
        model digram:
                            ---------
        irradiance     ----|  lstm   |--------- |
                            ---------           |
                                                |
                            ---------           |           ----------------            -----------
        meteorological ----|  lstm   |--------- |----------| lstm(optional) | --------| regression |
                            ---------           |           ----------------      |     -----------
                                                |                                 |
                            -----    ------     |                                 |
        sky image      ----| CNN |--| lstm |----|                                 |
                            -----    ------                                       |
        hour index     -----------------------------------------------------------

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
    def __init__(self, data, target, hour_index, keep_prob, config):
        """
        @brief The constructor of the model
        @param data: the input the data of the model (features) data[0], data[1], ... for multi-modality
               target: the groundtruth of the model
               config: the configuration of the model and it may contains following values:
                    lr: learning rate
                    regressor: the regressor type chosen from {"mse", "msvr", "prob"}
                    epsilon, C: params for epsilon-insensitive multi-support regression
                    quantile: param for the quantile regression
        """
        #load the config
        #the input data
        self.data = data
        self.target = target
        self.hour_index = hour_index
        self.keep_prob = keep_prob

        #the network parameters
        self.n_first_hidden = config.n_first_hidden
        self.n_second_hidden = config.n_second_hidden
        self.n_third_hidden = config.n_third_hidden
        self.n_hidden_level2 = config.n_hidden_level2
        self.n_target = config.n_target

        #modality configuration
        self.modality = config.modality

        #regressor type {'mse', 'msvr', 'prob'}
        self.regressor = config.regressor

        #train params
        self.lr = config.lr

        self.C = config.C

        #loss params (svr params)
        if self.regressor == "msvr":
            self.epsilon = config.epsilon

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

    def _get_last_out(self, outputs):
        #outputs: [batch_size, n_step, n_hidden] -->> [n_step, batch_size, n_hidden]
        #output: [batch_size, n_hidden]
        #get the last output of the lstm as the input of the regressor
        outputs = tf.transpose(outputs, [1, 0, 2])
        output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

        return output

    def _Gaussian_Kernel(self, x, theta):
        return tf.exp((-0.5 * tf.square(x / theta))) / theta / 2.0

    @property
    def prediction(self):
        """
        Build the graph of the model and return the prediction value of the model
        NOTE: You can easily treat it as an member of this class (model.prediction to refer)
        """
        if self._prediction is None:

            # build the graph
            outputs = []
            outputs_size = 0

            if self.modality[0] == 1:
                print "The irradiance modality is used"
                coeffs = pywt.wavedec(data[0], wavelet=self.wavelet, level=self.level)
                for l in range(level+1):
                    with tf.variable_scope('irradiance_level_' + str(l)):
                        cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden_size, state_is_tuple=True)
                        output, state = tf.nn.dynamic_rnn(cell, coeffs[i], dtype=tf.float32)
                        outputs.append(self._get_last_out(output))
                        outputs_size += self.n_hidden_size

            if self.modality[1] == 1:
                coeffs = pywt.wavedec(data[1], wavelet=self.wavelet, level=self.level)
                for l in range(level+1):
                    with tf.variable_scope('meteorological_level_' + str(l)):
                        cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden_size, state_is_tuple=True)
                        output, state = tf.nn.dynamic_rnn(cell, coeffs[i], dtype=tf.float32)
                        outputs.append(self._get_last_out(output))
                        outputs_size += self.n_hidden_size

            # concat features into a feature
            output = tf.concat(1, outputs)

            # regression
            with tf.variable_scope("regression"):
                weight = tf.Variable(tf.truncated_normal(shape=[outputs_size, self.n_target], stddev=3.0), dtype=tf.float32)
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
            if self.regressor == "mse": #only work on the target is one-dim
                self._loss = tf.reduce_mean(tf.square(self.prediction - self.target)) + (self.w_sum + self.b_sum) * self.C
            elif self.regressor == "msvr":
                #the loss of the train set
                diff = self.prediction - self.target
                err = tf.sqrt(tf.reduce_sum(tf.square(diff), reduction_indices=1)) - self.epsilon * self.n_target
                err_greater_than_espilon = tf.cast(err > 0, tf.float32)
                total_err = tf.reduce_mean(tf.mul(tf.square(err), err_greater_than_espilon))

                self._loss = self.C * self.w_sum + total_err
            elif self.regressor == "quantile":
                diff = self.prediction - self.target
                coeff = tf.cast(diff>0, tf.float32) - self.quantile_rate
                self._loss = tf.reduce_sum(tf.mul(diff, coeff)) + (self.w_sum + self.b_sum) * self.C
            elif self.regressor == "mcc":
                theta = 20
                diff = self.prediction - self.target
                ones_like_vec = tf.ones_like(diff)
                diff_expand = tf.matmul(diff, tf.transpose(ones_like_vec))
                # self._loss = tf.reduce_mean(tf.exp(-0.5 * tf.square((diff_expand - tf.transpose(diff_expand)) / theta)))
                self._loss =  - tf.reduce_mean(self._Gaussian_Kernel(diff_expand - tf.transpose(diff_expand), 2 * theta)) - \
                            tf.reduce_mean(self._Gaussian_Kernel(diff, theta))

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
