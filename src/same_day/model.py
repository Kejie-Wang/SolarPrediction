import tensorflow as tf

class Model:
    def __init__(self, data, config, keep_prob, model_index):

        self.mete_data = data[0]
        self.target = data[1]
        self.config = config

        self.model = config.model
        self.hidden_size = config.hidden_size
        self.n_target = config.n_target
        self.model_index = model_index

        self.lr = config.lr

        #regularization params
        self.C = config.C

        if self.model == 'msvr':
            self.epsilon = config.epsilon

        if self.model == 'quantile':
            self.quantile_rate = config.quantile_rate

        self.keep_prob = keep_prob

        self._prediction = None
        self._optimize = None
        self._loss = None
        self._mae = None
        self._rmse = None
        self._coverage_rate = None

        #initialize the property
        self.prediction
        self.optimize
        self.loss

    @property
    def prediction(self):
        if self._prediction is None:
            with tf.variable_scope("model" + str(self.model_index)):
                with tf.variable_scope("meteorological"):
                    cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True)
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.keep_prob)
                    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 2, state_is_tuple=True)
                    outputs, state = tf.nn.dynamic_rnn(cell, self.mete_data, dtype=tf.float32)

                outputs = tf.transpose(outputs, [1, 0, 2])
                output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

                with tf.variable_scope("regression"):
                    weight = tf.Variable(tf.truncated_normal(shape=[self.hidden_size, self.n_target], stddev=3.0), dtype=tf.float32)
                    bias = tf.Variable(tf.constant(0.0, shape=[self.n_target]), dtype=tf.float32)

                    self.w_sum = tf.reduce_sum(tf.square(weight))
                    self.bias = bias
            self._prediction = tf.matmul(output, weight) + bias

        return self._prediction

    @property
    def optimize(self):
        if self._optimize is None:
            optimizer = tf.train.AdamOptimizer(self.lr)
            self._optimize = optimizer.minimize(self.loss)
        return self._optimize

    @property
    def loss(self):
        if self._loss is None:
            if self.model == "msvr":
                print "use the msvr model"
                #the loss of the train set
                err = tf.sqrt(tf.reduce_sum(tf.square(self.prediction - target), reduction_indices=1)) - self.epsilon
                err_greater_than_espilon = tf.cast(err > 0, tf.float32)
                total_err = tf.reduce_mean(tf.mul(tf.square(err), err_greater_than_espilon))

                # #sum of weight
                # m = tf.matmul(tf.transpose(self.weight,[1,0]), self.weight)
                # diag = tf.matrix_diag_part(m)
                # w_sum = tf.reduce_sum(diag)

                self._loss = total_err + self.w_sum * self.C
            elif self.model == "lin":
                print "use the linear regression model"
                self._loss = tf.reduce_mean(tf.square(self.prediction - self.target)) + self.w_sum * self.C
            elif self.model == "quantile":
                print "use the quantile regression model, the nominal rate is", quantile_rate
                #define the loss
                diff = self.prediction - self.target
                coeff = tf.cast(diff>0, tf.float32) - self.quantile_rate
                self._loss = tf.reduce_sum(tf.mul(diff, coeff))

        return self._loss


    @property
    def mae(self):
        if self._mae is None:
            self._mae = tf.reduce_mean(tf.abs(self.prediction - self.target))
        return self._mae

    @property
    def rmse(self):
        if self._rmse is None:
            self._rmse = tf.reduce_mean(tf.square(self.prediction - self.target))
        return self._rmse

    @property
    def coverage_rate(self):
        if self._coverage_rate is None:
            self._coverage_rate = tf.reduce_mean(tf.cast((self.prediction - self.target)>0, tf.float32))
        return self._coverage_rate
