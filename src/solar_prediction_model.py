# -*- coding: utf-8 -*-

import tensorflow as tf
import solar_prediction_reader
import matplotlib.pyplot as plt

class Config:

    data_path = "../dataset/solar_data.pkl"    
    #input_group_solar = ['Avg Global CMP22 (vent/cor) [W/m^2]', 'Avg Direct CHP1-1 [W/m^2]', 'Avg Diffuse 8-48 (vent) [W/m^2]']
    input_group_temp = ['Avg Zenith Angle [degrees]','Avg Azimuth Angle [degrees]','Avg Airmass','Avg Tower Dry Bulb Temp [deg C]', 
                            'Avg Deck Dry Bulb Temp [deg C]', 'Avg Tower Wet Bulb Temp [deg C]', 'Avg Tower Dew Point Temp [deg C]',
                            'Avg Total Cloud Cover [%]','Avg Opaque Cloud Cover [%]', 'Avg Precipitation [mm]']
    target_group = ['Avg Global CMP22 (vent/cor) [W/m^2]']

    input_group_solar = ['Avg Total Cloud Cover [%]','Avg Tower Dry Bulb Temp [deg C]','Avg Direct CHP1-1 [W/m^2]']

    group = []

    batch_size = 100
    n_step = 120
    n_predict = 1

    data_length = 14400 #600 days
    data_step = 24  #the step in generating the trian set if 1, most overlap; if n_step, no overlap
    train_prop = 0.8
    epoch_size = 1000
    print_step = 50
    test_step = 200

    n_hidden_solar = 64
    n_hidden_temp = 64
    n_hidden_level2 = 128

    n_model = 24

    model_path = "model.ckpt"


class SolarPredictionModel:
    def __init__(self, data, target, config, n_model):
        self._prediction = None
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


def main(_):
    #get the config
    config = Config()
    n_step = config.n_step
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

            if i%config.test_step == 0:
                test_results = []
                solar_test_input, temp_test_input, test_targets = reader.get_test_set(7)
                for k in range(n_model):
                    test_result = sess.run(predictions[k], feed_dict={x_solar[k]:solar_test_input, x_temp[k]:temp_test_input})
                    test_results.append(test_result)

                print test_targets
                print test_results
                #first test result
                test_target_all = []
                test_result_all = []

                for i in range(7):
                    test_target_all = test_target_all + test_targets[i]
                    test_result_all = test_result_all + list(zip(*test_results)[i])

                # test_target_0 = test_targets[0]
                # test_result_0 = zip(*test_results)[0]

                print "-"*5, "test target", "-"*5, "test results", "-"*5
                for i in range(len(test_target_all)):
                    print test_target_all[i], test_result_all[i]
                
                plt.figure(0)
                plt.plot(test_target_all, color='blue')
                plt.hold
                plt.plot(test_result_all, color='red')
                plt.title("Test Results")
                plt.show()

                    
        

        save_path = saver.save(sess, config.model_path)
        

        #test
        # test_results = []
        # solar_test_input, temp_test_input, test_targets = reader.get_test_set(7)
        # for k in range(n_model):
        #     test_result = sess.run(predictions[k], feed_dict={x_solar[k]:solar_test_input, x_temp[k]:temp_test_input})
        #     test_results.append(test_result)

        # #first test result
        # test_target_0 = test_targets[0]
        # test_result_0 = zip(*test_results)[0]

        # print "-"*5, "test target", "-"*5, "test results", "-"*5
        # for i in range(len(test_target_0)):
        #     print test_target_0[i], test_result_0[i]
        
        # plt.figure(0)
        # plt.plot(test_target_0, color='blue')
        # plt.hold
        # plt.plot(test_result_0, color='red')
        # plt.title("Test Results")
        # plt.show()

        



if __name__ == "__main__":
    tf.app.run()

