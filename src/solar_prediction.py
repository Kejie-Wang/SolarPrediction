# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '21/11/2016'

import tensorflow as tf
import numpy as np
from config import Model_Config
from reader import Reader
from model import Model
from util import MSE_And_MAE, test_figure_plot

def do_eval(sess,
            prediction,
            x_ir_placeholder,
            x_mete_placeholder,
            training_placeholder,
            input_data):
    feed_dict = {
        x_ir_placeholder: input_data[0],
        x_mete_placeholder: input_data[1],
        training_placeholder: False
    }
    return sess.run(prediction, feed_dict=feed_dict)

def main(_):
    #get the config
    config = Model_Config()

    n_step = config.n_step
    n_target = config.n_target
    n_input_ir = 36
    n_input_mete = 26

    epoch_size = config.epoch_size
    print_step = config.print_step

    test_num = config.test_num

    #define the input and output
    x_ir = tf.placeholder(tf.float32, [None, n_step, n_input_ir])
    x_mete = tf.placeholder(tf.float32, [None, n_step, n_input_mete])
    y_ = tf.placeholder(tf.float32, [None, n_target])

    training = tf.placeholder(tf.bool)

    reader = Reader(config)

    model = Model([x_ir, x_mete], y_, training, config)

    prediction = model.prediction
    loss = model.loss
    optimize = model.optimize

    #new a saver to save the model
    saver = tf.train.Saver()

    validation_last_loss = float('inf')

    best_test_result = None

    with tf.Session() as sess:
        # initialize all variables
        tf.global_variables_initializer().run()

        # path = tf.train.latest_checkpoint('.')
        # save_path = saver.restore(sess, path)
        ir_train_input, mete_train_input, train_target = reader.get_train_set()
        ir_test_input, mete_test_input, test_target = reader.get_test_set()
        ir_validation_input, mete_validation_input, validation_target = reader.get_validation_set()

        for i in range(epoch_size):
            # test
            if i%config.test_step == 0:
                test_result = do_eval(sess, prediction, x_ir, x_mete, training, [ir_test_input, mete_test_input])
                mse, mae = MSE_And_MAE(test_target, test_result)
                print "Test MSE: ", mse
                print "Test MAE: ", mae

                validation_result = do_eval(sess, prediction, x_ir, x_mete, training, [ir_validation_input, mete_validation_input])
                mse, mae = MSE_And_MAE(validation_target, validation_result)
                print "Validation MSE: ", mse
                print "Validation MAE: ", mae

                train_result = do_eval(sess, prediction, x_ir, x_mete, training, [ir_train_input, mete_train_input])
                mse, mae = MSE_And_MAE(train_target, train_result)
                print "Train MSE: ", mse
                print "Train MAE: ", mae

                print "sum of w: ", sess.run(model.w_sum)

                print "\n"
                print "bias of regression: ", sess.run(model.bias)

            #train
            batch = reader.next_batch()
            train_feed = {x_ir:batch[0], x_mete:batch[1], y_:batch[2],training: True}
            sess.run(optimize, feed_dict=train_feed)

            #print step
            if i%config.print_step == 0:
                train_feed = {x_ir:batch[0], x_mete:batch[1], y_:batch[2],training: False}
                print "train loss:",sess.run(loss, feed_dict=train_feed)
                print "validation loss: ", validation_last_loss

            #validation
            validation_set = reader.get_validation_set()
            validation_feed = {x_ir:validation_set[0], x_mete:validation_set[1], y_:validation_set[2],training: False}
            validation_loss = sess.run(loss,feed_dict=validation_feed)

            if i%50 == 0 and i > 0 and validation_loss < validation_last_loss:
                save_path = saver.save(sess, "model.ckpt")
                print "save the model"

            #compare the validation with the last loss
            if validation_loss < validation_last_loss:
                validation_last_loss = validation_loss
                best_test_result = do_eval(sess, prediction, x_ir, x_mete, training, [ir_test_input, mete_test_input])

            # print "validation loss: ", validation_loss


        test_result_path = "../output/" + str(config.regressor) + "/" + str(config.h_ahead) + "_" + str(config.h_ahead + config.n_target) + ".res"
        np.savetxt(test_result_path, best_test_result, fmt="%.4f", delimiter=',')

if __name__ == "__main__":
    tf.app.run()
