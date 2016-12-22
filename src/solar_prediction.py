# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '21/11/2016'

import tensorflow as tf
import numpy as np
from config import Model_Config
from reader import Reader
from model import Model
from util import MSE_And_MAE, test_figure_plot

def main(_):
    #get the config
    config = Model_Config()

    n_step = config.n_step
    n_target = config.n_target
    n_input_ir = 6
    n_input_mete = 9
    n_input_sky_cam = 1000

    epoch_size = config.epoch_size
    print_step = config.print_step

    test_num = config.test_num

    #define the input and output
    x_ir = tf.placeholder(tf.float32, [None, n_step, n_input_ir])
    x_mete = tf.placeholder(tf.float32, [None, n_step, n_input_mete])
    y_ = tf.placeholder(tf.float32, [None, n_target])
    keep_prob = tf.placeholder(tf.float32)

    reader = Reader(config)

    model = Model([x_ir, x_mete], y_, keep_prob, config)

    prediction = model.prediction
    loss = model.loss
    optimize = model.optimize

    #new a saver to save the model
    saver = tf.train.Saver()

    validation_last_loss = float('inf')

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # initialize all variables
        sess.run(init_op)

        # path = tf.train.latest_checkpoint('.')
        # save_path = saver.restore(sess, path)

        for i in range(epoch_size):
            # test
            if i%config.test_step == 0:
                ir_test_input, mete_test_input, test_target = reader.get_test_set()
                test_feed = {x_ir:ir_test_input, x_mete:mete_test_input, keep_prob:1.0}
                test_result = sess.run(prediction, feed_dict=test_feed)

                diff = sorted(np.abs(test_result - test_target))
                for i in diff:
                    print i,

                # for i in range(len(test_result)):
                    # print testc_result[i], test_target[i]

                test_feed = {x_ir:ir_test_input, x_mete:mete_test_input, y_:test_target, keep_prob:1.0}
                print "test_loss = ", sess.run(loss,feed_dict=test_feed)
                #calculate the mse and mae
                mse, mae = MSE_And_MAE(test_target, test_result)
                print "Test MSE: ", mse
                print "Test MAE: ", mae

                validation_set = reader.get_validation_set()
                validation_feed = {x_ir:validation_set[0], x_mete:validation_set[1],keep_prob:1.0}
                validation_result = sess.run(prediction, feed_dict=validation_feed)
                mse, mae = MSE_And_MAE(validation_set[2], validation_result)
                print "Validation MSE: ", mse
                print "Validation MAE: ", mae

                ir_train_input, mete_train_input, train_target = reader.next_batch()
                train_feed = {x_ir: ir_train_input, x_mete:mete_train_input, keep_prob:1.0}
                train_result = sess.run(prediction, feed_dict=train_feed)
                mse, mae = MSE_And_MAE(train_target, train_result)
                print "Train MSE: ", mse
                print "Train MAE: ", mae


            #train
            batch = reader.next_batch()
            train_feed = {x_ir:batch[0], x_mete:batch[1], y_:batch[2],keep_prob:0.5}
            sess.run(optimize, feed_dict=train_feed)

            #print step
            if i%config.print_step == 0:
                print "train loss:",sess.run(loss, feed_dict=train_feed)
                print "validation loss: ", validation_last_loss

                # print "train weight sum:", sess.run(model.w_sum, feed_dict=train_feed)
                # w = sess.run(model.weight)
                # for i in w:
                #     print i,
            #validation
            validation_set = reader.get_validation_set()
            validation_feed = {x_ir:validation_set[0], x_mete:validation_set[1], y_:validation_set[2],keep_prob:1.0}
            validation_loss = sess.run(loss,feed_dict=validation_feed)

            #compare the validation with the last loss
            if(validation_loss < validation_last_loss):
                validation_last_loss = validation_loss
            # else:
            #     # break
            #     print "break"

            # print "validation loss: ", validation_loss
            if i%200 == 0 and i > 0:
                save_path = saver.save(sess, "model.ckpt")
                print "save the model"


if __name__ == "__main__":
    tf.app.run()
