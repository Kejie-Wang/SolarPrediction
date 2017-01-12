from __future__ import division
# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '05/01/2017'

import tensorflow as tf
import numpy as np
import os
import sys

HOUR_IN_A_DAY = 24

n_step = 24
h_ahead = 18
n_target = 1

model = "lin"

epsilon = 10

quantile_rate = 0.3

hidden_size = 500
lr = 0.005
batch_size = 100
epoch_size = 10000
print_step = 200
test_step = 500

ir_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/ir_train_data.csv"
mete_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/mete_train_data.csv"
target_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/target_train_data.csv"

ir_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/ir_validation_data.csv"
mete_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/mete_validation_data.csv"
target_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/target_validation_data.csv"

ir_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/ir_test_data.csv"
mete_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/mete_test_data.csv"
target_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/target_test_data.csv"

#make the saved model path
saved_model_path = "../saved_model/"
if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)
saved_model_path += "same_day/"
if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)
saved_model_path += (model + "/")
if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)
if model == "lin" or model == "msvr":
    saved_model_path += (str(h_ahead) + "_ahead" + "__" + str(n_target) + "_output" + "__" + str(hidden_size) + "_hidden/")
elif model == "quantile":
    saved_model_path += (str(h_ahead) + "_ahead" + "__" + str(n_target) + "_output" + "__" + \
                    str(epsilon) + "_epsilon" + "__" + str(quantile_rate) + "_rate" + "__" + str(hidden_size) + "_hidden/")
if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)

output_path = "../output/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
output_path += "same_day/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
output_path += (model + "/")
if not os.path.exists(output_path):
    os.mkdir(output_path)
output_path += str(h_ahead) + "_" + str(h_ahead + n_target) + ".csv"

print "saved model path:", saved_model_path
print "result saved path:", output_path

def get_valid_index(features, targets):
  num = len(features)
  missing_index = []
  for i in range(len(features)):
    if True in np.isnan(features[i]) or True in np.isnan(targets[i]):
      missing_index.append(i)
  print missing_index
  return np.setdiff1d(np.arange(num), np.array(missing_index))

#load data
ir_train_raw_data = np.loadtxt(ir_train_data_path, delimiter=',')
ir_validation_raw_data = np.loadtxt(ir_validation_data_path, delimiter=',')
ir_test_raw_data = np.loadtxt(ir_test_data_path, delimiter=',')

mete_train_raw_data = np.loadtxt(mete_train_data_path, delimiter=',')
mete_validation_raw_data = np.loadtxt(mete_validation_data_path, delimiter=',')
mete_test_raw_data = np.loadtxt(mete_test_data_path, delimiter=',')

# target_train_raw_data = np.loadtxt(target_train_data_path, delimiter=',')
# target_validation_raw_data = np.loadtxt(target_validation_data_path, delimiter=',')
# target_test_raw_data = np.loadtxt(target_test_data_path, delimiter=',')

target_train_raw_data = ir_train_raw_data[:, 7]
target_test_raw_data = ir_test_raw_data[:,7]
target_validation_raw_data = ir_validation_raw_data[:,7]

np.set_printoptions(precision=4)
print np.array(sorted(target_test_raw_data[np.arange(h_ahead, len(target_test_raw_data), HOUR_IN_A_DAY)]))

n_input_mete = mete_train_raw_data.shape[1]
mete_train_data = np.reshape(mete_train_raw_data, [-1, HOUR_IN_A_DAY, n_input_mete])
target_train_data = np.reshape(target_train_raw_data, [-1, HOUR_IN_A_DAY])[:, h_ahead:h_ahead+n_target]
mete_validation_data = np.reshape(mete_validation_raw_data, [-1, HOUR_IN_A_DAY, n_input_mete])
target_validation_data = np.reshape(target_validation_raw_data, [-1, HOUR_IN_A_DAY])[:, h_ahead:h_ahead+n_target]
mete_test_data = np.reshape(mete_test_raw_data, [-1, HOUR_IN_A_DAY, n_input_mete])
target_test_data = np.reshape(target_test_raw_data, [-1, HOUR_IN_A_DAY])[:, h_ahead:h_ahead+n_target]

#get valid index
train_valid_index = get_valid_index(mete_train_data, target_train_data)
validation_valid_index = get_valid_index(mete_validation_data, target_validation_data)
test_valid_index = get_valid_index(mete_test_data, target_test_data)


print "----------------------Running model info----------------------"
print "valid train number:", len(train_valid_index)
print "valid validation number:", len(validation_valid_index)
print "valid test number:", len(test_valid_index)
print "-------------------------------------------------------------"

#eliminate the missing value index
mete_train_data = mete_train_data[train_valid_index]
target_train_data = target_train_data[train_valid_index]
mete_validation_data = mete_validation_data[validation_valid_index]
target_validation_data = target_validation_data[validation_valid_index]
mete_test_data = mete_test_data[test_valid_index]
target_test_data = target_test_data[test_valid_index]

#concatenate all valid data
mete_valid_data = np.concatenate((mete_train_data, mete_validation_data, mete_test_data), axis=0)
#feature scale
mete_mean = np.mean(mete_valid_data, axis=0)
mete_std = np.std(mete_valid_data, axis=0)
mete_std[mete_std==0] = 1.0
mete_train_data = (mete_train_data - mete_mean) / mete_std
mete_validation_data = (mete_validation_data - mete_mean) / mete_std
mete_test_data = (mete_test_data - mete_mean) / mete_std

train_num = len(mete_train_data)
validation_num =len(mete_validation_data)
test_num = len(mete_test_data)

x_mete = tf.placeholder(tf.float32, [None, n_step, n_input_mete])
y_ = tf.placeholder(tf.float32, [None, n_target])
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("first_level1"):
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
    # cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
    # cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 2, state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(cell, x_mete, dtype=tf.float32)

    # cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
    # cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
    # outputs, states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=x_mete, dtype=tf.float32)
    # outputs = tf.concat(2, outputs)

outputs = tf.transpose(outputs, [1, 0, 2])
output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

with tf.variable_scope("regression"):
  weight = tf.Variable(tf.truncated_normal(shape=[hidden_size, n_target], stddev=2.0), dtype=tf.float32)
  bias = tf.Variable(tf.constant(0.0, shape=[n_target]), dtype=tf.float32)

prediction = tf.matmul(output, weight) + bias

MSE = tf.reduce_mean(tf.square(prediction - y_))
MAE = tf.reduce_mean(tf.abs(prediction - y_))

if model == "msvr":
    #sum of weight
    print "use the msvr model"
    m = tf.matmul(tf.transpose(weight,[1,0]), weight)
    diag = tf.matrix_diag_part(m)
    w_sum = tf.reduce_sum(diag)

    #the loss of the train set
    err = tf.sqrt(tf.reduce_sum(tf.square(prediction - y_), reduction_indices=1)) - epsilon
    err_greater_than_espilon = tf.cast(err > 0, tf.float32)
    total_err = tf.reduce_mean(tf.mul(tf.square(err), err_greater_than_espilon))

    loss = total_err + w_sum * 3
elif model == "lin":
    print "use the linear regression model"
    w_sum = tf.reduce_sum(tf.square(weight))
    loss = tf.reduce_mean(tf.square(prediction - y_))+w_sum*5
elif model == "quantile":
    print "use the quantile regression model, the quantile rate is", quantile_rate
    #define the loss
    diff = prediction - y_
    coeff = tf.cast(diff>0, tf.float32) - quantile_rate
    loss = tf.reduce_sum(tf.mul(diff, coeff))

    #define the eval
    eval_rate = tf.reduce_mean(tf.cast((prediction - y_)>0, tf.float32))

optimize = tf.train.AdamOptimizer(lr).minimize(loss)

w_sum = tf.reduce_sum(tf.square(weight))

#new a saver to save the model
saver = tf.train.Saver()
validation_last_loss = 'inf'

with tf.Session() as sess:
  # initialize all variables
  # the new method in r0.12
  # if you are use the earlier version, please replace it with initial_all_variable
  tf.global_variables_initializer().run()

  path = tf.train.latest_checkpoint(saved_model_path)
  if not (path is None):
      save_path = saver.restore(sess, path)
      print "restore model"

  for i in range(epoch_size):
    index = np.random.choice(np.arange(train_num), batch_size, replace=False)
    mete_input, target = mete_train_data[index], target_train_data[index]
    sess.run(optimize, feed_dict={x_mete: mete_input, y_: target, keep_prob: 0.8})

    if i%print_step == 0:
      l = sess.run(loss, feed_dict={x_mete: mete_train_data, y_: target_train_data, keep_prob: 1.0})
      print "Step %d train loss: %.4f" %(i, l)
      print "sum of weight", sess.run(w_sum)

    if i%test_step == 0:
      print "-----------Do a test----------"
      if model == "msvr" or model == "lin":
          mse, mae = sess.run([MSE, MAE], feed_dict={x_mete: mete_train_data, y_: target_train_data, keep_prob: 1.0})
          print "train mse:",mse
          print "train mae", mae

          l = sess.run(loss, feed_dict={x_mete: mete_validation_data, y_: target_validation_data, keep_prob: 1.0})
          print "validation loss: %.4f" %(l)
          mse, mae = sess.run([MSE, MAE], feed_dict={x_mete: mete_validation_data, y_: target_validation_data, keep_prob: 1.0})
          print "validation mse:",mse
          print "validation mae", mae

          l = sess.run(loss, feed_dict={x_mete: mete_test_data, y_: target_test_data, keep_prob: 1.0})
          print "test loss: %.4f" %(l)
          mse, mae = sess.run([MSE, MAE], feed_dict={x_mete: mete_test_data, y_: target_test_data, keep_prob: 1.0})
          print "test mse:",mse
          print "test mae:", mae
      elif model == "quantile":
          rate = sess.run(eval_rate, feed_dict={x_mete: mete_train_data, y_: target_train_data, keep_prob: 1.0})
          print "train rate", rate
          rate = sess.run(eval_rate, feed_dict={x_mete: mete_validation_data, y_: target_validation_data, keep_prob: 1.0})
          print "validation rate:", rate
          rate = sess.run(eval_rate, feed_dict={x_mete: mete_test_data, y_: target_test_data, keep_prob: 1.0})
          print "test rate", rate
      print "-----------finish the test----------"

      if i%100 == 0 and i>0:
          l = sess.run(loss, feed_dict={x_mete: mete_validation_data, y_: target_validation_data, keep_prob: 1.0})
          if l < validation_last_loss:
              validation_last_loss = l
              pred = sess.run(prediction, feed_dict={x_mete: mete_test_data, keep_prob: 1.0})
              np.savetxt(output_path, pred, fmt="%.4f", delimiter=',')
              print "save the output"
              save_path = saver.save(sess, saved_model_path + "model.ckpt")
              print "save the model to ", save_path
