from __future__ import division
# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '05/01/2017'

import tensorflow as tf
import numpy as np

HOUR_IN_A_DAY = 24
n_step = 24
h_ahead = 9
n_target = 1

hidden_size = 500
lr = 0.005
batch_size = 100
epoch_size = 10000
print_step = 200
test_step = 500

mete_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/mete_train_data.csv"
target_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/target_train_data.csv"

mete_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/mete_validation_data.csv"
target_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/target_validation_data.csv"

mete_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/mete_test_data.csv"
target_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/target_test_data.csv"

save_folder_path = "./same_day_pred_model/"

def get_valid_index(features, targets):
  num = len(features)
  missing_index = []
  for i in range(len(features)):
    if True in np.isnan(features[i]) or True in np.isnan(targets[i]):
      missing_index.append(i)
  print missing_index
  return np.setdiff1d(np.arange(num), np.array(missing_index))

def MSE_And_MAE(targets, results):
    diff = results - targets
    mse = np.sum(diff * diff) / diff.size
    mae = np.sum(np.abs(diff)) / diff.size

    return mse, mae


#load data
mete_train_raw_data = np.loadtxt(mete_train_data_path, delimiter=',')
mete_validation_raw_data = np.loadtxt(mete_validation_data_path, delimiter=',')
mete_test_raw_data = np.loadtxt(mete_test_data_path, delimiter=',')

target_train_raw_data = np.loadtxt(target_train_data_path, delimiter=',')
target_validation_raw_data = np.loadtxt(target_validation_data_path, delimiter=',')
target_test_raw_data = np.loadtxt(target_test_data_path, delimiter=',')

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

print "valid train number:", len(train_valid_index)
print "valid validation number:", len(validation_valid_index)
print "valid test number:", len(test_valid_index)

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

with tf.variable_scope("first_level1"):
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(cell, x_mete, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

with tf.variable_scope("regression"):
  weight = tf.Variable(tf.truncated_normal(shape=[hidden_size, n_target], stddev=2.0), dtype=tf.float32)
  bias = tf.Variable(tf.constant(0.0, shape=[n_target]), dtype=tf.float32)

w_sum = tf.reduce_sum(tf.square(weight))

prediction = tf.matmul(output, weight) + bias
loss = tf.reduce_mean(tf.square(prediction - y_))+w_sum*7
optimize = tf.train.AdamOptimizer(lr).minimize(loss)

w_sum = tf.reduce_sum(tf.square(weight))

#new a saver to save the model
saver = tf.train.Saver()

with tf.Session() as sess:
  # initialize all variables
  # the new method in r0.12
  # if you are use the earlier version, please replace it with initial_all_variable
  tf.global_variables_initializer().run()

  path = tf.train.latest_checkpoint(save_folder_path)
  if not (path is None):
      save_path = saver.restore(sess, path)
      print "restore model"

  for i in range(epoch_size):
    index = np.random.choice(np.arange(train_num), batch_size, replace=False)
    mete_input, target = mete_train_data[index], target_train_data[index]
    sess.run(optimize, feed_dict={x_mete: mete_input, y_: target})

    if i%10 == 0:
        sess.run(optimize, feed_dict={x_mete: mete_validation_data, y_: target_validation_data})

    if i%print_step == 0:
      l = sess.run(loss, feed_dict={x_mete: mete_train_data, y_: target_train_data})
      print "Step %d train loss: %.4f" %(i, l)
      print "sum of weight", sess.run(w_sum)

    if i%test_step == 0:
      print "-----------Do a test----------"
      res = sess.run(prediction, feed_dict={x_mete: mete_train_data})
      mse, mae = MSE_And_MAE(target_train_data, res)
      print "train mse:",mse
      print "train mae", mae

      l = sess.run(loss, feed_dict={x_mete: mete_validation_data, y_: target_validation_data})
      print "validation loss: %.4f" %(l)
      res = sess.run(prediction, feed_dict={x_mete: mete_validation_data})
      mse, mae = MSE_And_MAE(target_validation_data, res)
      print "validation mse:",mse
      print "validation mae", mae

      l = sess.run(loss, feed_dict={x_mete: mete_test_data, y_: target_test_data})
      print "test loss: %.4f" %(l)
      res = sess.run(prediction, feed_dict={x_mete: mete_test_data})
      mse, mae = MSE_And_MAE(target_test_data, res)
      print "test mse:",mse
      print "test mae", mae
      print "-----------finish the test----------"

      if i%1000 == 0 and i>0:
          save_path = saver.save(sess, save_folder_path + "model.ckpt")
          print "save the model to ", save_path
