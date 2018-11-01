from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time
import csv

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

import com.freebirdweij.goldanalyse.dl.input_datas_common as input_data
import com.freebirdweij.goldanalyse.dl.gold_price_common as gold

def placeholder_inputs(input_nums,output_nodes,num_steps,rnn_rand):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  if rnn_rand :
    inputs_placeholder = tf.placeholder(tf.float32, shape=(None,num_steps,
                                                           input_nums))
    if output_nodes == 1 :
      labels_placeholder = tf.placeholder(tf.float32, shape=(None,num_steps))
    else :
      labels_placeholder = tf.placeholder(tf.float32, shape=(None,num_steps,output_nodes))
  else :
    inputs_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           input_nums))
    if output_nodes == 1 :
      labels_placeholder = tf.placeholder(tf.float32, shape=(None))
    else :
      labels_placeholder = tf.placeholder(tf.float32, shape=(None,output_nodes))
  return inputs_placeholder, labels_placeholder


def fill_feed_dict(data_set, inputs_pl, labels_pl,batch_size,use_brnn,rnn_rand):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    inputs_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  if not rnn_rand :
    if use_brnn == True :
      inputs_feed, labels_feed = data_set.next_batch(batch_size,
                                                   FLAGS.fake_data, shuffle=False)
    else :
      inputs_feed, labels_feed = data_set.next_batch(batch_size,
                                                   FLAGS.fake_data)
  else :
      inputs_feed, labels_feed = data_set.next_batch(batch_size,
                                                   FLAGS.fake_data)
    
    
  feed_dict = {
      inputs_pl: inputs_feed,
      labels_pl: labels_feed,
  }
  return feed_dict
  
  
def fill_feed_dict_eval(data_set, inputs_pl, labels_pl,data_index):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    inputs_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  inputs_feed, labels_feed = data_set._inputs[data_index:data_index+1], data_set._labels[data_index:data_index+1]
  feed_dict = {
      inputs_pl: inputs_feed,
      labels_pl: labels_feed,
  }
  return feed_dict
  
  
def get_inference_datas(sess,
            inputs_placeholder,
            labels_placeholder,
            outputs,
            data_set,
            output_mode):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    inputs_placeholder: The inputs placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  #true_correct = 0.0  # Counts the number of correct predictions.
  #err_meam = 0.0
  #err_div = 0.0
  #mes_tmp = 0.0
  #steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  datas = []
  num_examples = FLAGS.eval_size
  for step in xrange(num_examples):
    a_row = []
    t1_row = []
    t2_row = []
    feed_dict = fill_feed_dict_eval(data_set,
                               inputs_placeholder,
                               labels_placeholder,
                               step)
    
    labels_placeholder_max_idx = tf.argmax(labels_placeholder,1)
    outputs_max_idx = tf.argmax(outputs,1)
    outputs_max = tf.reduce_max(outputs)
    
    inputs_placeholder_val,outdatas,labels_placeholder_max_idx_val,outputs_max_idx_val,outputs_max_val = sess.run([inputs_placeholder,
           outputs,labels_placeholder_max_idx,outputs_max_idx,outputs_max], feed_dict=feed_dict)
          
    a_row.extend(labels_placeholder_max_idx_val)
    a_row.extend(outputs_max_idx_val)
    a_row.append(outputs_max_val)
    t1_row.extend(outdatas)
    a_row.extend(t1_row[0])
    t2_row.extend(inputs_placeholder_val)
    a_row.extend(t2_row[0])
      
    datas.append(a_row)
    

  return datas

def write_a_dataset_to_a_csv(filename,
                             dataset):
  with open(filename, 'w',newline='') as csvfile:
    spamwriter = csv.writer(csvfile,dialect='excel')
    for i in range(len(dataset)):
      spamwriter.writerow(dataset[i])



def run_inference():

  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data,one_hot=FLAGS.one_hot,output_nodes=FLAGS.output_nodes)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    with tf.device('/cpu:0'):
      # Generate placeholders for the images and labels.
      inputs_placeholder, labels_placeholder = placeholder_inputs(
          FLAGS.input_nums,FLAGS.output_nodes,FLAGS.num_steps,FLAGS.rnn_rand)
    
      sess = tf.Session()
      step = 1
      (FLAGS.dropout_in,FLAGS.dropout_low,FLAGS.dropout_middle,FLAGS.dropout_high,FLAGS.is_test,FLAGS.dropout_conv1, FLAGS.dropout_conv2,FLAGS.dropout_conv3,
      FLAGS.dropout_conv4,FLAGS.dropout_conv5,FLAGS.dropout_cnn,FLAGS.dropout_blstm,FLAGS.dropout_alstm) = (1.0,1.0,1.0,1.0,True,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)
      outputs,Ylogits,update_ema = gold.inference(inputs_placeholder,FLAGS.const,FLAGS.init_struct,
              FLAGS.input_nums,FLAGS.input_nodes,FLAGS.low_nodes,FLAGS.low_nums,FLAGS.middle_nodes,
              FLAGS.middle_nums,FLAGS.high_nodes,FLAGS.high_nums,FLAGS.input_fun,FLAGS.low_fun,FLAGS.middle_fun,
              FLAGS.high_fun,FLAGS.regular,FLAGS.regular_rate,FLAGS.output_nodes,FLAGS.output_mode,
              FLAGS.use_biases,tf.cast(FLAGS.is_test,tf.bool),step,FLAGS.dropout_in,FLAGS.use_bn_input,FLAGS.dropout_low,
              FLAGS.use_bn_low,FLAGS.dropout_middle,FLAGS.use_bn_middle,FLAGS.dropout_high,FLAGS.use_bn_high,FLAGS.use_bn,
              FLAGS.use_cnn,FLAGS.need_reshape,FLAGS.x_length,FLAGS.x_width,FLAGS.x_deep,FLAGS.conv1_length,
              FLAGS.conv1_width,FLAGS.conv1_deep,FLAGS.conv2_length,FLAGS.conv2_width,FLAGS.conv2_deep,FLAGS.conv3_length,
              FLAGS.conv3_width,FLAGS.conv3_deep,FLAGS.conv4_length,FLAGS.conv4_width,FLAGS.conv4_deep,FLAGS.conv5_length,
              FLAGS.conv5_width,FLAGS.conv5_deep,FLAGS.stride_length,FLAGS.stride_width,FLAGS.pool_length,FLAGS.pool_width,
              FLAGS.pool_type,FLAGS.padding,FLAGS.fullconn_length,FLAGS.fullconn_width,FLAGS.fullconn_deep,FLAGS.dropout_conv1,
              FLAGS.dropout_conv2,FLAGS.dropout_conv3,FLAGS.dropout_conv4,FLAGS.dropout_conv5,FLAGS.dropout_cnn,FLAGS.use_bn_cnn,
              FLAGS.use_brnn,FLAGS.use_arnn,FLAGS.num_seqs,FLAGS.num_steps,FLAGS.lstm_binputs,FLAGS.lstm_bsize,FLAGS.dropout_blstm,
              FLAGS.lstm_blayers,FLAGS.num_seqs,FLAGS.num_steps,FLAGS.lstm_ainputs,FLAGS.lstm_asize, FLAGS.dropout_alstm,
              FLAGS.lstm_alayers,FLAGS.rnn_rand,FLAGS.rand_test,FLAGS.batch_size)

      saver = tf.train.Saver()


      # Run the Op to initialize the variables.
      init = tf.global_variables_initializer()
      sess.run(init)
      restorepoint_file = os.path.join(FLAGS.log_bak, '2017-9-25-low-round-class20-elu-adadela-hidden2-batchsize100-1/model.ckpt-16694999')
      saver.restore(sess, restorepoint_file)
      
      datas = get_inference_datas(sess,
                  inputs_placeholder,
                  labels_placeholder,
                  outputs,
                  data_sets.train,
                  FLAGS.output_mode)
          
      out_file = os.path.join(FLAGS.input_data_dir, 'output-hjxh365-high-class14-ori-trim16-office-2018-7-2-train')

      write_a_dataset_to_a_csv(out_file,datas)
          
          
def main(_):
  run_inference()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_learning_rate',
      type=float,
      default=0.03,
      help='max_learning_rate.'
  )
  parser.add_argument(
      '--min_learning_rate',
      type=float,
      default=0.0001,
      help='min_learning_rate.'
  )
  parser.add_argument(
      '--decay_speed',
      type=int,
      default=1000,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=50000000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--const',
      type=float,
      default=1,
      help='Nums of const.'
  )
  parser.add_argument(
      '--init_struct',
      type=str,
      default='truncated_normal',
      help='Initial Mode.'
  )
  parser.add_argument(
      '--input_nums',
      type=int,
      default=99,
      help='Number of input.'
  )
  parser.add_argument(
      '--input_nodes',
      type=int,
      default=99,
      help='Number of input_nodes.'
  )
  parser.add_argument(
      '--low_nodes',
      type=int,
      default=99,
      help='Number of low_nodes.'
  )
  parser.add_argument(
      '--low_nums',
      type=int,
      default=0,
      help='Number of low_nums.'
  )
  parser.add_argument(
      '--middle_nodes',
      type=int,
      default=99,
      help='Number of middle_nodes.'
  )
  parser.add_argument(
      '--middle_nums',
      type=int,
      default=0,
      help='Number of middle_nums.'
  )
  parser.add_argument(
      '--high_nodes',
      type=int,
      default=99,
      help='Number of high_nodes.'
  )
  parser.add_argument(
      '--high_nums',
      type=int,
      default=0,
      help='Number of high_nums.'
  )
  parser.add_argument(
      '--output_nodes',
      type=int,
      default=1,
      help='Number of output_nodes.'
  )
  parser.add_argument(
      '--regular',
      type=str,
      default='l2',
      help='Regular mode.'
  )
  parser.add_argument(
      '--regular_rate',
      type=float,
      default=0.5,
      help='regular_rate.'
  )
  parser.add_argument(
      '--input_fun',
      type=str,
      default='relu',
      help='Input function.'
  )
  parser.add_argument(
      '--low_fun',
      type=str,
      default='relu',
      help='low_fun function.'
  )
  parser.add_argument(
      '--middle_fun',
      type=str,
      default='relu',
      help='middle_fun function.'
  )
  parser.add_argument(
      '--high_fun',
      type=str,
      default='relu',
      help='high_fun function.'
  )
  parser.add_argument(
      '--act_fun',
      type=str,
      default='relu',
      help='Active function.'
  )
  parser.add_argument(
      '--drop_out',
      type=str,
      default=None,
      help='Active function.'
  )
  parser.add_argument(
      '--drop_rate',
      type=float,
      default=0.5,
      help='regular_rate.'
  )
  parser.add_argument(
      '--dropout_in',
      type=float,
      default=1.0,
      help='dropout_in.'
  )
  parser.add_argument(
      '--dropout_low',
      type=float,
      default=1.0,
      help='dropout_low.'
  )
  parser.add_argument(
      '--dropout_middle',
      type=float,
      default=1.0,
      help='dropout_middle.'
  )
  parser.add_argument(
      '--dropout_high',
      type=float,
      default=1.0,
      help='dropout_high.'
  )
  parser.add_argument(
      '--dropout_conv1',
      type=float,
      default=1.0,
      help='dropout_conv1.'
  )
  parser.add_argument(
      '--dropout_conv2',
      type=float,
      default=1.0,
      help='dropout_conv2.'
  )
  parser.add_argument(
      '--dropout_conv3',
      type=float,
      default=1.0,
      help='dropout_conv3.'
  )
  parser.add_argument(
      '--dropout_conv4',
      type=float,
      default=1.0,
      help='dropout_conv4.'
  )
  parser.add_argument(
      '--dropout_conv5',
      type=float,
      default=1.0,
      help='dropout_conv5.'
  )
  parser.add_argument(
      '--dropout_cnn',
      type=float,
      default=1.0,
      help='dropout_cnn.'
  )
  parser.add_argument(
      '--dropout_blstm',
      type=float,
      default=1.0,
      help='dropout_blstm.'
  )
  parser.add_argument(
      '--dropout_alstm',
      type=float,
      default=1.0,
      help='dropout_alstm.'
  )
  parser.add_argument(
      '--hidden_nums',
      type=int,
      default=10,
      help='Number of hidden_layers.'
  )
  parser.add_argument(
      '--hidden_nodes',
      type=int,
      default=99,
      help='Number of hidden_nodes.'
  )
  parser.add_argument(
      '--output_mode',
      type=str,
      default='regression',
      help='Output_mode.'
  )
  parser.add_argument(
      '--train_mode',
      type=str,
      default='Gradient',
      help='Train_mode.'
  )
  parser.add_argument(
      '--use_biases',
      type=str,
      default='no',
      help='use biases.'
  )
  parser.add_argument(
      '--use_average',
      type=str,
      default='no',
      help='use average.'
  )
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.5,
      help='momentum.'
  )
  parser.add_argument(
      '--decay',
      type=float,
      default=0.99,
      help='decay.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=10,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--eval_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--test_size',
      type=int,
      default=100,
      help='Test size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='/home/freebirdweij/tf_works/autd/use',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/home/freebirdweij/tf_works/autd/logs',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--log_bak',
      type=str,
      default='/home/freebirdweij/tf_works/autd/logs_bak',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      type=bool,
      default=False,
      help='If true, uses fake data for unit testing.',
      #action='store_true'
  )
  parser.add_argument(
      '--one_hot',
      type=bool,
      default=False,
      help='If true, uses one hot labels.',
      #action='store_true'
  )
  parser.add_argument(
      '--is_test',
      type=bool,
      default=True,
      help='If true, is_test.',
      #action='store_true'
  )
  parser.add_argument(
      '--use_bn_input',
      type=bool,
      default=False,
      help='If true, use_bn_input.',
      #action='store_true'
  )
  parser.add_argument(
      '--use_bn_low',
      type=bool,
      default=False,
      help='If true, use_bn_low.',
      #action='store_true'
  )
  parser.add_argument(
      '--use_bn_middle',
      type=bool,
      default=False,
      help='If true, use_bn_middle.',
      #action='store_true'
  )
  parser.add_argument(
      '--use_bn_high',
      type=bool,
      default=False,
      help='If true, use_bn_high.',
      #action='store_true'
  )
  parser.add_argument(
      '--use_bn',
      type=bool,
      default=False,
      help='If true, use_bn.',
      #action='store_true'
  )
  parser.add_argument(
      '--use_bn_cnn',
      type=bool,
      default=False,
      help='If true, use_bn_cnn.',
      #action='store_true'
  )
  parser.add_argument(
      '--use_brnn',
      type=bool,
      default=False,
      help='If true, use before rnn.',
      #action='store_true'
  )
  parser.add_argument(
      '--use_arnn',
      type=bool,
      default=False,
      help='If true, use after rnn.',
      #action='store_true'
  )
  parser.add_argument(
      '--rnn_rand',
      type=bool,
      default=False,
      help='If true, use rnn_rand ',
      #action='store_true'
  )
  parser.add_argument(
      '--rand_test',
      type=bool,
      default=False,
      help='If true, use rand_test ',
      #action='store_true'
  )
  parser.add_argument(
      '--use_cnn',
      type=bool,
      default=False,
      help='If true, use_cnn.',
      #action='store_true'
  )
  parser.add_argument(
      '--need_reshape',
      type=bool,
      default=False,
      help='If true, use_cnn need_reshape.',
      #action='store_true'
  )
  parser.add_argument(
      '--x_length',
      type=int,
      default=10,
      help='Number of x_length.'
  )
  parser.add_argument(
      '--x_width',
      type=int,
      default=10,
      help='Number of x_width.'
  )
  parser.add_argument(
      '--x_deep',
      type=int,
      default=1,
      help='Number of x_deep.'
  )
  parser.add_argument(
      '--conv1_length',
      type=int,
      default=0,
      help='Number of conv1_length.'
  )
  parser.add_argument(
      '--conv1_width',
      type=int,
      default=0,
      help='Number of conv1_width.'
  )
  parser.add_argument(
      '--conv1_deep',
      type=int,
      default=1,
      help='Number of conv1_deep.'
  )
  parser.add_argument(
      '--conv2_length',
      type=int,
      default=0,
      help='Number of conv2_length.'
  )
  parser.add_argument(
      '--conv2_width',
      type=int,
      default=0,
      help='Number of conv2_width.'
  )
  parser.add_argument(
      '--conv2_deep',
      type=int,
      default=1,
      help='Number of conv2_deep.'
  )
  parser.add_argument(
      '--conv3_length',
      type=int,
      default=0,
      help='Number of conv3_length.'
  )
  parser.add_argument(
      '--conv3_width',
      type=int,
      default=0,
      help='Number of conv3_width.'
  )
  parser.add_argument(
      '--conv3_deep',
      type=int,
      default=1,
      help='Number of conv3_deep.'
  )
  parser.add_argument(
      '--conv4_length',
      type=int,
      default=0,
      help='Number of conv4_length.'
  )
  parser.add_argument(
      '--conv4_width',
      type=int,
      default=0,
      help='Number of conv4_width.'
  )
  parser.add_argument(
      '--conv4_deep',
      type=int,
      default=1,
      help='Number of conv4_deep.'
  )
  parser.add_argument(
      '--conv5_length',
      type=int,
      default=0,
      help='Number of conv5_length.'
  )
  parser.add_argument(
      '--conv5_width',
      type=int,
      default=0,
      help='Number of conv5_width.'
  )
  parser.add_argument(
      '--conv5_deep',
      type=int,
      default=1,
      help='Number of conv5_deep.'
  )
  parser.add_argument(
      '--stride_length',
      type=int,
      default=1,
      help='Number of stride_length.'
  )
  parser.add_argument(
      '--stride_width',
      type=int,
      default=1,
      help='Number of stride_width.'
  )
  parser.add_argument(
      '--pool_length',
      type=int,
      default=0,
      help='Number of pool_length.'
  )
  parser.add_argument(
      '--pool_width',
      type=int,
      default=0,
      help='Number of pool_width.'
  )
  parser.add_argument(
      '--pool_type',
      type=str,
      default='max',
      help='pool_type.'
  )
  parser.add_argument(
      '--padding',
      type=str,
      default='SAME',
      help='padding.'
  )
  parser.add_argument(
      '--fullconn_length',
      type=int,
      default=10,
      help='Number of fullconn_length.'
  )
  parser.add_argument(
      '--fullconn_width',
      type=int,
      default=10,
      help='Number of fullconn_width.'
  )
  parser.add_argument(
      '--fullconn_deep',
      type=int,
      default=1,
      help='Number of fullconn_deep.'
  )
  parser.add_argument(
      '--num_seqs',
      type=int,
      default=1,
      help='Number of num_seqs.'
  )
  parser.add_argument(
      '--num_steps',
      type=int,
      default=1,
      help='Number of num_steps.'
  )
  parser.add_argument(
      '--lstm_binputs',
      type=int,
      default=1,
      help='Number of lstm_binputs.'
  )
  parser.add_argument(
      '--lstm_bsize',
      type=int,
      default=1,
      help='Number of lstm_bsize.'
  )
  parser.add_argument(
      '--lstm_blayers',
      type=int,
      default=1,
      help='Number of lstm_blayers.'
  )
  parser.add_argument(
      '--lstm_ainputs',
      type=int,
      default=1,
      help='Number of lstm_ainputs.'
  )
  parser.add_argument(
      '--lstm_asize',
      type=int,
      default=1,
      help='Number of lstm_asize.'
  )
  parser.add_argument(
      '--lstm_alayers',
      type=int,
      default=1,
      help='Number of lstm_alayers.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
