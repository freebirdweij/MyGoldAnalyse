from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

import numpy as np

def init_params_construct(init_struct,input_nums,input_nodes,const):
  
  weights_struct = {
      'zeros': lambda input_nums,input_nodes : tf.get_variable('weights',[input_nums, input_nodes],initializer=tf.zeros_initializer()),
      'constant': lambda input_nums,input_nodes,const : tf.get_variable('weights',[input_nums, input_nodes],initializer=tf.constant_initializer(value=const)),
      'random_normal': lambda input_nums,input_nodes,const : tf.get_variable('weights',[input_nums, input_nodes],initializer=tf.random_normal_initializer(stddev=const / math.sqrt(float(input_nodes)))),
      'truncated_normal': lambda input_nums,input_nodes,const : tf.get_variable('weights',[input_nums, input_nodes],initializer=tf.truncated_normal_initializer(stddev=const / math.sqrt(float(input_nodes)))),
      'random_uniform': lambda input_nums,input_nodes : tf.get_variable('weights',[input_nums, input_nodes],initializer=tf.random_uniform_initializer()),
      'uniform_unit': lambda input_nums,input_nodes : tf.get_variable('weights',[input_nums, input_nodes],initializer=tf.uniform_unit_scaling_initializer()),
  }
  
  biases_struct = {
      'zeros': lambda input_nodes : tf.get_variable('biases',[input_nodes],initializer=tf.zeros_initializer()),
      'constant': lambda input_nodes,const : tf.get_variable('biases',[input_nodes],initializer=tf.constant_initializer(value=const)),
      'random_normal': lambda input_nodes,const : tf.get_variable('biases',[input_nodes],initializer=tf.random_normal_initializer(stddev=const / math.sqrt(float(input_nodes)))),
      'truncated_normal': lambda input_nodes,const : tf.get_variable('biases',[input_nodes],initializer=tf.truncated_normal_initializer(stddev=const / math.sqrt(float(input_nodes)))),
      'random_uniform': lambda input_nodes : tf.get_variable('biases',[input_nodes],initializer=tf.random_uniform_initializer()),
      'uniform_unit': lambda input_nodes : tf.get_variable('biases',[input_nodes],initializer=tf.uniform_unit_scaling_initializer()),
  }
  # Get the function from switcher dictionary
  weights = weights_struct[init_struct](input_nums,input_nodes,const)
  biases = biases_struct[init_struct](input_nodes,const)
  # Execute the function
  return weights,biases

def activation_fun_construct(inputs,input_fun,use_biases,weights,biases):
  
  input_struct_ubiases = {
      'linear': lambda inputs,weights,biases : tf.matmul(inputs, weights) + biases,
      'relu': lambda inputs,weights,biases : tf.nn.relu(tf.matmul(inputs, weights) + biases),
      'relu6': lambda inputs,weights,biases : tf.nn.relu6(tf.matmul(inputs, weights) + biases),
      'softplus': lambda inputs,weights,biases : tf.nn.softplus(tf.matmul(inputs, weights) + biases),
      'sigmoid': lambda inputs,weights,biases : tf.nn.sigmoid(tf.matmul(inputs, weights) + biases),
      'tanh': lambda inputs,weights,biases : tf.nn.tanh(tf.matmul(inputs, weights) + biases),
      'elu': lambda inputs,weights,biases : tf.nn.elu(tf.matmul(inputs, weights) + biases),
      'crelu': lambda inputs,weights,biases : tf.nn.crelu(tf.matmul(inputs, weights) + biases),
      'softsign': lambda inputs,weights,biases : tf.nn.softsign(tf.matmul(inputs, weights) + biases),
  }
  
  input_struct_nbiases = {
      'linear': lambda inputs,weights : tf.matmul(inputs, weights),
      'relu': lambda inputs,weights : tf.nn.relu(tf.matmul(inputs, weights)),
      'relu6': lambda inputs,weights : tf.nn.relu6(tf.matmul(inputs, weights)),
      'softplus': lambda inputs,weights : tf.nn.softplus(tf.matmul(inputs, weights)),
      'sigmoid': lambda inputs,weights : tf.nn.sigmoid(tf.matmul(inputs, weights)),
      'tanh': lambda inputs,weights : tf.nn.tanh(tf.matmul(inputs, weights)),
      'elu': lambda inputs,weights : tf.nn.elu(tf.matmul(inputs, weights)),
      'crelu': lambda inputs,weights : tf.nn.crelu(tf.matmul(inputs, weights)),
      'softsign': lambda inputs,weights : tf.nn.softsign(tf.matmul(inputs, weights)),
  }

  if use_biases == 'yes' :
    the_layer = input_struct_ubiases[input_fun](inputs,weights,biases)
  else :
    the_layer = input_struct_nbiases[input_fun](inputs,weights)

  return the_layer

def output_construct(inputs,input_fun,use_biases,weights,biases):
  
  if use_biases == 'yes' :
    Ylogits = tf.matmul(inputs, weights) + biases
  else :
    Ylogits = tf.matmul(inputs, weights)

  output_struct = {
      'regression': lambda : Ylogits,
      'classes': lambda : tf.nn.softmax(Ylogits),
  }
  
  the_layer = output_struct[input_fun]()

  return the_layer,Ylogits

def add_regular(regular,regular_rate,weights):
  
  regular_struct = {
      'l1': lambda regular_rate,weights : tf.add_to_collection('losses',tf.contrib.layers.l1_regularizer(regular_rate)(weights)),
      'l2': lambda regular_rate,weights : tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regular_rate)(weights)),
      'l1_l2': lambda regular_rate,weights : tf.add_to_collection('losses',tf.contrib.layers.l1_l2_regularizer(regular_rate,regular_rate)(weights)),
  }

  regular_struct[regular](regular_rate,weights)
  

def batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, bnepsilon)
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    return Ylogits, tf.no_op()

def batchnorm_cnn(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

def cnn_construct(inputs,need_reshape,x_length,x_width,x_deep,conv1_length,conv1_width,conv1_deep,
                  conv2_length,conv2_width,conv2_deep,conv3_length,conv3_width,conv3_deep,conv4_length,
                  conv4_width,conv4_deep,conv5_length,conv5_width,conv5_deep,stride_length,stride_width,
                  pool_length,pool_width,pool_type,padding,fullconn_length,fullconn_width,fullconn_deep,
                  dropout_conv1,dropout_conv2,dropout_conv3,dropout_conv4,dropout_conv5,dropout_cnn,use_bn,is_test,step):
  
  if need_reshape == True :
    x_image = tf.reshape(inputs, [-1, x_length, x_width, x_deep])
  else :
    x_image = inputs

  cnn_outs = None
  cnn_deep = None
  update_ema_cnn = []
  
  if conv1_length > 0 :
    W_conv1 = weight_variable([conv1_length, conv1_width, x_deep, conv1_deep])
    b_conv1 = bias_variable([conv1_deep])
    if use_bn == True :
      Y1l = tf.nn.conv2d(x_image, W_conv1,strides=[1, stride_length, stride_width, 1], padding=padding)
      Y1bn, update_ema1 = batchnorm_cnn(Y1l, is_test, step, b_conv1, convolutional=True)
      update_ema_cnn.append(update_ema1)
      h_conv1 = tf.nn.relu(Y1bn)
    else :
      h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1,strides=[1, stride_length, stride_width, 1], padding=padding) + b_conv1)

    if dropout_conv1 < 1 :
      h_conv1 = tf.nn.dropout(h_conv1, dropout_conv1, compatible_convolutional_noise_shape(h_conv1))
      
    if pool_length > 1 :
      if pool_type == 'max' :
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, pool_length, pool_width, 1], strides=[1, pool_length, pool_width, 1], padding=padding)
      else :
        h_pool1 = tf.nn.avg_pool(h_conv1, ksize=[1, pool_length, pool_width, 1], strides=[1, pool_length, pool_width, 1], padding=padding)
    else :
      h_pool1 = h_conv1
    cnn_outs = h_pool1
    cnn_deep = conv1_deep
    
  if conv2_length > 0 :
    W_conv2 = weight_variable([conv2_length, conv2_width, conv1_deep, conv2_deep])
    b_conv2 = bias_variable([conv2_deep])
    if use_bn == True :
      Y2l = tf.nn.conv2d(h_pool1, W_conv2,strides=[1, stride_length, stride_width, 1], padding=padding)
      Y2bn, update_ema2 = batchnorm_cnn(Y2l, is_test, step, b_conv2, convolutional=True)
      update_ema_cnn.append(update_ema2)
      h_conv2 = tf.nn.relu(Y2bn)
    else :
      h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,strides=[1, stride_length, stride_width, 1], padding=padding) + b_conv2)
      
    if dropout_conv2 < 1 :
      h_conv2 = tf.nn.dropout(h_conv2, dropout_conv2, compatible_convolutional_noise_shape(h_conv2))
      
    if pool_length > 1 :
      if pool_type == 'max' :
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, pool_length, pool_width, 1], strides=[1, pool_length, pool_width, 1], padding=padding)
      else :
        h_pool2 = tf.nn.avg_pool(h_conv2, ksize=[1, pool_length, pool_width, 1], strides=[1, pool_length, pool_width, 1], padding=padding)
    else :
      h_pool2 = h_conv2
    cnn_outs = h_pool2
    cnn_deep = conv2_deep
    
  if conv3_length > 0 :
    W_conv3 = weight_variable([conv3_length, conv3_width, conv2_deep, conv3_deep])
    b_conv3 = bias_variable([conv3_deep])
    if use_bn == True :
      Y3l = tf.nn.conv2d(h_pool2, W_conv3,strides=[1, stride_length, stride_width, 1], padding=padding)
      Y3bn, update_ema3 = batchnorm_cnn(Y3l, is_test, step, b_conv3, convolutional=True)
      update_ema_cnn.append(update_ema3)
      h_conv3 = tf.nn.relu(Y3bn)
    else :
      h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3,strides=[1, stride_length, stride_width, 1], padding=padding) + b_conv3)
      
    if dropout_conv3 < 1 :
      h_conv3 = tf.nn.dropout(h_conv3, dropout_conv3, compatible_convolutional_noise_shape(h_conv3))
      
    if pool_length > 1 :
      if pool_type == 'max' :
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, pool_length, pool_width, 1], strides=[1, pool_length, pool_width, 1], padding=padding)
      else :
        h_pool3 = tf.nn.avg_pool(h_conv3, ksize=[1, pool_length, pool_width, 1], strides=[1, pool_length, pool_width, 1], padding=padding)
    else :
      h_pool3 = h_conv3
    cnn_outs = h_pool3
    cnn_deep = conv3_deep
    
  if conv4_length > 0 :
    W_conv4 = weight_variable([conv4_length, conv4_width, conv3_deep, conv4_deep])
    b_conv4 = bias_variable([conv4_deep])
    if use_bn == True :
      Y4l = tf.nn.conv2d(h_pool3, W_conv4,strides=[1, stride_length, stride_width, 1], padding=padding)
      Y4bn, update_ema4 = batchnorm_cnn(Y4l, is_test, step, b_conv4, convolutional=True)
      update_ema_cnn.append(update_ema4)
      h_conv4 = tf.nn.relu(Y4bn)
    else :
      h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4,strides=[1, stride_length, stride_width, 1], padding=padding) + b_conv4)
      
    if dropout_conv4 < 1 :
      h_conv4 = tf.nn.dropout(h_conv4, dropout_conv4, compatible_convolutional_noise_shape(h_conv4))
      
    if pool_length > 1 :
      if pool_type == 'max' :
        h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, pool_length, pool_width, 1], strides=[1, pool_length, pool_width, 1], padding=padding)
      else :
        h_pool4 = tf.nn.avg_pool(h_conv4, ksize=[1, pool_length, pool_width, 1], strides=[1, pool_length, pool_width, 1], padding=padding)
    else :
      h_pool4 = h_conv4
    cnn_outs = h_pool4
    cnn_deep = conv4_deep
    
  if conv5_length > 0 :
    W_conv5 = weight_variable([conv5_length, conv5_width, conv4_deep, conv5_deep])
    b_conv5 = bias_variable([conv5_deep])
    if use_bn == True :
      Y5l = tf.nn.conv2d(h_pool4, W_conv5,strides=[1, stride_length, stride_width, 1], padding=padding)
      Y5bn, update_ema5 = batchnorm_cnn(Y5l, is_test, step, b_conv5, convolutional=True)
      update_ema_cnn.append(update_ema5)
      h_conv5 = tf.nn.relu(Y5bn)
    else :
      h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool4, W_conv5,strides=[1, stride_length, stride_width, 1], padding=padding) + b_conv5)
      
    if dropout_conv5 < 1 :
      h_conv5 = tf.nn.dropout(h_conv5, dropout_conv5, compatible_convolutional_noise_shape(h_conv5))
      
    if pool_length > 1 :
      if pool_type == 'max' :
        h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1, pool_length, pool_width, 1], strides=[1, pool_length, pool_width, 1], padding=padding)
      else :
        h_pool5 = tf.nn.avg_pool(h_conv5, ksize=[1, pool_length, pool_width, 1], strides=[1, pool_length, pool_width, 1], padding=padding)
    else :
      h_pool5 = h_conv5
    cnn_outs = h_pool5
    cnn_deep = conv5_deep
    
  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc = weight_variable([fullconn_length * fullconn_width * cnn_deep, fullconn_deep])
  b_fc = bias_variable([fullconn_deep])

  h_pool_flat = tf.reshape(cnn_outs, [-1, fullconn_length * fullconn_width * cnn_deep])
  h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  if dropout_cnn < 1 :
    h_fc = tf.nn.dropout(h_fc, dropout_cnn)

  return  h_fc ,update_ema_cnn

def build_inputs(inputs,num_seqs, num_steps,num_inputs,is_test,is_stack):
    with tf.name_scope('lstm_inputs'):
      if is_test == True :
        num_seqs = inputs.shape[0]-num_steps
      if is_stack == True :
        t_tsr = []
        for i in range(num_seqs) :
          a_seq = tf.slice(inputs, [i,0], [num_steps,num_inputs], name=None)
          t_tsr.append(a_seq)

        lstm_inputs = tf.stack(t_tsr, axis=0, name='stack')
      else :
        lstm_inputs = tf.reshape(inputs, [-1,num_steps,num_inputs])
      #lstm_inputs = tf.concat(0,t_tsr)
      #lstm_inputs = tf.pack(t_tsr)

    return lstm_inputs

def build_lables(lables,num_seqs, num_steps,num_outputs,is_test,is_stack):
    with tf.name_scope('lstm_inputs'):
      if is_test == True :
        num_seqs = lables.shape[0]-num_steps
      if is_stack == True :
        t_tsr = []
        for i in range(num_seqs) :
          a_seq = tf.slice(lables, [i,0], [num_steps,num_outputs], name=None)
          t_tsr.append(a_seq)

        lstm_lables = tf.stack(t_tsr, axis=0, name='stack')
      else :
        lstm_lables = tf.reshape(lables, [-1,num_steps,num_outputs])
      #lstm_inputs = tf.concat(0,t_tsr)
      #lstm_inputs = tf.pack(t_tsr)
      
      seq_output = tf.concat(lstm_lables, 1)
      lstm_lables = tf.reshape(seq_output, [-1,num_outputs])

    return lstm_lables

def build_lstm(inputs,num_seqs, num_steps,num_inputs,lstm_size, dropout_lstm,num_layers,is_test,is_stack,rnn_rand,rand_test,batch_size):
    def get_a_cell(lstm_size, dropout_lstm):
        #lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        #drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout_lstm)
        drop = None
        return drop

    if rnn_rand :
      if is_stack :
        lstm_inputs = tf.reshape(inputs, [-1,num_steps,num_inputs])
      else :
        lstm_inputs = inputs
        print('lstm_inputs-------')
        print(lstm_inputs)
    else :
      lstm_inputs = build_inputs(inputs,num_seqs, num_steps,num_inputs,is_test,is_stack)
    with tf.name_scope('lstm'):
        cell = None
        #cell = tf.nn.rnn_cell.MultiRNNCell(
        #    [get_a_cell(lstm_size, dropout_lstm) for _ in range(num_layers)]
        #)

        #initial_state = tf.cond(is_test, lambda: cell.zero_state(5, tf.float32), lambda: cell.zero_state(num_seqs, tf.float32))
        #if rand_test :
        #initial_state = cell.zero_state(batch_size, tf.float32)
        #else :
        #  initial_state = cell.zero_state(num_seqs, tf.float32)

        #print('initial_state-------')
        #print(initial_state)
        #lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_inputs, initial_state=initial_state)
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_inputs,dtype=tf.float32)

    #final_state = tf.reshape(final_state, [-1,num_inputs])
    #lstm_outputs = tf.slice(lstm_outputs, [0,num_steps-1,0], [num_seqs,1,num_inputs], name=None)
    #lstm_outputs = tf.reshape(lstm_outputs, [-1,num_inputs])

##    seq_output = tf.concat(lstm_outputs, 1)
##    x = tf.reshape(seq_output, [-1,lstm_size])
    x=tf.transpose(lstm_outputs, [1, 0, 2])[-1]
            
    return x

def inference(inputs,const,init_struct,input_nums,input_nodes,low_nodes,low_nums,middle_nodes,
              middle_nums,high_nodes,high_nums,input_fun,low_fun,middle_fun,high_fun,regular,
              regular_rate,output_nodes,output_mode,use_biases,is_test,step,dropout_in,use_bn_input,
              dropout_low,use_bn_low,dropout_middle,use_bn_middle,dropout_high,use_bn_high,use_bn,use_cnn,
              need_reshape,x_length,x_width,x_deep,conv1_length,conv1_width,conv1_deep,conv2_length,
              conv2_width,conv2_deep,conv3_length,conv3_width,conv3_deep,conv4_length,conv4_width,
              conv4_deep,conv5_length,conv5_width,conv5_deep,stride_length,stride_width,pool_length,
              pool_width,pool_type,padding,fullconn_length,fullconn_width,fullconn_deep,dropout_conv1,
              dropout_conv2,dropout_conv3,dropout_conv4,dropout_conv5,dropout_cnn,use_bn_cnn,use_brnn,
              use_arnn,num_bseqs,num_bsteps,num_binputs,lstm_bsize, dropout_blstm,num_blayers,num_aseqs,
              num_asteps,num_ainputs,lstm_asize, dropout_alstm,num_alayers,rnn_rand,rand_test,batch_size):

  high_update_ema_all = []

  # Input
  with tf.variable_scope('high_Input',reuse=False):
    
    weights,biases = init_params_construct(init_struct,input_nums,input_nodes,const)

    add_regular(regular,regular_rate,weights)
    
    high_input_layer = activation_fun_construct(inputs,input_fun,use_biases,weights,biases)

    if dropout_in < 1 :
      high_input_layer = tf.nn.dropout(high_input_layer, dropout_in)

    if use_bn_input == True :
      scale = tf.Variable(tf.ones([input_nodes]))
      offset = tf.Variable(tf.zeros([input_nodes]))
      high_input_layer, high_update_ema = batchnorm(high_input_layer, offset, scale, is_test, step)
      high_update_ema_all.append(high_update_ema)
    
  high_outputs_temp = input_nodes
  
  # Hidden_low
  high_low_layer = high_input_layer
  high_input_nodes = input_nodes
  if low_nums > 0 :
    nodes_inc = int((low_nodes-high_input_nodes)/low_nums)
    for i in range(0,low_nums):
      with tf.variable_scope('high_hidden_low'+str(i),reuse=False):
          
        if nodes_inc ==  0  : 
          weights,biases = init_params_construct(init_struct,high_input_nodes,low_nodes,const)
          high_input_nodes = low_nodes
        elif abs(high_input_nodes-low_nodes) < abs(nodes_inc) :
          weights,biases = init_params_construct(init_struct,high_input_nodes,low_nodes,const)
          high_input_nodes = low_nodes
        else :
          weights,biases = init_params_construct(init_struct,high_input_nodes,high_input_nodes+nodes_inc,const)
          high_input_nodes += nodes_inc

        add_regular(regular,regular_rate,weights)
    
        high_low_layer = activation_fun_construct(high_low_layer,low_fun,use_biases,weights,biases)

        if dropout_low < 1 :
          high_low_layer = tf.nn.dropout(high_low_layer, dropout_low)
        
        if use_bn_low == True :
          scale = tf.Variable(tf.ones([high_input_nodes]))
          offset = tf.Variable(tf.zeros([high_input_nodes]))
          high_low_layer, high_update_ema = batchnorm(high_low_layer, offset, scale, is_test, step)
          high_update_ema_all.append(high_update_ema)

    high_outputs_temp = high_input_nodes
  high_low_nodes = high_outputs_temp

  # Hidden_middle
  high_middle_layer = high_low_layer
  if middle_nums > 0 :
    nodes_inc = int((middle_nodes-high_low_nodes)/middle_nums)
    for i in range(0,middle_nums):
      with tf.variable_scope('high_hidden_middle'+str(i),reuse=False):
          
        if nodes_inc ==  0  : 
          weights,biases = init_params_construct(init_struct,high_low_nodes,middle_nodes,const)
          high_low_nodes = middle_nodes
        elif abs(high_low_nodes-middle_nodes) < abs(nodes_inc) :
          weights,biases = init_params_construct(init_struct,high_low_nodes,middle_nodes,const)
          high_low_nodes = middle_nodes
        else :
          weights,biases = init_params_construct(init_struct,high_low_nodes,high_low_nodes+nodes_inc,const)
          high_low_nodes += nodes_inc

        add_regular(regular,regular_rate,weights)
    
        high_middle_layer = activation_fun_construct(high_middle_layer,middle_fun,use_biases,weights,biases)

        if dropout_middle < 1 :
          high_middle_layer = tf.nn.dropout(high_middle_layer, dropout_middle)
        
        if use_bn_middle == True :
          scale = tf.Variable(tf.ones([high_low_nodes]))
          offset = tf.Variable(tf.zeros([high_low_nodes]))
          high_middle_layer, high_update_ema = batchnorm(high_middle_layer, offset, scale, is_test, step)
          high_update_ema_all.append(high_update_ema)

    high_outputs_temp = high_low_nodes
  high_middle_nodes = high_outputs_temp
    
  # Hidden_high
  high_high_layer = high_middle_layer
  if high_nums > 0 :
    nodes_inc = int((high_nodes-high_middle_nodes)/high_nums)
    for i in range(0,high_nums):
      with tf.variable_scope('high_hidden_high'+str(i),reuse=False):
          
        if nodes_inc ==  0  : 
          weights,biases = init_params_construct(init_struct,high_middle_nodes,high_nodes,const)
          high_middle_nodes = high_nodes
        elif abs(high_middle_nodes-high_nodes) < abs(nodes_inc) :
          weights,biases = init_params_construct(init_struct,high_middle_nodes,high_nodes,const)
          high_middle_nodes = high_nodes
        else :
          weights,biases = init_params_construct(init_struct,high_middle_nodes,high_middle_nodes+nodes_inc,const)
          high_middle_nodes += nodes_inc

        add_regular(regular,regular_rate,weights)
    
        high_high_layer = activation_fun_construct(high_high_layer,high_fun,use_biases,weights,biases)

        if dropout_high < 1 :
          high_high_layer = tf.nn.dropout(high_high_layer, dropout_high)
        
        if use_bn_high == True :
          scale = tf.Variable(tf.ones([high_middle_nodes]))
          offset = tf.Variable(tf.zeros([high_middle_nodes]))
          high_high_layer, high_update_ema = batchnorm(high_high_layer, offset, scale, is_test, step)
          high_update_ema_all.append(high_update_ema)

  high_outputs_temp = high_middle_nodes

  # Outputs 
  with tf.variable_scope('high_outputs',reuse=False):
      
    weights,biases = init_params_construct(init_struct,high_outputs_temp,output_nodes,const)

    add_regular(regular,regular_rate,weights)
    
    high_outputs,high_Ylogits = output_construct(high_high_layer,output_mode,use_biases,weights,biases)

  #Construct low price network
  low_update_ema_all = []

  # Input
  with tf.variable_scope('low_Input',reuse=False):
    
    weights,biases = init_params_construct(init_struct,input_nums,input_nodes,const)

    add_regular(regular,regular_rate,weights)
    
    low_input_layer = activation_fun_construct(inputs,input_fun,use_biases,weights,biases)

    if dropout_in < 1 :
      low_input_layer = tf.nn.dropout(low_input_layer, dropout_in)

    if use_bn_input == True :
      scale = tf.Variable(tf.ones([input_nodes]))
      offset = tf.Variable(tf.zeros([input_nodes]))
      low_input_layer, low_update_ema = batchnorm(low_input_layer, offset, scale, is_test, step)
      low_update_ema_all.append(low_update_ema)
    
  low_outputs_temp = input_nodes
  
  # Hidden_low
  low_low_layer = low_input_layer
  low_input_nodes = input_nodes
  if low_nums > 0 :
    nodes_inc = int((low_nodes-low_input_nodes)/low_nums)
    for i in range(0,low_nums):
      with tf.variable_scope('low_hidden_low'+str(i),reuse=False):
          
        if nodes_inc ==  0  : 
          weights,biases = init_params_construct(init_struct,low_input_nodes,low_nodes,const)
          low_input_nodes = low_nodes
        elif abs(input_nodes-low_nodes) < abs(nodes_inc) :
          weights,biases = init_params_construct(init_struct,low_input_nodes,low_nodes,const)
          low_input_nodes = low_nodes
        else :
          weights,biases = init_params_construct(init_struct,low_input_nodes,low_input_nodes+nodes_inc,const)
          low_input_nodes += nodes_inc

        add_regular(regular,regular_rate,weights)
    
        low_low_layer = activation_fun_construct(low_low_layer,low_fun,use_biases,weights,biases)

        if dropout_low < 1 :
          low_low_layer = tf.nn.dropout(low_low_layer, dropout_low)
        
        if use_bn_low == True :
          scale = tf.Variable(tf.ones([low_input_nodes]))
          offset = tf.Variable(tf.zeros([low_input_nodes]))
          low_low_layer, low_update_ema = batchnorm(low_low_layer, offset, scale, is_test, step)
          low_update_ema_all.append(low_update_ema)

    low_outputs_temp = low_input_nodes
  low_low_nodes = low_outputs_temp

  # Hidden_middle
  low_middle_layer = low_low_layer
  if middle_nums > 0 :
    nodes_inc = int((middle_nodes-low_low_nodes)/middle_nums)
    for i in range(0,middle_nums):
      with tf.variable_scope('low_hidden_middle'+str(i),reuse=False):
          
        if nodes_inc ==  0  : 
          weights,biases = init_params_construct(init_struct,low_low_nodes,middle_nodes,const)
          low_low_nodes = middle_nodes
        elif abs(low_low_nodes-middle_nodes) < abs(nodes_inc) :
          weights,biases = init_params_construct(init_struct,low_low_nodes,middle_nodes,const)
          low_low_nodes = middle_nodes
        else :
          weights,biases = init_params_construct(init_struct,low_low_nodes,low_low_nodes+nodes_inc,const)
          low_low_nodes += nodes_inc

        add_regular(regular,regular_rate,weights)
    
        low_middle_layer = activation_fun_construct(low_middle_layer,middle_fun,use_biases,weights,biases)

        if dropout_middle < 1 :
          low_middle_layer = tf.nn.dropout(low_middle_layer, dropout_middle)
        
        if use_bn_middle == True :
          scale = tf.Variable(tf.ones([low_low_nodes]))
          offset = tf.Variable(tf.zeros([low_low_nodes]))
          low_middle_layer, low_update_ema = batchnorm(low_middle_layer, offset, scale, is_test, step)
          low_update_ema_all.append(low_update_ema)

    low_outputs_temp = low_low_nodes
  low_middle_nodes = low_outputs_temp
    
  # Hidden_high
  low_high_layer = low_middle_layer
  if high_nums > 0 :
    nodes_inc = int((high_nodes-low_middle_nodes)/high_nums)
    for i in range(0,high_nums):
      with tf.variable_scope('low_hidden_high'+str(i),reuse=False):
          
        if nodes_inc ==  0  : 
          weights,biases = init_params_construct(init_struct,low_middle_nodes,high_nodes,const)
          low_middle_nodes = high_nodes
        elif abs(low_middle_nodes-high_nodes) < abs(nodes_inc) :
          weights,biases = init_params_construct(init_struct,low_middle_nodes,high_nodes,const)
          low_middle_nodes = high_nodes
        else :
          weights,biases = init_params_construct(init_struct,low_middle_nodes,low_middle_nodes+nodes_inc,const)
          low_middle_nodes += nodes_inc

        add_regular(regular,regular_rate,weights)
    
        low_high_layer = activation_fun_construct(low_high_layer,high_fun,use_biases,weights,biases)

        if dropout_high < 1 :
          low_high_layer = tf.nn.dropout(low_high_layer, dropout_high)
        
        if use_bn_high == True :
          scale = tf.Variable(tf.ones([low_middle_nodes]))
          offset = tf.Variable(tf.zeros([low_middle_nodes]))
          low_high_layer, low_update_ema = batchnorm(low_high_layer, offset, scale, is_test, step)
          low_update_ema_all.append(low_update_ema)

  low_outputs_temp = low_middle_nodes

  # Outputs 
  with tf.variable_scope('low_outputs',reuse=False):
      
    weights,biases = init_params_construct(init_struct,low_outputs_temp,output_nodes,const)

    add_regular(regular,regular_rate,weights)
    
    low_outputs,low_Ylogits = output_construct(low_high_layer,output_mode,use_biases,weights,biases)

    
  return high_outputs,high_Ylogits,high_update_ema_all,low_outputs,low_Ylogits,low_update_ema_all
  
def inference_average(inputs,const,init_struct,input_nums,input_nodes,low_nodes,low_nums,middle_nodes,
              middle_nums,high_nodes,high_nums,input_fun,low_fun,middle_fun,high_fun,regular,
              regular_rate,output_nodes,output_mode,use_biases,is_test,step,use_bn_input,
              use_bn_low,use_bn_middle,use_bn_high,use_bn,avg_class,use_cnn,
              need_reshape,x_length,x_width,x_deep,conv1_length,conv1_width,conv1_deep,conv2_length,
              conv2_width,conv2_deep,conv3_length,conv3_width,conv3_deep,conv4_length,conv4_width,
              conv4_deep,conv5_length,conv5_width,conv5_deep,stride_length,stride_width,pool_length,
              pool_width,pool_type,padding,fullconn_length,fullconn_width,fullconn_deep,use_bn_cnn,
              use_brnn,use_arnn,num_bseqs,num_bsteps,num_binputs,lstm_bsize,num_blayers,num_aseqs,
              num_asteps,num_ainputs,lstm_asize,num_alayers,rnn_rand):

  if not rnn_rand :
    with tf.variable_scope('In',reuse=True):
      
      if use_bn == True :
        scale = tf.get_variable('scale')
        offset = tf.get_variable('offset')
        inputs, update_ema = batchnorm(inputs, offset, scale, is_test, step)

    # Use CNN
    if use_cnn == True :
      with tf.variable_scope('Cnn',reuse=True):
        inputs ,_ = cnn_construct(inputs,need_reshape,x_length,x_width,x_deep,conv1_length,
                    conv1_width,conv1_deep,conv2_length,conv2_width,conv2_deep,conv3_length,conv3_width,
                    conv3_deep,conv4_length,conv4_width,conv4_deep,conv5_length,conv5_width,conv5_deep,
                    stride_length,stride_width,pool_length,pool_width,pool_type,padding,fullconn_length,
                    fullconn_width,fullconn_deep,1.0,1.0,1.0,1.0,1.0,1.0,use_bn_cnn,is_test,step)
    else :
      use_brnn = True
      
  # Use BeforeRnn
  with tf.variable_scope('BeforeRnn',reuse=True):
    if use_brnn == True :
      inputs = build_lstm(inputs,num_bseqs, num_bsteps,input_nums,input_nums, 1.0,num_blayers,is_test,True,rnn_rand)
      rnn_rand = False

  # Input
  with tf.variable_scope('Input',reuse=True):
    
    weights = tf.get_variable('weights')
    biases = tf.get_variable('biases')
    
    input_layer = activation_fun_construct(inputs,input_fun,use_biases,avg_class.average(weights),avg_class.average(biases))
    
    if use_bn_input == True :
      scale = tf.get_variable('scale')
      offset = tf.get_variable('offset')
      input_layer, update_ema = batchnorm(input_layer, offset, scale, is_test, step)
    outputs_temp = input_layer.shape[1]
    
  
  # Hidden_low
  low_layer = input_layer
  if low_nums > 0 :
    for i in range(0,low_nums):
      with tf.variable_scope('hidden_low'+str(i),reuse=True):
          
        weights = tf.get_variable('weights')
        biases = tf.get_variable('biases')
    
        low_layer = activation_fun_construct(low_layer,low_fun,use_biases,avg_class.average(weights),avg_class.average(biases))

        if use_bn_low == True :
          scale = tf.get_variable('scale')
          offset = tf.get_variable('offset')
          low_layer, update_ema = batchnorm(low_layer, offset, scale, is_test, step)
    outputs_temp = low_layer.shape[1]

  # Hidden_middle
  middle_layer = low_layer
  if middle_nums > 0 :
    for i in range(0,middle_nums):
      with tf.variable_scope('hidden_middle'+str(i),reuse=True):
          
        weights = tf.get_variable('weights')
        biases = tf.get_variable('biases')
    
        middle_layer = activation_fun_construct(middle_layer,middle_fun,use_biases,avg_class.average(weights),avg_class.average(biases))

        
        if use_bn_middle == True :
          scale = tf.get_variable('scale')
          offset = tf.get_variable('offset')
          middle_layer, update_ema = batchnorm(middle_layer, offset, scale, is_test, step)
    outputs_temp = middle_layer.shape[1]

  # Hidden_high
  high_layer = middle_layer
  if high_nums > 0 :
    for i in range(0,low_nums):
      with tf.variable_scope('hidden_high'+str(i),reuse=True):
          
        weights = tf.get_variable('weights')
        biases = tf.get_variable('biases')
    
        high_layer = activation_fun_construct(high_layer,high_fun,use_biases,avg_class.average(weights),avg_class.average(biases))

        if use_bn_high == True :
          scale = tf.get_variable('scale')
          offset = tf.get_variable('offset')
          high_layer, update_ema = batchnorm(high_layer, offset, scale, is_test, step)

    outputs_temp = high_layer.shape[1]
  # Use AfterRNN
  with tf.variable_scope('AfterRNN',reuse=True):
    if use_arnn == True :
      high_layer = build_lstm(high_layer,num_aseqs, num_asteps,outputs_temp,outputs_temp, 1.0,num_alayers,is_test,not use_brnn,rnn_rand)

  # Outputs 
  with tf.variable_scope('outputs',reuse=True):
      
    weights = tf.get_variable('weights')
    biases = tf.get_variable('biases')
    
    outputs,Ylogits = output_construct(high_layer,output_mode,use_biases,avg_class.average(weights),avg_class.average(biases))

    
  return outputs
  
  
def loss(inputs,high_outputs,low_outputs, labels,regular,output_mode,batch_size,use_brnn,num_bseqs, num_bsteps,use_arnn,num_aseqs, num_asteps,output_nodes,is_test,rnn_rand):

  
  if output_mode == 'regression' :
    high_mse = tf.reduce_mean(tf.square(labels-high_outputs))
    if regular != None:
      tf.add_to_collection('high_losses',high_mse)
      high_loss = tf.add_n(tf.get_collection('high_losses'))
    else:
      high_loss = high_mse
    return high_loss
  
  if output_mode == 'classes' :
    #cross_entropy = -tf.reduce_sum(labels*tf.log(outputs))
    high_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=high_outputs, labels=labels)
    high_cross_entropy = tf.reduce_mean(high_cross_entropy)*batch_size
    if regular != None:
      tf.add_to_collection('high_losses',high_cross_entropy)
      high_loss = tf.add_n(tf.get_collection('high_losses'))
    else:
      high_loss = high_cross_entropy  
    return high_loss
    
  if output_mode == 'outcomes' :
    #1.Get the first and second maximum output probabilities.
    high_findMaxIndices = np.argsort(high_outputs)
    high_twoMaxIndices=high_findMaxIndices[-1:-3:-1]   #Lowindex of maximum 2 .  
    high_firstSecondProb=high_outputs[:,high_twoMaxIndices]
    #2.Get the predicted integral predict predict two values
    high_predictIndices = (high_outputs[:,0]*0+high_outputs[:,1]*1+high_outputs[:,2]*2+high_outputs[:,3]*3+high_outputs[:,4]*4+high_outputs[:,5]*5
      +high_outputs[:,6]*6+high_outputs[:,7]*7+high_outputs[:,8]*8+high_outputs[:,9]*9+high_outputs[:,10]*10
      +high_outputs[:,11]*11+high_outputs[:,12]*12+high_outputs[:,13]*13+high_outputs[:,14]*14+high_outputs[:,15]*15
      +high_outputs[:,16]*16+high_outputs[:,17]*17+high_outputs[:,18]*18+high_outputs[:,19]*19+high_outputs[:,20]*20)
    high_predictTwoProb = high_firstSecondProb[:,0]+high_firstSecondProb[:,1]
    high_predictTwoIndices = (high_twoMaxIndices[:,0]*high_firstSecondProb[:0]+high_twoMaxIndices[:,1]*high_firstSecondProb[:,1])/high_predictTwoProb
    high_intergalValues = (high_twoMaxIndices[:,0]-10)*inputs[:,1]/200+inputs[:,1]    
    high_predictValues = (high_predictIndices[:,0]-10)*inputs[:,1]/200+inputs[:,1]    
    high_predictTwoValues = (high_predictTwoIndices[:,0]-10)*inputs[:,1]/200+inputs[:,1]
    #3.Get the final values
    high_dotDifferenceValues =  high_predictTwoValues-high_predictTwoValues*0.1/100
    high_dotFivePercent = (inputs[:,1]-inputs[:,1]*0.1/100)*0.5/100
    with tf.variable_scope('high_outcomes',reuse=False):
      W_Calibrate = weight_variable()
    high_probCalibateValues = high_dotDifferenceValues+high_dotFivePercent(1-high_firstSecondProb[:,0])*W_Calibrate   
    
    '''Construct low price lose'''
    #1.Get the first and second maximum output probabilities.
    low_findMaxIndices = np.argsort(low_outputs)
    low_twoMaxIndices=low_findMaxIndices[-1:-3:-1]   #Lowindex of maximum 2 .  
    low_firstSecondProb=low_outputs[:,low_twoMaxIndices]
    #2.Get the predicted integral predict predict two values
    low_predictIndices = (low_outputs[:,0]*0+low_outputs[:,1]*1+low_outputs[:,2]*2+low_outputs[:,3]*3+low_outputs[:,4]*4+low_outputs[:,5]*5
      +low_outputs[:,6]*6+low_outputs[:,7]*7+low_outputs[:,8]*8+low_outputs[:,9]*9+low_outputs[:,10]*10
      +low_outputs[:,11]*11+low_outputs[:,12]*12+low_outputs[:,13]*13+low_outputs[:,14]*14+low_outputs[:,15]*15
      +low_outputs[:,16]*16+low_outputs[:,17]*17+low_outputs[:,18]*18+low_outputs[:,19]*19+low_outputs[:,20]*20)
    low_predictTwoProb = low_firstSecondProb[:,0]+low_firstSecondProb[:,1]
    low_predictTwoIndices = (low_twoMaxIndices[:,0]*low_firstSecondProb[:0]+low_twoMaxIndices[:,1]*low_firstSecondProb[:,1])/low_predictTwoProb
    low_intergalValues = (low_twoMaxIndices[:,0]-10)*inputs[:,2]/200+inputs[:,2]    
    low_predictValues = (low_predictIndices[:,0]-10)*inputs[:,2]/200+inputs[:,2]    
    low_predictTwoValues = (low_predictTwoIndices[:,0]-10)*inputs[:,2]/200+inputs[:,2]
    #3.Get the final values
    low_dotDifferenceValues =  low_predictTwoValues-low_predictTwoValues*0.1/100
    low_dotFivePercent = (inputs[:,2]-inputs[:,2]*0.1/100)*0.5/100
    with tf.variable_scope('low_outcomes',reuse=False):
      W_Calibrate = weight_variable()
    low_probCalibateValues = low_dotDifferenceValues+low_dotFivePercent(1-low_firstSecondProb[:,0])*W_Calibrate 
    
    #Get differences of two predicted prices
    diffPercentOfPredict = (high_dotDifferenceValues-low_dotDifferenceValues)*100/low_dotDifferenceValues
    lowValueMadeByHigh = high_probCalibateValues-low_dotDifferenceValues*diffPercentOfPredict/100 
    highValueMadeByLow = low_probCalibateValues+low_dotDifferenceValues*diffPercentOfPredict/100
    #Get real differences and prices
    realHighDotValue = labels[:,0]-labels[:,0]*0.1/100
    realLowDotValue = labels[:,1]+labels[:,1]*0.1/100
    realDiffPercent = (realHighDotValue-realLowDotValue)*100/realLowDotValue
    #Make final two lost
    if realHighDotValue >= high_probCalibateValues and realHighDotValue <= high_probCalibateValues+high_dotFivePercent :
      if realLowDotValue <= lowValueMadeByHigh :
        high_profits = diffPercentOfPredict*0.5
      else :
        high_profits = (realDiffPercent-0.5)*0.5*0.5
    else :
      if realHighDotValue > high_probCalibateValues+high_dotFivePercent :
        high_profits = -0.25
      else :
        high_profits = 0
        
    if realLowDotValue <= low_probCalibateValues and realLowDotValue >= low_probCalibateValues-low_dotFivePercent :
      if realHighDotValue >= highValueMadeByLow :
        low_profits = diffPercentOfPredict*0.5
      else :
        low_profits = (realDiffPercent-0.5)*0.5*0.5
    else :
      if realLowDotValue < low_probCalibateValues-low_dotFivePercent :
        low_profits = -0.25
      else :
        low_profits = 0
        
    high_mse = tf.reduce_mean(tf.square(realDiffPercent*0.5-high_profits))
    if regular != None:
      tf.add_to_collection('high_losses',high_mse)
      high_loss = tf.add_n(tf.get_collection('high_losses'))
    else:
      high_loss = high_mse
      
    low_mse = tf.reduce_mean(tf.square(realDiffPercent*0.5-low_profits))
    if regular != None:
      tf.add_to_collection('low_losses',low_mse)
      low_loss = tf.add_n(tf.get_collection('low_losses'))
    else:
      low_loss = low_mse
  
    return high_loss,low_loss,high_profits,low_profits,realDiffPercent
  
def training(high_loss,low_loss, learning_rate,train_mode,momentum,decay):

  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('high_loss', high_loss)
  tf.summary.scalar('low_loss', low_loss)
  # Create the gradient descent optimizer with the given learning rate.
  if train_mode == 'Gradient' :
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  if train_mode == 'Adadelta' :
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=0.95,
              epsilon=1e-08,use_locking=False, name='Adadelta')
  if train_mode == 'Adam' :
    optimizer = tf.train.AdamOptimizer(learning_rate)
  if train_mode == 'Adagrad' :
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  if train_mode == 'Momentum' :
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum)
  if train_mode == 'Ftrl' :
    optimizer = tf.train.FtrlOptimizer(learning_rate,momentum)
  if train_mode == 'RMSProp' :
    optimizer = tf.train.RMSPropOptimizer(learning_rate,decay,momentum)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  high_train_op = optimizer.minimize(high_loss,global_step)
  low_train_op = optimizer.minimize(low_loss,global_step)
  
  return high_train_op,low_train_op

def evaluation(high_outputs,low_outputs, labels,output_mode,batch_size,use_brnn,num_bseqs, num_bsteps,use_arnn,num_aseqs, num_asteps,output_nodes,is_test,rnn_rand):

  
  if output_mode == 'classes' :
    correct_prediction = tf.equal(tf.argmax(high_outputs,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  
  if output_mode == 'regression' :
    correct_prediction = (tf.abs(labels-high_outputs))/labels
    accuracy = 1.0-tf.reduce_mean(tf.cast(correct_prediction, "float"))
  
  if output_mode == 'outcomes' :
    correct_prediction = (tf.abs(high_outputs+low_outputs))/labels
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  
  return accuracy
