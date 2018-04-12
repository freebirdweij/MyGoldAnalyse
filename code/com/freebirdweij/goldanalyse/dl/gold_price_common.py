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
      #以下进行输入数据变换
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
      #以下进行输入数据变换
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
    # 创建单个cell并堆叠多层
    def get_a_cell(lstm_size, dropout_lstm):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout_lstm)
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
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_a_cell(lstm_size, dropout_lstm) for _ in range(num_layers)]
        )

        #initial_state = tf.cond(is_test, lambda: cell.zero_state(5, tf.float32), lambda: cell.zero_state(num_seqs, tf.float32))
        #if rand_test :
        #initial_state = cell.zero_state(batch_size, tf.float32)
        #else :
        #  initial_state = cell.zero_state(num_seqs, tf.float32)

        #print('initial_state-------')
        #print(initial_state)
        # 通过dynamic_rnn对cell展开时间维度
        #lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_inputs, initial_state=initial_state)
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_inputs,dtype=tf.float32)

    #final_state = tf.reshape(final_state, [-1,num_inputs])
    #lstm_outputs = tf.slice(lstm_outputs, [0,num_steps-1,0], [num_seqs,1,num_inputs], name=None)
    #lstm_outputs = tf.reshape(lstm_outputs, [-1,num_inputs])

    # 通过lstm_outputs得到概率
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

  update_ema_all = []

  if rnn_rand == False :
    with tf.variable_scope('In',reuse=False):
      if use_bn == True :
        scale = tf.Variable(tf.ones([input_nums]))
        offset = tf.Variable(tf.zeros([input_nums]))
        inputs, update_ema = batchnorm(inputs, offset, scale, is_test, step)
        update_ema_all.append(update_ema)

    # Use CNN
    if use_cnn == True :
      with tf.variable_scope('Cnn',reuse=False):
        inputs ,update_ema_cnn = cnn_construct(inputs,need_reshape,x_length,x_width,x_deep,conv1_length,
                    conv1_width,conv1_deep,conv2_length,conv2_width,conv2_deep,conv3_length,conv3_width,
                    conv3_deep,conv4_length,conv4_width,conv4_deep,conv5_length,conv5_width,conv5_deep,
                    stride_length,stride_width,pool_length,pool_width,pool_type,padding,fullconn_length,
                    fullconn_width,fullconn_deep,dropout_conv1,dropout_conv2,dropout_conv3,dropout_conv4,
                    dropout_conv5,dropout_cnn,use_bn_cnn,is_test,step)
        update_ema_all.append(update_ema_cnn)
        input_nums = fullconn_deep
  else :
      #inputs = None
    if use_brnn == False :
      inputs = tf.concat(inputs, 1)
      inputs = tf.reshape(inputs, [-1,input_nums])
    else :
      use_brnn = True
      
  # Use BeforeRnn
  with tf.variable_scope('BeforeRnn',reuse=False):
    if use_brnn == True :
      inputs = build_lstm(inputs,num_bseqs, num_bsteps,input_nums,input_nums, dropout_blstm,num_blayers,is_test,True,rnn_rand,rand_test,batch_size)
      rnn_rand = False

  # Input
  with tf.variable_scope('Input',reuse=False):
    
    weights,biases = init_params_construct(init_struct,input_nums,input_nodes,const)

    add_regular(regular,regular_rate,weights)
    
    input_layer = activation_fun_construct(inputs,input_fun,use_biases,weights,biases)

    if dropout_in < 1 :
      input_layer = tf.nn.dropout(input_layer, dropout_in)

    if use_bn_input == True :
      scale = tf.Variable(tf.ones([input_nodes]))
      offset = tf.Variable(tf.zeros([input_nodes]))
      input_layer, update_ema = batchnorm(input_layer, offset, scale, is_test, step)
      update_ema_all.append(update_ema)
    
  outputs_temp = input_nodes
  
  # Hidden_low
  low_layer = input_layer
  if low_nums > 0 :
    nodes_inc = int((low_nodes-input_nodes)/low_nums)
    for i in range(0,low_nums):
      with tf.variable_scope('hidden_low'+str(i),reuse=False):
          
        if nodes_inc ==  0  : 
          weights,biases = init_params_construct(init_struct,input_nodes,low_nodes,const)
          input_nodes = low_nodes
        elif abs(input_nodes-low_nodes) < abs(nodes_inc) :
          weights,biases = init_params_construct(init_struct,input_nodes,low_nodes,const)
          input_nodes = low_nodes
        else :
          weights,biases = init_params_construct(init_struct,input_nodes,input_nodes+nodes_inc,const)
          input_nodes += nodes_inc

        add_regular(regular,regular_rate,weights)
    
        low_layer = activation_fun_construct(low_layer,low_fun,use_biases,weights,biases)

        if dropout_low < 1 :
          low_layer = tf.nn.dropout(low_layer, dropout_low)
        
        if use_bn_low == True :
          scale = tf.Variable(tf.ones([input_nodes]))
          offset = tf.Variable(tf.zeros([input_nodes]))
          low_layer, update_ema = batchnorm(low_layer, offset, scale, is_test, step)
          update_ema_all.append(update_ema)

    outputs_temp = input_nodes
  low_nodes = outputs_temp

  # Hidden_middle
  middle_layer = low_layer
  if middle_nums > 0 :
    nodes_inc = int((middle_nodes-low_nodes)/middle_nums)
    for i in range(0,middle_nums):
      with tf.variable_scope('hidden_middle'+str(i),reuse=False):
          
        if nodes_inc ==  0  : 
          weights,biases = init_params_construct(init_struct,low_nodes,middle_nodes,const)
          low_nodes = middle_nodes
        elif abs(low_nodes-middle_nodes) < abs(nodes_inc) :
          weights,biases = init_params_construct(init_struct,low_nodes,middle_nodes,const)
          low_nodes = middle_nodes
        else :
          weights,biases = init_params_construct(init_struct,low_nodes,low_nodes+nodes_inc,const)
          low_nodes += nodes_inc

        add_regular(regular,regular_rate,weights)
    
        middle_layer = activation_fun_construct(middle_layer,middle_fun,use_biases,weights,biases)

        if dropout_middle < 1 :
          middle_layer = tf.nn.dropout(middle_layer, dropout_middle)
        
        if use_bn_middle == True :
          scale = tf.Variable(tf.ones([low_nodes]))
          offset = tf.Variable(tf.zeros([low_nodes]))
          middle_layer, update_ema = batchnorm(middle_layer, offset, scale, is_test, step)
          update_ema_all.append(update_ema)

    outputs_temp = low_nodes
  middle_nodes = outputs_temp
    
  # Hidden_high
  high_layer = middle_layer
  if high_nums > 0 :
    nodes_inc = int((high_nodes-middle_nodes)/high_nums)
    for i in range(0,high_nums):
      with tf.variable_scope('hidden_high'+str(i),reuse=False):
          
        if nodes_inc ==  0  : 
          weights,biases = init_params_construct(init_struct,middle_nodes,high_nodes,const)
          middle_nodes = high_nodes
        elif abs(middle_nodes-high_nodes) < abs(nodes_inc) :
          weights,biases = init_params_construct(init_struct,middle_nodes,high_nodes,const)
          middle_nodes = high_nodes
        else :
          weights,biases = init_params_construct(init_struct,middle_nodes,middle_nodes+nodes_inc,const)
          middle_nodes += nodes_inc

        add_regular(regular,regular_rate,weights)
    
        high_layer = activation_fun_construct(high_layer,high_fun,use_biases,weights,biases)

        if dropout_high < 1 :
          high_layer = tf.nn.dropout(high_layer, dropout_high)
        
        if use_bn_high == True :
          scale = tf.Variable(tf.ones([middle_nodes]))
          offset = tf.Variable(tf.zeros([middle_nodes]))
          high_layer, update_ema = batchnorm(high_layer, offset, scale, is_test, step)
          update_ema_all.append(update_ema)

  outputs_temp = middle_nodes

  # Use AfterRNN
  with tf.variable_scope('AfterRNN',reuse=False):
    if use_arnn == True :
      high_layer = build_lstm(high_layer,num_aseqs, num_asteps,outputs_temp,outputs_temp, dropout_alstm,num_alayers,is_test,not use_brnn,rnn_rand,rand_test,batch_size)

  # Outputs 
  with tf.variable_scope('outputs',reuse=False):
      
    weights,biases = init_params_construct(init_struct,outputs_temp,output_nodes,const)

    add_regular(regular,regular_rate,weights)
    
    outputs,Ylogits = output_construct(high_layer,output_mode,use_biases,weights,biases)

    
  return outputs,Ylogits,update_ema_all
  
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
  
  
def loss(outputs, labels,regular,output_mode,batch_size,use_brnn,num_bseqs, num_bsteps,use_arnn,num_aseqs, num_asteps,output_nodes,is_test,rnn_rand):

  if not rnn_rand :
    if use_brnn == True :
      #labels = build_lables(labels,num_bseqs, num_bsteps,output_nodes,is_test,True)
      labels=tf.transpose(labels, [1, 0, 2])[-1]      
  else :
    labels=tf.transpose(labels, [1, 0, 2])[-1]      
    
##  if use_arnn == True :
##    #labels = build_lables(labels,num_aseqs, num_asteps,output_nodes,is_test,not use_brnn)
##    labels=tf.transpose(labels, [1, 0, 2])[-1]      
    
##  if rnn_rand or use_brnn or use_arnn :
##    outputs = tf.reshape(outputs, [-1,num_bsteps,output_nodes])
##    labels = tf.reshape(labels, [-1,num_bsteps,output_nodes])
##    outputs = tf.concat(outputs, 0)
##    outputs = tf.reshape(outputs, [-1,output_nodes])
##    labels = tf.concat(labels, 0)
##    labels = tf.reshape(labels, [-1,output_nodes])
##    to = tf.split(outputs, num_bsteps, 0)
##    tl = tf.split(labels, num_bsteps, 0)
##    outputs = to[num_bsteps-1]
##    labels = tl[num_bsteps-1]
  
  if output_mode == 'regression' :
    mse = tf.reduce_mean(tf.square(labels-outputs))
    if regular != None:
      tf.add_to_collection('losses',mse)
      loss = tf.add_n(tf.get_collection('losses'))
    else:
      loss = mse
    return loss
  
  if output_mode == 'classes' :
    #cross_entropy = -tf.reduce_sum(labels*tf.log(outputs))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
    if rnn_rand == True :
      cross_entropy = tf.reduce_mean(cross_entropy)*batch_size
    else :
      cross_entropy = tf.reduce_mean(cross_entropy)*batch_size
    if regular != None:
      tf.add_to_collection('losses',cross_entropy)
      loss = tf.add_n(tf.get_collection('losses'))
    else:
      loss = cross_entropy  
    return loss
  
def training(loss, learning_rate,train_mode,momentum,decay):

  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
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
  train_op = optimizer.minimize(loss,global_step)
  
  return train_op

def evaluation(outputs, labels,output_mode,batch_size,use_brnn,num_bseqs, num_bsteps,use_arnn,num_aseqs, num_asteps,output_nodes,is_test,rnn_rand):

  #correct_prediction = tf.abs((labels-outputs)*2/(tf.abs(labels-200)+tf.abs(outputs-200)))
  #accuracy = 1.0-tf.reduce_mean(tf.cast(correct_prediction, "float"))
  if not rnn_rand :
    if use_brnn == True :
      #labels = build_lables(labels,num_bseqs, num_bsteps,output_nodes,is_test,True)
      labels=tf.transpose(labels,[1, 0, 2])[-1]      
  else :
    labels=tf.transpose(labels,[1, 0, 2])[-1]      
    
##  if use_arnn == True :
##    #labels = build_lables(labels,num_aseqs, num_asteps,output_nodes,is_test,not use_brnn)
##    labels=tf.transpose(labels,[1, 0, 2])[-1]      

##  if rnn_rand or use_brnn or use_arnn :
##    outputs = tf.reshape(outputs, [-1,num_bsteps,output_nodes])
##    labels = tf.reshape(labels, [-1,num_bsteps,output_nodes])
##    outputs = tf.concat(outputs, 0)
##    outputs = tf.reshape(outputs, [-1,output_nodes])
##    labels = tf.concat(labels, 0)
##    labels = tf.reshape(labels, [-1,output_nodes])
##    to = tf.split(outputs, num_bsteps, 0)
##    tl = tf.split(labels, num_bsteps, 0)
##    outputs = to[num_bsteps-1]
##    labels = tl[num_bsteps-1]
  
  if output_mode == 'classes' :
    correct_prediction = tf.equal(tf.argmax(outputs,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  
  if output_mode == 'regression' :
    correct_prediction = (tf.abs(labels-outputs))/labels
    accuracy = 1.0-tf.reduce_mean(tf.cast(correct_prediction, "float"))
  
  return accuracy
