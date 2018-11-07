from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time

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
      labels_placeholder = tf.placeholder(tf.float32, shape=(None,num_steps,44))
  else :
    inputs_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           input_nums))
    if output_nodes == 1 :
      labels_placeholder = tf.placeholder(tf.float32, shape=(None))
    else :
      labels_placeholder = tf.placeholder(tf.float32, shape=(None,44))
  return inputs_placeholder, labels_placeholder


def placeholder_inputs_eval(eval_size,input_nums,output_nodes):
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
  inputs_placeholder = tf.placeholder(tf.float32, shape=(eval_size,
                                                         input_nums))
  if output_nodes == 1 :
    labels_placeholder = tf.placeholder(tf.float32, shape=(eval_size))
  else :
    labels_placeholder = tf.placeholder(tf.float32, shape=(eval_size,output_nodes))
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
  
  
def do_eval(sess,
            eval_correct,
            inputs_placeholder,
            labels_placeholder,
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
  true_correct = 0.0  # Counts the number of correct predictions.
  #err_meam = 0.0
  #err_div = 0.0
  #mes_tmp = 0.0
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               inputs_placeholder,
                               labels_placeholder,
                               FLAGS.batch_size,
                               FLAGS.use_brnn,
                               FLAGS.rnn_rand)
    true_tmp= sess.run(eval_correct, feed_dict=feed_dict)
    true_correct = true_correct+true_tmp
    #err_meam = err_meam+meam_tmp
    #err_div = err_div+div_tmp
    #mes_tmp = mes_tmp+tf.square(div_tmp)
  precision = float(true_correct) / num_examples
  if output_mode == 'outcomes' :
    precision = precision * FLAGS.batch_size
    true_correct = true_correct * FLAGS.batch_size
  #err_meam = float(err_meam) / num_examples
  #err_div = float(err_div) / num_examples
  #err_div = tf.sqrt(float(mes_tmp) / num_examples)
  print('  Num examples: %d  Get values: %0.04f  Precision @ 1: %0.04f' %
        (num_examples, true_correct, precision))

  return precision


  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


def run_training():

  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data,one_hot=FLAGS.one_hot,rnn_rand=FLAGS.rnn_rand,output_nodes=FLAGS.output_nodes,
                                        eval_size=FLAGS.eval_size,test_size=FLAGS.test_size,num_steps=FLAGS.num_steps)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    with tf.device('/cpu:0'):
      # Generate placeholders for the images and labels.
      inputs_placeholder, labels_placeholder = placeholder_inputs(
          FLAGS.input_nums,FLAGS.output_nodes,FLAGS.num_steps,FLAGS.rnn_rand)
    
      #inputs_placeholder_eval, labels_placeholder_eval = placeholder_inputs_eval(
       #   FLAGS.eval_size,FLAGS.input_nums,FLAGS.output_nodes)
      
      #config = tf.ConfigProto()
      #config.gpu_options.per_process_gpu_memory_fraction=0.6 
      #sess = tf.Session(config=config)
      sess = tf.Session()
      #global_step = tf.Variable(1, name='global_step', trainable=False)
      #sess.run(variable_averages_op)
      # Build a Graph that computes predictions from the inference model.
      #variable_averages = tf.train.ExponentialMovingAverage.__init__(decay=0.99,self=variable_averages)
      #print("trainable_variables:",tf.trainable_variables()[0])
      step = 1
      high_outputs,high_Ylogits,high_update_ema,low_outputs,low_Ylogits,low_update_ema = gold.inference(inputs_placeholder,FLAGS.const,FLAGS.init_struct,
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

      # Add to the Graph the Ops for loss calculation.
      high_loss,low_loss,high_profits,low_profits,realDiffPercent,high_entropy,low_entropy = gold.loss(inputs_placeholder,high_outputs,low_outputs,high_Ylogits,low_Ylogits, labels_placeholder,FLAGS.regular,FLAGS.output_mode,FLAGS.batch_size,FLAGS.use_brnn,FLAGS.num_seqs,FLAGS.num_steps,
                       FLAGS.use_arnn,FLAGS.num_seqs, FLAGS.num_steps,FLAGS.output_nodes,FLAGS.is_test,FLAGS.rnn_rand)

        
      # learning rate decay (without batch norm)
      #max_learning_rate = 0.003
      #min_learning_rate = 0.0001
      #decay_speed = 2000
      # learning rate decay (with batch norm)
      if FLAGS.learning_rate > 5 :
        learning_rate = FLAGS.min_learning_rate + (FLAGS.max_learning_rate - FLAGS.min_learning_rate) * np.math.exp(-step/FLAGS.decay_speed)
      else :
        learning_rate = FLAGS.learning_rate

      # Add to the Graph the Ops that calculate and apply gradients.
      if FLAGS.use_average == 'yes' :
        high_entropy_step,low_entropy_step,high_train_step,low_train_step = gold.training( high_loss,low_loss,high_entropy,low_entropy, learning_rate,FLAGS.train_mode,FLAGS.momentum,FLAGS.decay)
        variable_averages = tf.train.ExponentialMovingAverage(decay=0.99,num_updates=step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(variables_averages_op,high_train_step,low_train_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
      else :
        high_entropy_step,low_entropy_step,high_train_step,low_train_step = gold.training( high_loss,low_loss,high_entropy,low_entropy, learning_rate,FLAGS.train_mode,FLAGS.momentum,FLAGS.decay)
        train_op = tf.group(high_entropy_step,low_entropy_step,high_train_step,low_train_step)
        saver = tf.train.Saver()
      # Add the Op to compare the logits to the labels during evaluation.

      # Add the variable initializer Op.
      if FLAGS.use_average == 'yes' :
        outputs_average = gold.inference_average(inputs_placeholder,FLAGS.const,FLAGS.init_struct,FLAGS.input_nums,
              FLAGS.input_nodes,FLAGS.low_nodes,FLAGS.low_nums,FLAGS.middle_nodes,FLAGS.middle_nums,FLAGS.high_nodes,
              FLAGS.high_nums,FLAGS.input_fun,FLAGS.low_fun,FLAGS.middle_fun,FLAGS.high_fun,FLAGS.regular,
              FLAGS.regular_rate,FLAGS.output_nodes,FLAGS.output_mode,FLAGS.use_biases,tf.cast(True,tf.bool),step,
              FLAGS.use_bn_input,FLAGS.use_bn_low,FLAGS.use_bn_middle,FLAGS.use_bn_high,FLAGS.use_bn,variable_averages,
              FLAGS.use_cnn,FLAGS.need_reshape,FLAGS.x_length,FLAGS.x_width,FLAGS.x_deep,FLAGS.conv1_length,
              FLAGS.conv1_width,FLAGS.conv1_deep,FLAGS.conv2_length,FLAGS.conv2_width,FLAGS.conv2_deep,FLAGS.conv3_length,
              FLAGS.conv3_width,FLAGS.conv3_deep,FLAGS.conv4_length,FLAGS.conv4_width,FLAGS.conv4_deep,FLAGS.conv5_length,
              FLAGS.conv5_width,FLAGS.conv5_deep,FLAGS.stride_length,FLAGS.stride_width,FLAGS.pool_length,FLAGS.pool_width,
              FLAGS.pool_type,FLAGS.padding,FLAGS.fullconn_length,FLAGS.fullconn_width,FLAGS.fullconn_deep,FLAGS.use_bn_cnn,
              FLAGS.use_brnn,FLAGS.use_arnn,FLAGS.num_seqs,FLAGS.num_steps,FLAGS.lstm_binputs,FLAGS.lstm_bsize,FLAGS.lstm_blayers,
              FLAGS.num_seqs,FLAGS.num_steps,FLAGS.lstm_ainputs,FLAGS.lstm_asize,FLAGS.lstm_alayers,FLAGS.rnn_rand)
      

      # Create a saver for writing training checkpoints.
      #saver = tf.train.Saver(variable_averages.variables_to_restore())

      # Create a session for running Ops on the Graph.
      #variables_averages_op = variable_averages.apply(tf.trainable_variables())
      #outputs1 = gold.inference_test(inputs_placeholder,variable_averages)

      # Instantiate a SummaryWriter to output summaries and the Graph.
      summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
      train_writer = tf.summary.FileWriter(FLAGS.log_dir+'//train')
      test_writer = tf.summary.FileWriter(FLAGS.log_dir+'//test')

      # And then after everything is built:

      # Run the Op to initialize the variables.
      init = tf.global_variables_initializer()
      sess.run(init)
      #restorepoint_file = os.path.join(FLAGS.log_bak, '2017-9-3-low-class20-elu-adadela-hidden2-batchsize100-1/model.ckpt-6417999')
      #saver.restore(sess, restorepoint_file)
      
      if FLAGS.use_average == 'yes' :
        eval_correct = gold.evaluation(outputs_average, labels_placeholder,FLAGS.output_mode,FLAGS.batch_size,FLAGS.use_brnn,FLAGS.num_seqs,FLAGS.num_steps,
                                       FLAGS.use_arnn,FLAGS.num_seqs, FLAGS.num_steps,FLAGS.output_nodes,FLAGS.is_test,FLAGS.rnn_rand)
      else :
        eval_correct = gold.evaluation(high_profits,low_profits,realDiffPercent,FLAGS.output_mode,FLAGS.batch_size,FLAGS.use_brnn,FLAGS.num_seqs,FLAGS.num_steps,
                                       FLAGS.use_arnn,FLAGS.num_seqs, FLAGS.num_steps,FLAGS.output_nodes,FLAGS.is_test,FLAGS.rnn_rand)
      
      if FLAGS.use_average == 'yes' :
        average_correct = gold.evaluation(outputs_average, labels_placeholder,FLAGS.output_mode,FLAGS.batch_size,FLAGS.use_brnn,FLAGS.num_seqs,FLAGS.num_steps,
                                          FLAGS.use_arnn,FLAGS.num_seqs, FLAGS.num_steps,FLAGS.output_nodes,FLAGS.is_test,FLAGS.rnn_rand)
        tf.summary.scalar('average_precision', average_correct)
      
      tf.summary.scalar('precision', eval_correct)

      if FLAGS.output_mode == 'classes':
        labels_placeholder_max_idx = tf.argmax(labels_placeholder,1)
        outputs_max_idx = tf.argmax(high_outputs,1)
        outputs_max = tf.reduce_max(high_outputs)
        
        tf.summary.scalar('max_probability', outputs_max)
        #tf.summary.histogram('max_probability', outputs_max)
      if FLAGS.output_mode == 'outcomes':
        labels_placeholder_max_idx = tf.argmax(labels_placeholder,1)
        high_outputs_max_idx = tf.argmax(high_outputs,1)
        high_outputs_max = tf.reduce_max(high_outputs)
        low_outputs_max_idx = tf.argmax(low_outputs,1)
        low_outputs_max = tf.reduce_max(low_outputs)
        
        tf.summary.scalar('high_max_probability', high_outputs_max)
        tf.summary.scalar('low_max_probability', low_outputs_max)
        #tf.summary.histogram('max_probability', outputs_max)
      
      #precision,err_meam,err_div =  do_eval2(outputs1,
      #            inputs_placeholder,
      #            labels_placeholder,
      #            data_sets.test)
      summary = tf.summary.merge_all()
      
      #tf.summary.scalar('precision', precision)
      #tf.summary.scalar('err_meam', err_meam)
      #tf.summary.scalar('err_div', err_div)
      #tf.summary.histogram('err_div', err_div)
      
      #variables_averages_op = variable_averages.apply(tf.trainable_variables())
      #with tf.control_dependencies([train_step,variables_averages_op]):
      #    train_op = tf.no_op(name='train')

      #saver = tf.train.Saver(variable_averages.variables_to_restore())
      # Start the training loop.
      for step in xrange(FLAGS.max_steps):
        start_time = time.time()

        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        feed_dict = fill_feed_dict(data_sets.train,
                                   inputs_placeholder,
                                   labels_placeholder,
                                   FLAGS.batch_size,
                                   FLAGS.use_brnn,
                                   FLAGS.rnn_rand)

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        #_, loss_value = sess.run([train_op,loss],
        #                         feed_dict=feed_dict)
        if FLAGS.output_mode == 'classes':
          _, loss_value,inputs_placeholder_val,labels_placeholder_val,outputs_val,outputs_use,eval_val =  sess.run([train_op,high_loss,inputs_placeholder,
               labels_placeholder_max_idx,outputs_max_idx,outputs_max,eval_correct],feed_dict=feed_dict)
        elif FLAGS.output_mode == 'outcomes':
          _, high_loss_value,low_loss_value,inputs_placeholder_val,eval_val =  sess.run([train_op,high_loss,low_loss,inputs_placeholder,
               eval_correct],feed_dict=feed_dict)
        else :
          _, loss_value,outputs_val =  sess.run([train_op,high_loss,high_outputs],feed_dict=feed_dict)

        if FLAGS.use_bn == True or FLAGS.use_bn_input == True or FLAGS.use_bn_low == True or FLAGS.use_bn_middle == True or FLAGS.use_bn_high == True or FLAGS.use_bn_cnn == True:
           sess.run([high_update_ema,low_update_ema],feed_dict=feed_dict)

        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % 100 == 0:
          # Print status to stdout.
          if FLAGS.output_mode == 'classes' and  FLAGS.batch_size > 1 :
            print('Step %d: loss = %.2f (%.3f sec) accuracy =  %.2f' % (step, loss_value, duration,eval_val))
          elif FLAGS.output_mode == 'outcomes' and  FLAGS.batch_size > 1 :
            print('Step %d: high_loss = %.2f low_loss = %.2f (%.3f sec) accuracy =  %.2f' % (step, high_loss_value,low_loss_value, duration,eval_val))
          else :
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            
          
          if FLAGS.output_mode == 'classes' and FLAGS.batch_size == 1 :
            if outputs_val == labels_placeholder_val :
                print('correct_probability ================= %.6f correct_num =========== %d ===' % (outputs_use, outputs_val))
            else :
                print('NO match :labels_num == %d == outputs_num ============= %d ' % (labels_placeholder_val, outputs_val))
              
          # Update the events file.
          summary_str = sess.run(summary, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          summary_writer.flush()

          #print('inputs_placeholder--------')
          #print(inputs_placeholder_val)

          #print('labels_placeholder--------')
          #print(labels_placeholder_val)

          #print('outputs--------')
          #print(outputs_val)

        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          #with tf.device('/cpu:0'):
            checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)
            
            (dropout_in,dropout_low,dropout_middle,dropout_high,is_test,dropout_conv1,dropout_conv2,
             dropout_conv3,dropout_conv4,dropout_conv5,dropout_cnn,dropout_blstm,dropout_alstm) = (FLAGS.dropout_in,FLAGS.dropout_low,
             FLAGS.dropout_middle,FLAGS.dropout_high,FLAGS.is_test,FLAGS.dropout_conv1,FLAGS.dropout_conv2,
             FLAGS.dropout_conv3,FLAGS.dropout_conv4,FLAGS.dropout_conv5,FLAGS.dropout_cnn,FLAGS.dropout_blstm,FLAGS.dropout_alstm)
                
            (FLAGS.dropout_in,FLAGS.dropout_low,FLAGS.dropout_middle,FLAGS.dropout_high,FLAGS.is_test,FLAGS.dropout_conv1, FLAGS.dropout_conv2,FLAGS.dropout_conv3,
             FLAGS.dropout_conv4,FLAGS.dropout_conv5,FLAGS.dropout_cnn,FLAGS.dropout_blstm,FLAGS.dropout_alstm) = (1.0,1.0,1.0,1.0,True,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)
            print('Training Data Eval:')
            do_eval(sess,
                    eval_correct,
                    inputs_placeholder,
                    labels_placeholder,
                    data_sets.train,
                    FLAGS.output_mode)

            batch_size = FLAGS.batch_size
            if FLAGS.rnn_rand :
              FLAGS.rand_test,FLAGS.batch_size = True,FLAGS.eval_size
            
            feed_dict = fill_feed_dict(data_sets.train,
                                     inputs_placeholder,
                                     labels_placeholder,
                                     FLAGS.eval_size,
                                     FLAGS.use_brnn,
                                     FLAGS.rnn_rand)
            summary_str = sess.run(summary, feed_dict=feed_dict)
            train_writer.add_summary(summary_str, step)
            train_writer.flush()

            FLAGS.rand_test,FLAGS.batch_size = False,batch_size
            
            """print('Validation Data Eval:')
            do_eval(sess,
                    eval_correct,
                    inputs_placeholder,
                    labels_placeholder,
                    data_sets.validation)
            """
            # Evaluate against the test set.
            print('Test Data Eval:')
            
            if FLAGS.use_average == 'yes' :
              do_eval(sess,
                    average_correct,
                    inputs_placeholder,
                    labels_placeholder,
                    data_sets.test,
                    FLAGS.output_mode)
            else :
              do_eval(sess,
                    eval_correct,
                    inputs_placeholder,
                    labels_placeholder,
                    data_sets.test,
                    FLAGS.output_mode)
            
            batch_size = FLAGS.batch_size
            if FLAGS.rnn_rand :
              FLAGS.rand_test,FLAGS.batch_size = True,FLAGS.test_size
            
            feed_dict = fill_feed_dict(data_sets.test,
                                     inputs_placeholder,
                                     labels_placeholder,
                                     FLAGS.test_size,
                                     FLAGS.use_brnn,
                                     FLAGS.rnn_rand)
            summary_str = sess.run(summary, feed_dict=feed_dict)
            test_writer.add_summary(summary_str, step)
            test_writer.flush()

            FLAGS.rand_test,FLAGS.batch_size = False,batch_size
            (FLAGS.dropout_in,FLAGS.dropout_low,FLAGS.dropout_middle,FLAGS.dropout_high,FLAGS.is_test,FLAGS.dropout_conv1,FLAGS.dropout_conv2,
             FLAGS.dropout_conv3,FLAGS.dropout_conv4,FLAGS.dropout_conv5,FLAGS.dropout_cnn,FLAGS.dropout_blstm,FLAGS.dropout_alstm) = (dropout_in,dropout_low,dropout_middle,
                dropout_high,is_test,dropout_conv1,dropout_conv2,dropout_conv3,dropout_conv4,dropout_conv5,dropout_cnn,dropout_blstm,dropout_alstm)
            

          
          
def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
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
      default=16,
      help='Number of input.'
  )
  parser.add_argument(
      '--input_nodes',
      type=int,
      default=16,
      help='Number of input_nodes.'
  )
  parser.add_argument(
      '--low_nodes',
      type=int,
      default=16,
      help='Number of low_nodes.'
  )
  parser.add_argument(
      '--low_nums',
      type=int,
      default=2,
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
      default=21,
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
      default='tanh',
      help='Input function.'
  )
  parser.add_argument(
      '--low_fun',
      type=str,
      default='elu',
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
      default='outcomes',
      help='Output_mode.'
  )
  parser.add_argument(
      '--train_mode',
      type=str,
      default='Adadelta',
      help='Train_mode.'
  )
  parser.add_argument(
      '--use_biases',
      type=str,
      default='yes',
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
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--eval_size',
      type=int,
      default=5500,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--test_size',
      type=int,
      default=1358,
      help='Test size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='/home/freebirdweij/tf_works/outtrain',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/home/freebirdweij/tf_works/outtrain/logs',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--log_bak',
      type=str,
      default='/home/freebirdweij/tf_works/outtrain/logs_bak',
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
      default=False,
      help='If true, is_test.',
      #action='store_true'
  )
  parser.add_argument(
      '--use_bn_input',
      type=bool,
      default=True,
      help='If true, use_bn_input.',
      #action='store_true'
  )
  parser.add_argument(
      '--use_bn_low',
      type=bool,
      default=True,
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
