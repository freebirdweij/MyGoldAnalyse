from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
import random
import tensorflow as tf

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


def dense_to_one_hot(labels_dense, num_classes,rnn_rand):
  """Convert class labels from scalars to one-hot vectors."""
  if rnn_rand :
    num_labels = labels_dense.shape[0] * labels_dense.shape[1]
  else :
    num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  if rnn_rand :
    labels_one_hot = labels_one_hot.reshape(labels_dense.shape[0],labels_dense.shape[1],num_classes)
  return labels_one_hot


class DataSet(object):

  def __init__(self,
               inputs,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=False,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid input dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert inputs.shape[0] == labels.shape[0], (
          'inputs.shape: %s labels.shape: %s' % (inputs.shape, labels.shape))
      self._num_examples = inputs.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert inputs.shape[3] == 1
        inputs = inputs.reshape(inputs.shape[0],
                                inputs.shape[1] * inputs.shape[2])
      #if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        #images = images.astype(numpy.float32)
        #images = numpy.multiply(images, 1.0 / 255.0)
    self._inputs = inputs
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def inputs(self):
    return self._inputs

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._inputs = self.inputs[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      inputs_rest_part = self._inputs[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._inputs = self.inputs[perm]
        self._labels = self.labels[perm]
        start = 0
        # Start next epoch
        self._index_in_epoch = batch_size - rest_num_examples
        end = self._index_in_epoch
        inputs_new_part = self._inputs[start:end]
        labels_new_part = self._labels[start:end]
        return numpy.concatenate((inputs_rest_part, inputs_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
      else :
        start =  random.randint(0, self._num_examples-batch_size)
        self._index_in_epoch = start + batch_size
        end = self._index_in_epoch
        return self._inputs[start:end], self._labels[start:end]
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._inputs[start:end], self._labels[start:end]

def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   rnn_rand=False,
                   dtype=dtypes.float32,
                   reshape=False,
                   validation_size=0,
                   seed=None,
                   output_nodes=2,
                   eval_size=0,
                   test_size=0,
                   num_steps=0):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  HIGH_TRAIN_INPUTS = 'train-hjxh365-day-2018-11-2-outtrain-high-ori16.csv'
  LOW_TRAIN_INPUTS = 'train-hjxh365-day-2018-11-2-outtrain-low-ori16.csv'
  #TRAIN_LABELS = 'train-labels-365-2017-4-27-day.csv'
  HIGH_TEST_INPUTS = 'test-hjxh365-day-2018-11-2-outtrain-high-ori16.csv'
  LOW_TEST_INPUTS = 'test-hjxh365-day-2018-11-2-outtrain-low-ori16.csv'
  #TEST_LABELS = 'test-labels-365-2017-4-27-day.csv'

  if one_hot:
    train_datas = base.load_csv_without_header(HIGH_TRAIN_INPUTS,target_dtype=numpy.uint8,
                                    features_dtype=numpy.float32,target_column=0)
    test_datas = base.load_csv_without_header(HIGH_TEST_INPUTS, target_dtype=numpy.uint8,
                                    features_dtype=numpy.float32,target_column=0)
  else:
    high_train_datas = base.load_csv_without_header(HIGH_TRAIN_INPUTS,target_dtype=numpy.float32,
                                    features_dtype=numpy.float32,target_column=0)
    low_train_datas = base.load_csv_without_header(LOW_TRAIN_INPUTS,target_dtype=numpy.float32,
                                    features_dtype=numpy.float32,target_column=0)
    high_test_datas = base.load_csv_without_header(HIGH_TEST_INPUTS, target_dtype=numpy.float32,
                                    features_dtype=numpy.float32,target_column=0)
    low_test_datas = base.load_csv_without_header(LOW_TEST_INPUTS, target_dtype=numpy.float32,
                                    features_dtype=numpy.float32,target_column=0)

  high_train_inputs = high_train_datas.data
  high_train_labels = high_train_datas.target
  high_test_inputs = high_test_datas.data
  high_test_labels = high_test_datas.target
  low_train_inputs = low_train_datas.data
  low_train_labels = low_train_datas.target
  low_test_inputs = low_test_datas.data
  low_test_labels = low_test_datas.target
  
  train_inputs = high_train_inputs
  train_labels = numpy.c_[high_train_labels,low_train_labels]
  test_inputs = high_test_inputs
  test_labels = numpy.c_[high_test_labels,low_test_labels]
    
  
  if one_hot:
    #train_labels = numpy.frombuffer(train_labels, dtype=numpy.uint8)
    #test_labels = numpy.frombuffer(test_labels, dtype=numpy.uint8)
    train_labels = dense_to_one_hot(train_labels, output_nodes,rnn_rand)
    test_labels = dense_to_one_hot(test_labels, output_nodes,rnn_rand)

  if not 0 <= validation_size <= len(train_inputs) :
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_inputs), validation_size))

  validation_inputs = train_inputs[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_inputs = train_inputs[validation_size:]
  train_labels = train_labels[validation_size:]

  print('train_inputs:')
  print(train_inputs)
  print('train_labels:')
  print(train_labels)
  print('test_inputs:')
  print(test_inputs)
  print('test_labels:')
  print(test_labels)

  train = DataSet(
      train_inputs, train_labels, dtype=dtype, reshape=reshape, seed=seed)
  validation = DataSet(
      validation_inputs,
      validation_labels,
      dtype=dtype,
      reshape=reshape,
      seed=seed)
  test = DataSet(
      test_inputs, test_labels, dtype=dtype, reshape=reshape, seed=seed)

  return base.Datasets(train=train, validation=validation, test=test)
