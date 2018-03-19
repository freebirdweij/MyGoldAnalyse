from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import argparse
import os.path
import sys
import collections
import csv
import os
import random
import tempfile
import time
import array 
import numpy as np

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_csv_with_header(filename,
                         target_dtype,
                         features_dtype,
                         target_column=-1):
  """Load dataset from CSV file with a header row."""
  with open(filename, 'r') as csv_file:
    data_file = csv.reader(csv_file)
    header = next(data_file)
    n_samples = int(header[0])
    n_features = int(header[1])
    data = np.zeros((n_samples, n_features), dtype=features_dtype)
    target = np.zeros((n_samples,), dtype=target_dtype)
    for i, row in enumerate(data_file):
      target[i] = np.asarray(row.pop(target_column), dtype=target_dtype)
      data[i] = np.asarray(row, dtype=features_dtype)

  return Dataset(data=data, target=target)

def load_csv_without_header(filename,
                            target_dtype,
                            features_dtype,
                            target_column=-1):
  """Load dataset from CSV file without a header row."""
  with open(filename, 'r') as csv_file:
    data_file = csv.reader(csv_file)
    data, target = [], []
    for row in data_file:
      target.append(row.pop(target_column))
      data.append(np.asarray(row, dtype=features_dtype))

  target = np.array(target, dtype=target_dtype)
  data = np.array(data)
  return Dataset(data=data, target=target)

def load_csv_datas_head_row(filename,
                         features_dtype,
                         target_column=-1):
  """Load dataset from CSV file with a header row."""
  with open(filename, 'r') as csv_file:
    data_file = csv.reader(csv_file)
    header = next(data_file)
    data = []
    bit_numbers = trans_a_row_to_bit_numbers(header)
    for row in data_file:
      data.append(np.asarray(row, dtype=features_dtype))

  return Dataset(data=data, target=bit_numbers)

def write_a_dataset_to_a_csv(filename,
                             dataset):
  with open(filename, 'w') as csvfile:
    spamwriter = csv.writer(csvfile,dialect='excel')
    for i in range(len(dataset)):
      spamwriter.writerow(dataset[i])


def trans_a_row_to_bit_numbers(a_row):
  bit_numbers = []
  for i in range(len(a_row)):
    a_data = bin(int(a_row[i]))
    charArray  = list( a_data )
    bit_numbers.append(len(charArray)-2)

  return bit_numbers

def trans_a_dec_to_bin_value_array(a_dec,w_value,bit_number):
  a_data = bin(math.ceil(a_dec))
  charArray  = list( a_data ) 
  dataArray = []
  tl = len(charArray)-2
  for i in range(bit_number):
    if i < tl :
      dataArray.append(w_value*math.ceil(charArray[i+2])*(2^i))
    else :
      dataArray.append(0)

  return dataArray

def trans_a_dec_to_bin_array(a_dec,bit_number):
  a_data = bin(math.ceil(a_dec))
  charArray  = list( a_data ) 
  dataArray = []
  tl = len(charArray)-2
  for i in range(bit_number):
    if i < tl :
      dataArray.append(int(charArray[i+2]))
    else :
      dataArray.append(0)

  return dataArray

def trans_a_row_to_bin_value_array(a_row,bit_numbers):
  datas = []
  idx_data = {7,63,71,79,87,95,103,111,119,127,135,143,151,159,167,175}
  idx_one = {23,24,25,26,27,28,29,30,31,32,33,40,41,42,43,44,45,46,47,48,49,50}
  i = 0
  length = len(a_row)
  while i < length:
    if i in idx_data :
      datas.append(a_row[i])
      i += 1
      datas.append(a_row[i])
      i += 1
    elif i in idx_one :
      a_dec = a_row[i]
      dataArray = trans_a_dec_to_bin_array(a_dec,bit_numbers[i])
      datas.extend(dataArray)
      i += 1
    elif i == 0:
      datas.append(a_row[i])
      i += 1
    elif i == 176:
      i += 1
    else :
      datas.append(a_row[i])
      i += 1
      a_dec = a_row[i]
      dataArray = trans_a_dec_to_bin_array(a_dec,bit_numbers[i])
      datas.extend(dataArray)
      i += 1

  return datas

def trans_a_dataset_to_bin_value_array(dataset,bit_numbers):
  datas = []
  for i in range(len(dataset)):
    datas.append(trans_a_row_to_bin_value_array(dataset[i],bit_numbers))

  return datas

def main():

##  data = 25.3
##  charArray  =  list(bin(int(data)) ) 
##  print(charArray)
  
  in_file = os.path.join(FLAGS.input_data_dir, FLAGS.in_file)
  out_file = os.path.join(FLAGS.input_data_dir, FLAGS.out_file)

  data_sets = load_csv_datas_head_row(in_file,features_dtype=np.float32)

  datas = trans_a_dataset_to_bin_value_array(data_sets.data,data_sets.target)

  write_a_dataset_to_a_csv(out_file,datas)
  
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=300000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='/home/freebirdweij/tf_works/invest',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/home/freebirdweij/tf_works/invest/logs',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--in_file',
      type=str,
      default='input_train_datas.csv',
      help='Input file name.'
  )
  parser.add_argument(
      '--out_file',
      type=str,
      default='new_train_datas.csv',
      help='Output file name.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  main()
  #sys.exit(main=main, argv=[sys.argv[0]] + unparsed)


