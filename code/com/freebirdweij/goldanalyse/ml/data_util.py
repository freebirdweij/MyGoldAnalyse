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
import datetime
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
  with open(filename,'w',newline='') as csvfile:
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

def str1_to_datetime(s_date):
  d = datetime.datetime.strptime(s_date, '%Y-%m-%d %H:%M')
  
  return d

def str2_to_datetime(s_date):
  d = datetime.datetime.strptime(s_date, '%Y/%m/%d-%H:%M')
  
  return d

def str3_to_datetime(s_date):
  d = datetime.datetime.strptime(s_date, '%Y-%m-%d')
  
  return d

def str4_to_datetime(s_date):
  d = datetime.datetime.strptime(s_date, '%Y/%m/%d')
  
  return d
'''
time_mode=1 : %Y-%m-%d %H:%M to %Y-%m-%d %H:%M
time_mode=2 : %Y-%m-%d %H:%M to %Y/%m/%d-%H:%M
time_mode=3 : %Y/%m/%d-%H:%M to %Y-%m-%d %H:%M
time_mode=4 : %Y/%m/%d-%H:%M to %Y/%m/%d-%H:%M
time_mode=5 : %Y-%m-%d to %Y-%m-%d
time_mode=6 : %Y-%m-%d to %Y/%m/%d
time_mode=7 : %Y/%m/%d to %Y-%m-%d
time_mode=8 : %Y/%m/%d to %Y/%m/%d
'''
def diff_two_datetimes(a_date,b_date,time_mode):
  if time_mode == 1 :
    diff = str1_to_datetime(a_date) - str1_to_datetime(b_date)
  elif time_mode == 2 :
    diff = str1_to_datetime(a_date) - str2_to_datetime(b_date)
  elif time_mode == 3 :
    diff = str2_to_datetime(a_date) - str1_to_datetime(b_date)
  elif time_mode == 4 :
    diff = str2_to_datetime(a_date) - str2_to_datetime(b_date)
  elif time_mode == 5 :
    diff = str3_to_datetime(a_date) - str3_to_datetime(b_date)
  elif time_mode == 6 :
    diff = str3_to_datetime(a_date) - str4_to_datetime(b_date)
  elif time_mode == 7 :
    diff = str4_to_datetime(a_date) - str3_to_datetime(b_date)
  elif time_mode == 8 :
    diff = str4_to_datetime(a_date) - str4_to_datetime(b_date)
  else :
    diff = str1_to_datetime(a_date) - str1_to_datetime(b_date)
    
  return diff


def compare_time_merge_datas(a_datas,b_datas,time_mode):
  c_datas = []
  a_date,b_date = a_datas.target,b_datas.target
  a_data,b_data = a_datas.data,b_datas.data
  ia,ib,ic = 0,0,0
  group = False
  while ia<len(a_data) and ib<len(b_data):
    diff = diff_two_datetimes(a_date[ia],b_date[ib],time_mode)
    if diff.days == 0 and diff.seconds == 0 :
      if group :
        ic += 1
        group = False
      c_row = []
      c_row.append(str(ic))
      c_row.append(a_date[ia])
      c_row.extend(a_data[ia])
      c_row.append(b_date[ib])
      c_row.extend(b_data[ib])
      c_datas.append(c_row)
      ia += 1
      ib += 1
    elif diff.days > 0  or (diff.days == 0 and diff.seconds > 0) :
      group = True
      ib += 1
    else :
      group = True
      ia += 1
      
  return c_datas

def compare_time_merge_datas2(a_datas,b_datas,time_mode):
  c_datas = []
  a_date,b_date = a_datas.target,b_datas.target
  a_data,b_data = a_datas.data,b_datas.data
  ia,ib,ic = 0,0,0
  group = False
  while ia<len(a_data) and ib<len(b_data):
    diff = diff_two_datetimes(a_date[ia],b_date[ib],time_mode)
    if (diff.days == 0 or diff.days == 1) and diff.seconds == 0 :
      if group :
        ic += 1
        group = False
      c_row = []
      c_row.append(str(ic))
      c_row.append(a_date[ia])
      c_row.extend(a_data[ia])
      c_row.append(b_date[ib])
      c_row.extend(b_data[ib])
      c_datas.append(c_row)
      ia += 1
      ib += 1
    elif diff.days > 1 or (diff.days > 0 and diff.seconds > 0) or (diff.days == 0 and diff.seconds > 0) or (diff.days == 1 and diff.seconds > 0) :
      group = True
      ib += 1
    else :
      group = True
      ia += 1
      
  return c_datas

def queue_time_merge_datas(a_datas,b_datas,time_mode):
  c_datas = []
  a_date,b_date = a_datas.target,b_datas.target
  a_data,b_data = a_datas.data,b_datas.data
  ia,ib,ic = 0,0,1
  group = False
  while ia<len(a_data) and ib<len(b_data):
    diff = diff_two_datetimes(a_date[ia],b_date[ib],time_mode)
    if diff.days == 0 and diff.seconds == 0 :
      if group :
        ic += 1
        group = False
      c_row = []
      c_row.append(str(0))
      c_row.append(a_date[ia])
      c_row.extend(a_data[ia])
      c_row.append(b_date[ib])
      c_row.extend(b_data[ib])
      c_datas.append(c_row)
      ia += 1
      ib += 1
    elif diff.days > 0  or (diff.days == 0 and diff.seconds > 0) :
      group = True
      c_row = []
      c_row.append(str(ic))
      if time_mode == 1 or time_mode == 4 or time_mode == 5 or time_mode == 8 :
        c_row.append(b_date[ib])
      elif time_mode == 2 :
        dt = str2_to_datetime(b_date[ib])
        c_row.append(dt.strftime('%Y-%m-%d %H:%M'))
      elif time_mode == 3 :
        dt = str1_to_datetime(b_date[ib])
        c_row.append(dt.strftime('%Y/%m/%d-%H:%M'))
      elif time_mode == 6 :
        dt = str4_to_datetime(b_date[ib])
        c_row.append(dt.strftime('%Y-%m-%d'))
      elif time_mode == 7 :
        dt = str3_to_datetime(b_date[ib])
        c_row.append(dt.strftime('%Y/%m/%d'))
      else :
        c_row.append(b_date[ib])
      c_row.extend(a_data[ia])
      c_row.append(b_date[ib])
      c_row.extend(b_data[ib])
      c_datas.append(c_row)
      ib += 1
    else :
      group = True
      c_row = []
      c_row.append(str(ic))
      c_row.append(a_date[ia])
      c_row.extend(a_data[ia])
      if time_mode == 1 or time_mode == 4 or time_mode == 5 or time_mode == 8 :
        c_row.append(a_date[ia])
      elif time_mode == 2 :
        dt = str1_to_datetime(a_date[ia])
        c_row.append(dt.strftime('%Y/%m/%d-%H:%M'))
      elif time_mode == 3 :
        dt = str2_to_datetime(a_date[ia])
        c_row.append(dt.strftime('%Y-%m-%d %H:%M'))
      elif time_mode == 6 :
        dt = str3_to_datetime(a_date[ia])
        c_row.append(dt.strftime('%Y/%m/%d'))
      elif time_mode == 7 :
        dt = str4_to_datetime(a_date[ia])
        c_row.append(dt.strftime('%Y-%m-%d'))
      else :
        c_row.append(a_date[ia])
      c_row.extend(b_data[ib])
      c_datas.append(c_row)
      ia += 1
      
  return c_datas

def query_sequence_chose_datas(a_datas,b_datas):
  c_datas = []
  a_date,b_date = a_datas.target,b_datas.target
  a_data,b_data = a_datas.data,b_datas.data
  for i in range(len(a_date)):
    c_row = []
    c_row.extend(b_data[a_date[i]-1])
    c_datas.append(c_row)
  
  return c_datas

def del_datas_by_index(in_datas):
  c_datas = []
  a_index = in_datas.target
  a_data = in_datas.data
  
  for i in range(len(a_index)):
    if a_index[i] == 0 :
      c_row = []
      c_row.extend(a_data[i])
      c_datas.append(c_row)
      
  return c_datas

def clear_dirty_datas_by_index(in_datas):
  c_datas = []
  a_index = in_datas.target
  a_data = in_datas.data
  d_index = 1
  for i in range(len(a_index)):
    if a_index[i] == d_index :
      d_index += 1
    else :
      c_row = []
      c_row.append(a_index[i])
      c_row.extend(a_data[i])
      c_datas.append(c_row)
  return c_datas

def make_train_datas_by_interval(in_datas,interval):
  c_datas = []
  a_date = in_datas.target
  a_data = in_datas.data
  for i in range(len(a_date)-interval):
      c_row = []
      diff = diff_two_datetimes(a_date[i+interval],a_date[i],1)
      minutes = diff.days*24*60 + diff.seconds/60      
      c_row.append(a_date[i+interval])
      c_row.append(a_data[i+interval][5])
      c_row.append(a_data[i+interval][0])
      c_row.append(int(minutes))
      c_row.append(a_date[i])
      c_row.append(a_data[i][0])
      c_row.append(a_data[i][5])
      c_row.append(a_data[i][9])
      c_row.append(a_data[i][11])
      c_row.append(a_data[i][12])
      c_row.append(a_data[i][13])
      c_row.append(a_data[i][14])
      c_datas.append(c_row)
      b_row = []
      b_row.append(a_date[i+interval])
      b_row.append(a_data[i+interval][6])
      b_row.append(a_data[i+interval][1])
      b_row.append(int(minutes))
      b_row.append(a_date[i])
      b_row.append(a_data[i][1])
      b_row.append(a_data[i][6])
      b_row.append(a_data[i][9])
      b_row.append(a_data[i][11])
      b_row.append(a_data[i][12])
      b_row.append(a_data[i][13])
      b_row.append(a_data[i][14])
      c_datas.append(b_row)
      e_row = []
      e_row.append(a_date[i+interval])
      e_row.append(a_data[i+interval][7])
      e_row.append(a_data[i+interval][2])
      e_row.append(int(minutes))
      e_row.append(a_date[i])
      e_row.append(a_data[i][2])
      e_row.append(a_data[i][7])
      e_row.append(a_data[i][9])
      e_row.append(a_data[i][11])
      e_row.append(a_data[i][12])
      e_row.append(a_data[i][13])
      e_row.append(a_data[i][14])
      c_datas.append(e_row)
      d_row = []
      d_row.append(a_date[i+interval])
      d_row.append(a_data[i+interval][8])
      d_row.append(a_data[i+interval][3])
      d_row.append(int(minutes))
      d_row.append(a_date[i])
      d_row.append(a_data[i][3])
      d_row.append(a_data[i][8])
      d_row.append(a_data[i][9])
      d_row.append(a_data[i][11])
      d_row.append(a_data[i][12])
      d_row.append(a_data[i][13])
      d_row.append(a_data[i][14])
      c_datas.append(d_row)
  return c_datas

def group_datas_by_time_interval(in_datas):
  c_datas = []
  a_intervl = in_datas.target
  a_data = in_datas.data
  dict1 = {}
  for i in range(len(a_intervl)):
    if a_intervl[i] in dict1 :
      a_row = []
      a_row.append(a_intervl[i])
      a_row.extend(a_data[i])
      dict1[a_intervl[i]].append(a_row)
    else :
      dict1[a_intervl[i]] = []
      a_row = []
      a_row.append(a_intervl[i])
      a_row.extend(a_data[i])
      dict1[a_intervl[i]].append(a_row)
  
  #dict1 = sorted(dict1.keys())   
  for key in sorted(dict1.keys()) :
    c_datas.extend(dict1[key])
    
  return c_datas

def make_day_datas_by_minute_datas(in_datas):
  c_datas = []
  a_date = in_datas.target
  a_data = in_datas.data
  startdate = a_date[0]
  op = a_data[0][0]
  hi = a_data[0][1]
  lo = a_data[0][2]
  cl = a_data[0][3]
  mo = 0
  for i in range(len(a_date)):
    diff = diff_two_datetimes(str1_to_datetime(a_date[i]).strftime("%Y-%m-%d"), str1_to_datetime(startdate).strftime("%Y-%m-%d"), 5)
    ed = str1_to_datetime(a_date[i])
    if diff.days >= 1 and ed.hour > 4 :
      c_row = []
      sd = str1_to_datetime(startdate)
      c_row.append(sd.strftime("%Y-%m-%d"))
      c_row.append(op)
      c_row.append(hi)
      c_row.append(lo)
      c_row.append(cl)
      c_row.append(mo)
      c_datas.append(c_row)
      startdate = a_date[i]
      op = a_data[i][0]
      hi = a_data[i][1]
      lo = a_data[i][2]
      cl = a_data[i][3]
      mo = a_data[i][4]
    if a_data[i][1] > hi :
      hi = a_data[i][1]
    if a_data[i][2] < lo :
      lo = a_data[i][2]
    cl = a_data[i][3]
    mo += a_data[i][4]
    
  return c_datas

def chose_datas_by_time(in_datas,the_time):
  c_datas = []
  a_date = in_datas.target
  a_data = in_datas.data
  c_row = []
  
  if the_time == 1 :
    t_dex = 0
    n_add = False
    for i in range(len(a_date)):
      if a_date[i] == t_dex :
        if n_add :
          a_row = []
          a_row.append(a_date[i-1])
          a_row.extend(a_data[i-1])
          c_datas.append(a_row)
      else :
        n_add = False
        t_dex = a_date[i]
        sd = str1_to_datetime(a_data[i][0])
        if sd.hour >= 9 and sd.hour <= 11 :
          n_add = True    
  elif the_time == 2 :
    t_dex = 0
    n_add = False
    for i in range(len(a_date)):
      if a_date[i] == t_dex :
        if n_add :
          a_row = []
          a_row.append(a_date[i-1])
          a_row.extend(a_data[i-1])
          c_datas.append(a_row)
      else :
        n_add = False
        t_dex = a_date[i]
        sd = str1_to_datetime(a_data[i][0])
        if sd.hour >= 13 and sd.hour <= 15 :
          n_add = True    
  elif the_time == 3 :
    t_dex = 0
    n_add = False
    for i in range(len(a_date)):
      if a_date[i] == t_dex :
        if n_add :
          a_row = []
          a_row.append(a_date[i-1])
          a_row.extend(a_data[i-1])
          c_datas.append(a_row)
      else :
        n_add = False
        t_dex = a_date[i]
        sd = str1_to_datetime(a_data[i][0])
        if sd.hour >= 20 and sd.hour <= 23 :
          n_add = True    
    
  return c_datas

def main():

##  data = 25.3
##  charArray  =  list(bin(int(data)) ) 
##  print(charArray)
  a_in = '365-hjxh-2018-7-11-check-office-test.csv'
  #a_data = load_csv_without_header(a_in,target_dtype=np.str,features_dtype=np.str,target_column=0)
  #c_datas = make_train_datas_by_interval(a_data,7)
  #c_out = '365-hjxh-2018-7-11-check-office-test2.csv'
  #write_a_dataset_to_a_csv(c_out,c_datas)
 
  #a_in = 'pufa-tdx-hjxh-2018-7-16-minute-5-merge-office.csv'
  #a_in = '365-hjxh-2018-7-11-check-office-test.csv'
  #a_in = '365-hjxh-2018-7-11-check-office-test2.csv'
  b_in = 'tdx-365-2018-7-11-check-office-test.csv'
  c_out = '365-hjxh-2018-8-20-high-low-real-office.csv'

  a_data = load_csv_without_header(a_in,target_dtype=np.int16,features_dtype=np.str,target_column=0)
  #a_data = load_csv_without_header(a_in,target_dtype=np.str,features_dtype=np.str,target_column=0)
  #a_data = load_csv_without_header(a_in,target_dtype=np.str,features_dtype=np.float32,target_column=0)
  b_data = load_csv_without_header(b_in,target_dtype=np.int16,features_dtype=np.str,target_column=0)
  #c_datas = queue_time_merge_datas(a_data,b_data,2)
  #c_datas = compare_time_merge_datas(a_data,b_data,1)
  #c_datas = clear_dirty_datas_by_index(a_data)
  #c_datas = make_train_datas_by_interval(a_data,4)
  #c_datas = group_datas_by_time_interval(a_data)
  #c_datas = make_day_datas_by_minute_datas(a_data)
  c_datas = query_sequence_chose_datas(a_data,b_data)
  

  print('c_datas:-----------------------')
  #print(c_datas)
  
  write_a_dataset_to_a_csv(c_out,c_datas)
  
  #for i in range(len(indexMat)):
  #  d = str2_to_datetime(indexMat[i])
  #  print(d)
 #in_file = os.path.join(FLAGS.input_data_dir, FLAGS.in_file)
  #out_file = os.path.join(FLAGS.input_data_dir, FLAGS.out_file)

  #data_sets = load_csv_datas_head_row(in_file,features_dtype=np.float32)

  #datas = trans_a_dataset_to_bin_value_array(data_sets.data,data_sets.target)

  #write_a_dataset_to_a_csv(out_file,datas)
  
  
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


