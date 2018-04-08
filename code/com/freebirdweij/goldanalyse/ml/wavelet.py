from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np


from numpy import shape
from scipy import linalg

import com.freebirdweij.goldanalyse.ml.data_util as base

#Zero mean method 
def zeroMean(dataMat):        
    meanVal=np.mean(dataMat,axis=0)     #To get means of every columns. i.e. means of every features.  
    newData=dataMat-meanVal  
    return newData,meanVal  
 
        
#Get normalize datas        
def mean_norm(dataMat):
    [n,d] = shape(dataMat)
    print('n===:%d----------d===:%d-----------'%(n, d))
    mean_v = dataMat.mean(axis=0)
    dataMat -= mean_v
    for i in range(d):
        dataMat[:,i] = dataMat[:,i]/linalg.norm(dataMat[:,i])     
    return dataMat

def construct_sequence_mat(seqData,seqLen):
    [n,d] = shape(seqData)
    print('n===:%d----------d===:%d-----------'%(n, d))
    if d > 1 :
      seqData = seqData[:,0]
    seqMat = []
    for i in range(n):
      if i < seqLen :
        tmp = seqData[0]
        row = []
        for j in seqLen-i-1 :
          row.append(tmp)
        row.append(seqData[0:i+1])
        seqMat.append(row)
      else :
        row = []
        row.append(seqData[i-seqLen+1:i+1])
        seqMat.append(row)
        
    return seqMat
  
def wavelet_decompose(seqMat):
    
    return
def main():
    
  DATA_INPUTS = 'audt365-2018-2-28-day.csv'

  input_datas = base.load_csv_without_header(DATA_INPUTS,target_dtype=np.float32,
                                  features_dtype=np.float32,target_column=0)
  
  dataMat = input_datas.data
  print('dataMat:-----------------------')
  print(dataMat)




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
      default=100000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--percentage',
      type=float,
      default=0.9,
      help='Number of float for pca remain percentage.'
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
      default=1,
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
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  main()
