'''
Created on 2018-9-8

@author: weij
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import numpy as np


from numpy import shape
from scipy import linalg
from sklearn import datasets,linear_model,cross_validation,svm

import com.freebirdweij.goldanalyse.ml.data_util as base

def test_linearSVC(*data):
  X_train,X_test,y_train,y_test = data
  cls = svm.LinearSVC()
  cls.fit(X_train, y_train)
  print('Coefficients:%s,Intercept:%s'%(cls.coef_,cls.intercept_))
  print('Scors:%.2f'%cls.score(X_test, y_test))
  
def test_SVC(*data):
  X_train,X_test,y_train,y_test = data
  cls = svm.SVC(kernel='poly')
  cls.fit(X_train, y_train)
  print('Coefficients:%s,Intercept:%s'%(cls.coef_,cls.intercept_))
  print('Scors:%.2f'%cls.score(X_test, y_test))
  
def main():
    
  DATA_INPUTS = 'hjxh365-2018-4-16-day-plus-check1-symmetry2-pre-pca.csv'

  input_datas = base.load_csv_without_header(DATA_INPUTS,target_dtype=np.float32,
                                  features_dtype=np.float32,target_column=0)
  
  dataMat = input_datas.data
  print('dataMat:-----------------------')
  print(dataMat)

  pcaData = np.dot(dataMat,eig_vect)
  #reconMat = np.dot(pcaData,eig_vect.T)+mean_v  #Reconstructed datas.
  print('k:-----------------------')
  print(k)
  print('pcaData:-----------------------')
  print(pcaData)
  print('reconMat:-----------------------')
  #print(reconMat)
  base.write_a_dataset_to_a_csv('hjxh365-2018-4-16-day-plus-check1-symmetry2-pca99.csv', pcaData)
  #base.write_a_dataset_to_a_csv('hjxh365-2018-4-16-day-plus-norm-clear-pca9999-recn.csv', reconMat)



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
      default=0.99,
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
