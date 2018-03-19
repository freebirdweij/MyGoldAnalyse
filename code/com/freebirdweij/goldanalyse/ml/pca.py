from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time
import numpy as np


from numpy import shape
from scipy import linalg

import com.freebirdweij.goldanalyse.ml.data_util as base

#���ֵ��  
def zeroMean(dataMat):        
    meanVal=np.mean(dataMat,axis=0)     #�������ֵ��������������ľ�ֵ  
    newData=dataMat-meanVal  
    return newData,meanVal  
 
def percentage2n(eigVals,percentage):  
    sortArray=np.sort(eigVals)   #����  
    sortArray=sortArray[-1::-1]  #��ת��������  
    arraySum=sum(sortArray)  
    tmpSum=0  
    num=0  
    for i in sortArray:  
        tmpSum+=i  
        num+=1  
        if tmpSum>=arraySum*percentage:  
            return num
        
         
def pca(dataMat,percentage=0.99):  
    newData,meanVal=zeroMean(dataMat)  
    covMat=np.cov(newData,rowvar=0)    #��Э�������,return ndarray����rowvar��0��һ�д���һ��������Ϊ0��һ�д���һ������  
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#������ֵ����������,���������ǰ��зŵģ���һ�д���һ����������  
    n=percentage2n(eigVals,percentage)                 #Ҫ�ﵽpercent�ķ���ٷֱȣ���Ҫǰn����������  
    eigValIndice=np.argsort(eigVals)            #������ֵ��С��������
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #����n������ֵ���±�  
    n_eigVect=eigVects[:,n_eigValIndice]        #����n������ֵ��Ӧ����������  
    lowDDataMat=newData*n_eigVect               #��ά�����ռ������  
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #�ع�����  
    return lowDDataMat,reconMat 

def pca_code(dataMat,percentage=0.99):
    [n,d] = shape(dataMat)
    mean_v = dataMat.mean(axis=0)
    dataMat -= mean_v
    XTX = np.dot(dataMat.T,dataMat)
    [eig_v,eig_vect] = linalg.eigh(XTX)
    
    for i in range(d):
        eig_vect[:,i] = eig_vect[:i]/linalg.norm(eig_vect[:i])
        
    idx = np.argsort(-eig_v)
    eig_v = eig_v[idx]
    eig_vect = eig_vect[:idx]
    k=percentage2n(eig_v,percentage)                 #Ҫ�ﵽpercent�ķ���ٷֱȣ���Ҫǰk����������  
    eig_v = eig_v[0:k].copy()
    eig_vect = eig_vect[:,0:k].copy()
       
    return k,eig_vect

def main():
    
  DATA_INPUTS = 'audt365-2018-2-28-day.csv'

  input_datas = base.load_csv_without_header(DATA_INPUTS,target_dtype=np.float32,
                                  features_dtype=np.float32,target_column=0)
  
  dataMat = input_datas.data
  print('dataMat:-----------------------')
  print(dataMat)

  k,eig_vect = pca_code(dataMat)
  pcaData = np.dot(dataMat,eig_vect)
  print('k:-----------------------')
  print(k)
  print('pcaData:-----------------------')
  print(pcaData)



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
