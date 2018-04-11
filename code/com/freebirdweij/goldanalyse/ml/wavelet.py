from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pywt

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
        for j in range(seqLen-i-1) :
          row.append(tmp)
        row.extend(seqData[0:i+1])
        seqMat.append(row)
      else :
        row = []
        row.extend(seqData[i-seqLen+1:i+1])
        seqMat.append(row)
        
        
    return seqMat

#pywt.families()
#['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
"""
haar family: haar
db family: db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, db22, db23, db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35, db36, db37, db38
sym family: sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20
coif family: coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8, coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17
bior family: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6, bior2.8, bior3.1, bior3.3, bior3.5, bior3.7, bior3.9, bior4.4, bior5.5, bior6.8
rbio family: rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4, rbio2.6, rbio2.8, rbio3.1, rbio3.3, rbio3.5, rbio3.7, rbio3.9, rbio4.4, rbio5.5, rbio6.8
dmey family: dmey
gaus family: gaus1, gaus2, gaus3, gaus4, gaus5, gaus6, gaus7, gaus8
mexh family: mexh
morl family: morl
cgau family: cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8
shan family: shan
fbsp family: fbsp
cmor family: cmor
"""
#print(pywt.Modes.modes)
#['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect']
def dwt_single_level(seqMat,waveObj,waveMode):
    dwtMat = []
    for row in seqMat:
      dwtRow = []
      cA, cD = pywt.dwt(row, wavelet=waveObj, mode=waveMode)
      dwtRow.extend(cA)
      dwtRow.extend(cD)
      dwtMat.append(dwtRow)
    return dwtMat
  
def dwt_multi_level(seqMat,waveObj,waveMode,waveLevel):
    if waveLevel == 1 :
      return dwt_single_level(seqMat,waveObj,waveMode)
    elif waveLevel == 2 :
      dwtMat = []
      for row in seqMat:
        dwtRow = []
        cA2, cD2, cD1 = pywt.wavedec(row, waveObj, mode=waveMode, level=2)
        dwtRow.extend(cA2)
        dwtRow.extend(cD2)
        dwtRow.extend(cD1)
        dwtMat.append(dwtRow)
      return dwtMat
    elif waveLevel == 3 :
      dwtMat = []
      for row in seqMat:
        dwtRow = []
        cA3, cD3,cD2, cD1 = pywt.wavedec(row, waveObj, mode=waveMode, level=3)
        dwtRow.extend(cA3)
        dwtRow.extend(cD3)
        dwtRow.extend(cD2)
        dwtRow.extend(cD1)
        dwtMat.append(dwtRow)
      return dwtMat
    elif waveLevel == 4 :
      dwtMat = []
      for row in seqMat:
        dwtRow = []
        cA4, cD4,cD3,cD2, cD1 = pywt.wavedec(row, waveObj, mode=waveMode, level=4)
        dwtRow.extend(cA4)
        dwtRow.extend(cD4)
        dwtRow.extend(cD3)
        dwtRow.extend(cD2)
        dwtRow.extend(cD1)
        dwtMat.append(dwtRow)
      return dwtMat
    else :
      return 
    return
  
def main():
    
  DATA_INPUTS = 'audt365-2018-2-28-day.csv'

  input_datas = base.load_csv_without_header(DATA_INPUTS,target_dtype=np.float32,
                                  features_dtype=np.float32,target_column=0)
  
  dataMat = input_datas.data

  seqMat = construct_sequence_mat(dataMat,20)
  print('seqMat:-----------------------')
  print(seqMat)
  base.write_a_dataset_to_a_csv('audt365-2018-4-2-day-seq.csv', seqMat)

#  dwtMat = dwt_single_level(seqMat,'db2','symmetric')
  dwtMat = dwt_multi_level(seqMat,'db1','symmetric',4)
  base.write_a_dataset_to_a_csv('audt365-2018-4-2-day-dwt-ml4.csv', dwtMat)


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
