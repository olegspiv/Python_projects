# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:40:09 2020

@author: olegs
"""
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime 
from datetime import date  
import time
import datetime
from numpy import nan
import math 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

minimum = 1
n_features = 1
reps = 60

def split_sequence(sequence, n_steps_in, n_steps_out):    
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


# time = np.arange(0, 500, 0.1)
# # amplitude = np.sin(time)
# amplitude = np.sin(time)*time


# bid_train = amplitude[0:4000]
# bid_test = amplitude[4000::]
# orig_test = bid_test


time = np.arange(0, 100, 0.1)
# amplitude = np.sin(time)
amplitude = np.sin(time)*time


bid_train = amplitude[0:900]
bid_test = amplitude[900::]
orig_test = bid_test


kama = 40 

# choose a number of time steps
n_steps_in, n_steps_out = int(minimum*kama/2), int((minimum*kama/2)) # ten minutes, five minutes
# split into samples
X_train, y_train = split_sequence(bid_train, n_steps_in, n_steps_out)


x_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(X_train)
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))

regression = Sequential()
regression.add(LSTM(units = 100, activation = 'relu', input_shape = (X_train.shape[1], n_features)))
# regression.add(LSTM(units = 200, activation = 'relu')) # second layer
regression.add(Dense(n_steps_out)) # we say 7 because we are predicting 7 days into future
regression.compile(loss = 'mse', optimizer = 'adam')   
regression.fit(X_train, y_train, epochs = 100, batch_size = 32)


total_points = []
points = 0
##########################################################################
for cycles in range(0,reps):
  
      
  if 'bu' in locals():
    X_test, y_test = split_sequence(bu, n_steps_in, n_steps_out)
  else:
    X_test, y_test = split_sequence(bid_test, n_steps_in, n_steps_out)
      
  X_test = x_scaler.transform(X_test) 
  y_test = y_scaler.transform(y_test) 

  X_test = X_test.reshape(len(X_test), n_steps_in, n_features)
  X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))

  # demonstrate prediction
  y_pred = regression.predict(X_test)
  y_pred = y_scaler.inverse_transform(y_pred)
  y_test = y_scaler.inverse_transform(y_test)
      
  pred = y_pred[:, y_pred.shape[1]-1]
  test = y_test[:, y_test.shape[1]-1]

  plt.plot(pred)
  plt.plot(test)
  plt.show()

  alles = np.concatenate((test, pred[len(pred)-1]), axis=None)

  # create future points
  points = []
  points.append(pred[len(pred)-1])
  total_points.append(pred[len(pred)-1])
############################################################################

  while len(alles)>kama:

    X_test, y_test = split_sequence(alles, n_steps_in, n_steps_out)

    X_test = x_scaler.transform(X_test) 
    y_test = y_scaler.transform(y_test)
    X_test = X_test.reshape(len(X_test), n_steps_in, n_features)

    y_pred = regression.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_test = y_scaler.inverse_transform(y_test)
      
    pred = y_pred[:, y_pred.shape[1]-1]
    test = y_test[:, y_test.shape[1]-1]


    alles = np.concatenate((test, pred[len(pred)-1]), axis=None)

    points.append(pred[len(pred)-1])
    total_points.append(pred[len(pred)-1])
###########################################################################
  if 'bu' in locals():
    x =  np.arange(len(bu),len(bu) + len(points), 1)
    plt.plot(bu)
    plt.plot(x, points)
    plt.show()
    bu = np.concatenate((bu, points), axis=None)
  else:
    x =  np.arange(len(bid_test),len(bid_test) + len(points), 1)
    plt.plot(bid_test)
    plt.plot(x, points) 
    plt.show()     
    bu = np.concatenate((bid_test, points), axis=None)
###############################################################################

x =  np.arange(len(bid_test),len(bid_test) + len(total_points), 1)
plt.plot(bid_test, color = 'blue')
plt.plot(x,total_points, color = 'red')
plt.show() 

del bu
