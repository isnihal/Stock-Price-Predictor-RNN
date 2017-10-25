# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:59:48 2017

@author: nihal369
"""

#Data Preprocessing

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the training set
train_set=pd.read_csv("Google_Stock_Price_Train.csv")
#Splitting to opening price,2D import to fit the RNN
train_set=train_set.iloc[:,1:2].values

#Scaling the training set,Normalize for a value between 0 and 1
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
train_set=scaler.fit_transform(train_set)

#x_train -> Stock price of day x,used for predicting the stock price of day x+k,where k is the timeframe
#y_train -> Stock price of day x+k

x_train=train_set[0:1257]
y_train=train_set[1:1258]


#Reshaping the training set,Including the timeframe value
x_train=np.reshape(x_train,(1257,1,1))


#Neural Network

#Importing the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Initalizing the RNN
rnn=Sequential()

#Adding the LSTM RNN Input layer
#Input shape:timeframe and number of input nodes
rnn.add(LSTM(units=4,activation='sigmoid',input_shape=(1,1)))

#Adding the output layer
rnn.add(Dense(units=1))

#Compiling the RNN
rnn.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the RNN
rnn.fit(x_train,y_train,batch_size=16,epochs=200)


#Prediction

#Importing the test set
test_set=pd.read_csv("Google_Stock_Price_Test.csv")
#Splitting the opening price
real_stock_price=test_set.iloc[:,1:2].values


