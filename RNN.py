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

#Reshaping the training set,Including the timeframe value
train_set=np.reshape(train_set,(1258,1,1))

#x_train -> Stock price of day x,used for predicting the stock price of day x+k,where k is the timeframe
#y_train -> Stock price of day x+k

x_train=train_set[0:1257]
y_train=train_set[1:1258]



