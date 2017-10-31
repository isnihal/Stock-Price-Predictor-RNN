#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

#PART 1:Data Preprocessing


#Importing the training set & Splitting to opening price,2D import to fit the RNN

#**********GOOGLE**********
google_train_set=pd.read_csv("Dataset/Google_Stock_Price_Train.csv")
google_train_set=google_train_set.iloc[:,1:2].values


#Scaling the training set,Normalize for a value between 0 and 1
google_scaler=MinMaxScaler()
google_train_set=google_scaler.fit_transform(google_train_set)

#x_train -> Stock price of day x,used for predicting the stock price of day x+k,where k is the timeframe
#y_train -> Stock price of day x+k

google_x_train=google_train_set[0:1257]
google_y_train=google_train_set[1:1258]


#Reshaping the training set,Including the timeframe value
google_x_train=np.reshape(google_x_train,(1257,1,1))

#**********APPLE**********
apple_train_set=pd.read_csv("Dataset/Apple_Stock_Price_Train.csv")
apple_train_set=apple_train_set.iloc[:,1:2]

apple_scaler=MinMaxScaler()
apple_train_set=apple_scaler.fit_transform(apple_train_set)

apple_x_train=apple_train_set[0:1257]
apple_y_train=apple_train_set[1:1258]

apple_x_train=np.reshape(apple_x_train,(1257,1,1))

#**********AMAZON**********
amazon_train_set=pd.read_csv("Dataset/Amazon_Stock_Price_Train.csv")
amazon_train_set=amazon_train_set.iloc[:,1:2].values

amazon_scaler=MinMaxScaler()
amazon_train_set=amazon_scaler.fit_transform(amazon_train_set)

amazon_x_train=amazon_train_set[0:1257]
amazon_y_train=amazon_train_set[1:1258]

amazon_x_train=np.reshape(amazon_x_train,(1257,1,1))

#**********MICROSOFT**********
microsoft_train_set=pd.read_csv("Dataset/Microsoft_Stock_Price_Train.csv")
microsoft_train_set=microsoft_train_set.iloc[:,1:2].values

microsoft_scaler=MinMaxScaler()
microsoft_train_set=microsoft_scaler.fit_transform(microsoft_train_set)

microsoft_x_train=microsoft_train_set[0:1257]
microsoft_y_train=microsoft_train_set[1:1258]

microsoft_x_train=np.reshape(microsoft_x_train,(1257,1,1))


#**********TESLA**********
tesla_train_set=pd.read_csv("Dataset/Tesla_Stock_Price_Train.csv")
tesla_train_set=tesla_train_set.iloc[:,1:2].values

tesla_scaler=MinMaxScaler()
tesla_train_set=tesla_scaler.fit_transform(tesla_train_set)

tesla_x_train=tesla_train_set[0:1257]
tesla_y_train=tesla_train_set[1:1258]

tesla_x_train=np.reshape(tesla_x_train,(1257,1,1))


#PART 2:Neural Network

#**********GOOGLE**********

#Initalizing the RNN
google_rnn=Sequential()

#Adding the LSTM RNN Input layer
#Input shape:timeframe and number of input nodes
google_rnn.add(LSTM(units=4,activation='sigmoid',input_shape=(1,1)))

#Adding the output layer
google_rnn.add(Dense(units=1))

#Compiling the RNN
google_rnn.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the RNN
google_rnn.fit(google_x_train,google_y_train,batch_size=16,epochs=200)

#**********APPLE**********
apple_rnn=Sequential()
apple_rnn.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
apple_rnn.add(Dense(units=1))
apple_rnn.compile(optimizer='adam',loss='mean_squared_error')
apple_rnn.fit(apple_x_train,apple_y_train,batch_size=16,epochs=200)


#**********AMAZON**********
amazon_rnn=Sequential()
amazon_rnn.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
amazon_rnn.add(Dense(units=1))
amazon_rnn.compile(optimizer='adam',loss='mean_squared_error')
amazon_rnn.fit(amazon_x_train,amazon_y_train,batch_size=16,epochs=200)


#**********MICROSOFT**********
microsoft_rnn=Sequential()
microsoft_rnn.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
microsoft_rnn.add(Dense(units=1))
microsoft_rnn.compile(optimizer='adam',loss='mean_squared_error')
microsoft_rnn.fit(microsoft_x_train,microsoft_y_train,batch_size=16,epochs=200)


#**********TESLA**********
tesla_rnn=Sequential()
tesla_rnn.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
tesla_rnn.add(Dense(units=1))
tesla_rnn.compile(optimizer='adam',loss='mean_squared_error')
tesla_rnn.fit(tesla_x_train,tesla_y_train,batch_size=16,epochs=200)


#PART 3:Prediction


#Importing the test set
google_test_set=pd.read_csv("Dataset/Google_Stock_Price_Test.csv")
#Splitting the opening price
google_real_stock_price=google_test_set.iloc[:,1:2].values

#Scaling the real prices,int the same format of training set

#**********GOOGLE**********
google_inputs=google_real_stock_price
google_inputs=google_scaler.transform(google_inputs)
google_inputs=np.reshape(google_inputs,(20,1,1))

#Prediciting the stock prices for first 20 days of January 2017
google_predicted_stock_price=google_rnn.predict(google_inputs)
#Inverse scaling the predicted prices
google_predicted_stock_price=google_scaler.inverse_transform(google_predicted_stock_price)

#**********APPLE**********
apple_test_set=pd.read_csv("Dataset/Apple_Stock_Price_Test.csv")
apple_real_stock_price=apple_test_set.iloc[:,1:2].values

apple_inputs=apple_real_stock_price
apple_inputs=apple_scaler.transform(apple_inputs)
apple_inputs=np.reshape(apple_inputs,(19,1,1))

apple_predicted_stock_price=apple_rnn.predict(apple_inputs)
apple_predicted_stock_price=apple_scaler.inverse_transform(apple_predicted_stock_price)


#**********AMAZON**********
amazon_test_set=pd.read_csv("Dataset/Amazon_Stock_Price_Test.csv")
amazon_real_stock_price=amazon_test_set.iloc[:,1:2].values

amazon_inputs=amazon_real_stock_price
amazon_inputs=amazon_scaler.transform(amazon_inputs)
amazon_inputs=np.reshape(amazon_inputs,(19,1,1))

amazon_predicted_stock_price=amazon_rnn.predict(amazon_inputs)
amazon_predicted_stock_price=amazon_scaler.inverse_transform(amazon_predicted_stock_price)

#**********MICROSOFT**********
microsoft_test_set=pd.read_csv("Dataset/Microsoft_Stock_Price_Test.csv")
microsoft_real_stock_price=microsoft_test_set.iloc[:,1:2].values

microsoft_inputs=microsoft_real_stock_price
microsoft_inputs=microsoft_scaler.transform(microsoft_inputs)
microsoft_inputs=np.reshape(microsoft_inputs,(19,1,1))

microsoft_predicted_stock_price=microsoft_rnn.predict(microsoft_inputs)
microsoft_predicted_stock_price=microsoft_scaler.inverse_transform(microsoft_predicted_stock_price)


#**********TESLA**********
tesla_test_set=pd.read_csv("Dataset/Tesla_Stock_Price_Test.csv")
tesla_real_stock_price=tesla_test_set.iloc[:,1:2].values

tesla_inputs=tesla_real_stock_price
tesla_inputs=tesla_scaler.transform(tesla_inputs)
tesla_inputs=np.reshape(tesla_inputs,(19,1,1))

tesla_predicted_stock_price=tesla_rnn.predict(tesla_inputs)
tesla_predicted_stock_price=tesla_scaler.inverse_transform(tesla_predicted_stock_price)


#PART 4:Plotting

#**********GOOGLE**********
#Initalizing the frame
google_figure=plt.figure()

#Plotting the values
plt.plot(google_real_stock_price,color='red',label='Real Price')
plt.plot(google_predicted_stock_price,color='blue',label='Predicted Price')

#Axis names
plt.xlabel("Days")
plt.ylabel("Stock Price($)")

#Graph name
plt.title("Google Stock Price Prediction (Jan 2017)")

#Show legend
plt.legend()

#Show the graph
plt.show()

#**********APPLE**********
apple_figure=plt.figure()
plt.plot(apple_real_stock_price,color='red',label='Real Price')
plt.plot(apple_predicted_stock_price,color='blue',label='Predicted Price')
plt.xlabel("Days")
plt.ylabel("Stock Price($)")
plt.title("Apple Stock Price Prediction (Jan 2017)")
plt.legend()
plt.show()

#**********AMAZON**********
amazon_figure=plt.figure()
plt.plot(amazon_real_stock_price,color='red',label='Real Price')
plt.plot(amazon_predicted_stock_price,color='blue',label='Predicted Price')
plt.xlabel("Days")
plt.ylabel("Stock Price($)")
plt.title("Amazon Stock Price Prediction (Jan 2017)")
plt.legend()
plt.show()

#**********MICROSOFT**********
microsoft_figure=plt.figure()
plt.plot(microsoft_real_stock_price,color='red',label='Real Price')
plt.plot(microsoft_predicted_stock_price,color='blue',label='Predicted Price')
plt.xlabel("Days")
plt.ylabel("Stock Price($)")
plt.title("Microsoft Stock Price Prediction (Jan 2017)")
plt.legend()
plt.show()


#**********TESLA**********
tesla_figure=plt.figure()
plt.plot(tesla_real_stock_price,color='red',label='Real Price')
plt.plot(tesla_predicted_stock_price,color='blue',label='Predicted Price')
plt.xlabel("Days")
plt.ylabel("Stock Price($)")
plt.title("Tesla Stock Price Prediction (Jan 2017)")
plt.legend()
plt.show()

###############################################################################

