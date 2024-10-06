
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data  # used to scrap data from yahoofinanace website
from keras.models import load_model
import streamlit as st

import yfinance as yf
# what anaylst do : if 100 daysma is above 200daysma then there is an  starting of uptrend else starting point of downtreand

start = '2010-01-01'
end = '2024-04-30'
st.title('stock Trend Prediction')
user_input= st.text_input('enter stock Ticker','AAPL')

df = yf.download(user_input, start, end)
# describing data
st.subheader('Data from 2010- 2024')
st.write(df.describe())
# visulaisations
st.subheader('Closing Price vs Time chart')
fig= plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100= df.Close.rolling(100).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100,'r', label='Time chart with 100MA')
plt.plot(df.Close,'g',label ='Closing Price')
st.pyplot(fig)
st.subheader('Closing Price vs Time chart with  100MA &200MA')
ma100= df.Close.rolling(100).mean()
ma200= df.Close.rolling(200).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='100MA')
plt.plot(ma200,'b',label='200MA')

plt.plot(df.Close,'g',label='Closing Price')
st.pyplot(fig)

# spliting our data into training and testing [70-30 ratio] for data prediction
data_training= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])   # strating from 0 index i have to 705 of the total values
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
#scaling down data between 0 and 1 for that to provide data to lstm model
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array= scaler.fit_transform(data_training)


#load my model
model= load_model('keras_model.h5')

# feeding data into the model, testing part
past_100_days= data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)
# making prediction
y_predicted= model.predict(x_test)
scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted= y_predicted* scale_factor
y_test= y_test*scale_factor
# final graph
st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label ='predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)

