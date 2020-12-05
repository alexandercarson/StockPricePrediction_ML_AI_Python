import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# get quote
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-12-04')
# show the data
df

# get the count of rows and columns in data set
df.shape

# visualize the closing price history
plt.figure(figsize=(16, 8))
plt.title('Closing Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# new df w/only close column
data = df.filter(['Close'])

# convert df to a numpy array
ds = data.values

# get or compute the # of rows to train LSTM model
training_data_length = math.ceil(len(ds) * .8)

training_data_length



# scale the data (its good practice)
scaler = MinMaxScaler(feature_range=(0, 1))
scale_data = scaler.fit_transform(ds)
# scaler.fit_transform = computes min/max values to be used for sclaing and transforms data based on the values
# range 0-1 inclusive (could be 0 -> could be 1)
scale_data


# Create the training dataset
# create the scaled training dataset
# scale_data[0:training_data_length] where 0 is the index
train_data = scale_data[0:training_data_length, :]
# split the data into X_train and y_train datasets
X_train = []  # independent training features
y_train = []  # dependent or target  variable

for i in range(60, len(train_data)):
    X_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(X_train)
        print(y_train)
        print()


# convert x_train and y_train to numpy arrays
# to use for training lstm model
X_train, y_train = np.array(X_train), np.array(y_train)


# reshape the x_train dataset
# lstm expects 3d and x_train is only 2d right now

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# compile the model
# epochs # of iterations in deep learning
model.compile(optimizer='adam', loss='mean_squared_error')


# train the model
model.fit(X_train, y_train, batch_size=1, epochs=3)


# create the testing dataset
# create a new array containing scaled values from index 1543 to end
# this will be the scaled testing dataset
test_data = scale_data[training_data_length - 60:, :]

# create the datasets x_test and y_test
X_test = []
y_test = ds[training_data_length:, :]  # contains 60 first values not scaled

# create test set
for i in range(60, len(test_data)):
    X_test.append(test_data[i - 60:i, 0])


# convert data into a numpy array
X_test = np.array(X_test)


# reshape the data, we need it 3d
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# get model predicted prive values
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# evaluate model = RMSE get the root mean squared error RMSE(how accurate model predicts res, lower indicates better fit)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse


# plot the data
training_Data = data[0:training_data_length]
validation = data[training_data_length:]
validation['Predictions'] = predictions

# visualize the data
plt.figure(figsize=(18, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Training', 'Validation', 'Predictions'], loc='lower right')
plt.show()


# show the valid or actual prices and predicted prices (compared)
validation


# Finally, lets try and predict the Apple stock price for a given (date)
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-12-03')

# create new df
new_df = apple_quote.filter(['Close'])

# get last 60 day closing price values and convert df to array
last_60_days = new_df[-60:].values

# scale between 0 & 1
last_60_days_scaled = scaler.transform(last_60_days)

# create an empty list
X_test = []

# append last_60_days_scaled to test
X_test.append(last_60_days_scaled)

# convert _X_test to numpy array
X_test = np.array(X_test)  # to use in lstm model

# reshape to 3d
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = model.predict(X_test)

# undo scaling
predicted_price = scaler.inverse_transform(predicted_price)

# print price
# The prediction is for one day after the above(apple_quote) end date
print(predicted_price)

# In[210]:


# Check actual price (if available)
apple_quote_2 = web.DataReader('AAPL', data_source='yahoo', start='2020-12-04', end='2020-12-04')
print(apple_quote_2['Close'])

# In[ ]:
