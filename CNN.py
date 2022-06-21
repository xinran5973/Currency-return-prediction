import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# read the excel file
currency = pd.read_excel('C:/Users/lenovo/Desktop/EURUSD.xlsx') 

### use new column 'return' instead of columns 'Open' and 'Close'
# add new column 'return'
currency['return'] = np.log(currency['Close']) - np.log(currency['Open'])
# delete columns 'Open' and 'Close'
del currency['Open'], currency['Close']


# Now I consider to use the return of last 5 minutes to predict the return of next 1 minute
# Firstly, construct the features and label matrix
cur_r= currency[['Date','return']]

#split train set (use data before June 14 as train set)
window = 5
x_train, y_train = [], []

for i in range(757,4114-5):
    x_train.append(cur_r.loc[i + 1 : i + window, 'return'])
    y_train.append(cur_r.loc[i, 'return'])

#convert the list to ndarray
x_train = np.array(x_train)
y_train = np.array(y_train)

# split test set (use data in June 14 as test set)
x_test, y_test = [], []

for i in range(752):
    x_test.append(cur_r.loc[i + 1 : i + window, 'return'])
    y_test.append(cur_r.loc[i, 'return'])

#convert the list to ndarray
x_test = np.array(x_test)
y_test = np.array(y_test)

# convert x_train to 3D tensor
# x_train[data_index, time_point, channel]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# convert x_test to 3D tensor
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

### build a CNN
# 2 convolutional layers and 1 FNN hidden layer
model = tf.keras.models.Sequential()

# First coonvolutional layer
model.add(tf.keras.layers.Conv1D(filters = 4, kernel_size = 2, activation = 'relu', input_shape = (window,1)))# 1 channel only

# Second convolutional layer
model.add(tf.keras.layers.Conv1D(filters = 6, kernel_size = 2, activation = 'relu'))

# Global average pooling to collapse the matrix into a vector
model.add(tf.keras.layers.GlobalAveragePooling1D())

# FNN hidden layer with 8 neurons
model.add(tf.keras.layers.Dense(8, activation = 'relu'))

# Output layer
model.add(tf.keras.layers.Dense(1, activation = 'linear'))

### train the model
model.compile(loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 100, verbose = 1)

# evaluate the model quality (on test set)
cur_r_test = cur_r[:752]
y_pred = model.predict(x_test)
y_pred = y_pred.flatten()
cur_r_test.loc[:,'predicted r'] = y_pred
