'''
Long short-term memory (LSTM)
is an artificial neural network
used in the fields of artificial intelligence
and deep learning.
Unlike standard feedforward neural networks,
LSTM has feedback connections.

Samuel Norman "Sam" Seaborn is a fictional character
portrayed by Rob Lowe on the television serial drama
The West Wing

https://www.kaggle.com/code/mahmoud87hassan/predict-future-crude-oil-prices-using-lstm-network
https://rahmadya.com/2021/03/04/belajar-recurrent-neural-network/
https://www.geeksforgeeks.org/introduction-to-convolutions-using-python/

Artificial Neural Network

Convolution Neural Network is a modification of
Recurrent Neural Network

https://www.nickmccullum.com/python-deep-learning/recurrent-neural-network-tutorial/
'''

import numpy as np #linear algebra
#data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import datetime
from pylab import rcParams
import matplotlib.pyplot as plt
import warnings
import itertools
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from keras.callbacks \
	import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import math
from sklearn.preprocessing import MinMaxScaler
#Input data files are available in the "../input/" directory.
#running this (by clicking run or pressing Shift+Enter)
#will list all files under the input directory
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Any results you write to the current directory
#are saved as output.
#Using TensorFlow backend.
#/kaggle/input/brent-oil-prices/BrentOilPrices.csv

#Convert date coulmns to specific format
dateparse = lambda x: pd.datetime.strptime(x, '%b %d, %Y')
#Read csv file
'''
df = pd.read_csv(r'/home/haga/Downloads/BrentOilPrices.csv', \
    sep = '-', parse_dates=['Date'], date_parser = dateparse)
'''
df = pd.read_csv(r'/home/haga/Downloads/BrentOilPrices.csv', \
    parse_dates=['Date'])
#Sort dataset by column Date
df = df.sort_values('Date')
df = df.groupby('Date')['Price'].sum().reset_index()
df.set_index('Date', inplace=True)
df=df.loc[datetime.date(year=2000,month=1,day=1):]

# Print some data rows.
df.head()

#Read dataframe info
def DfInfo(df_initial):
    # gives some infos on columns types and numer of null values
    tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0: 'null values (nb)'}))
    tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum() / df_initial.shape[0] * 100).T.
                               rename(index={0: 'null values (%)'}))
    return tab_info

DfInfo(df)

df.index

y = df['Price'].resample('MS').mean()

y.plot(figsize=(15, 6))
plt.show()

'''
matplotlib.rcParams contains some properties in
matplotlibrc file.
We can use it to control the defaults of almost every property
in Matplotlib: figure size and DPI, line width,
color and style, axes, axis and grid properties,
text and font properties and so on.

In order to use matplotlib.rcParams,
we should know what properties are stored in it.
These properties can be foud in matplotlibrc file.
import matplotlib
f = matplotlib.matplotlib_fname()
print(f)
C:\Users\fly165\.conda\envs\py3.6\lib\site-packages\matplotlib\mpl-data\matplotlibrc
'''
rcParams['figure.figsize'] = 18, 8

'''
sm = statsmodels
tsa = time series analysis -> autoregression models

Decomposition can be defined as the process of
solving a complex problem
and breaking it into more sub-problems
that can be solved easily.
Solving a complex problem may get difficult sometimes
but finding the solution for every sub-problem
will be simple after which the sub-problems
can be put together for finding the full solution
to the original problem.
It is a divide-and-conquer technique -> merge sort algorithm.

In statistics, an additive model (AM)
is a nonparametric regression method.

Nonparametric regression is a category of regression analysis
in which the predictor does not take a predetermined form
but is constructed according to information derived
from the data.
That is, no parametric form is assumed
for the relationship between predictors and dependent variable.
Nonparametric regression requires larger sample sizes
than regression based on parametric models
because the data must supply the model structure
as well as the model estimates.
'''
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

'''
Transform features by scaling each feature to a given range.

This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.
'''
# normalize the data_set 
sc = MinMaxScaler(feature_range = (0, 1))
df = sc.fit_transform(df)

# split into train and test sets
train_size = int(len(df) * 0.70)
test_size = len(df) - train_size
train, test = df[0:train_size, :], df[train_size:len(df), :]

# convert an array of values into a data_set matrix def
def create_data_set(_data_set, _look_back=1):
    data_x, data_y = [], []
    for i in range(len(_data_set) - _look_back - 1):
        a = _data_set[i:(i + _look_back), 0]
        data_x.append(a)
        data_y.append(_data_set[i + _look_back, 0])
    return np.array(data_x), np.array(data_y)

# reshape into X=t and Y=t+1
look_back =90
X_train,Y_train,X_test,Ytest = [],[],[],[]
X_train,Y_train=create_data_set(train,look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test,Y_test=create_data_set(test,look_back)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# create and fit the LSTM network regressor = Sequential() 
regressor = Sequential()

regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units = 60))
regressor.add(Dropout(0.1))

regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5)
history =regressor.fit(X_train, Y_train, epochs = 20, batch_size = 15,validation_data=(X_test, Y_test), callbacks=[reduce_lr],shuffle=False)

train_predict = regressor.predict(X_train)
test_predict = regressor.predict(X_test)

# invert predictions
train_predict = sc.inverse_transform(train_predict)
Y_train = sc.inverse_transform([Y_train])
test_predict = sc.inverse_transform(test_predict)
Y_test = sc.inverse_transform([Y_test])

print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();

#Compare Actual vs. Prediction
aa=[x for x in range(180)]
plt.figure(figsize=(8,4))
plt.plot(aa, Y_test[0][:180], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:180], 'r', label="prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();
