import pandas as pd
import numpy as np
import math
import datetime
import pandas_datareader.data as web
from pandas.plotting import register_matplotlib_converters



# import charting

import matplotlib.pyplot as plt
%matplotlib inline 

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


start = datetime.datetime(2013, 10, 8)
end = datetime.datetime(2018, 10, 8)

df = web.DataReader('GOOG', 'yahoo',start,end)

df.head()
#Sorting date issue

df.reset_index(inplace = True,drop= False)


#Charting now

df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']


#Predict Stock Prices

dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

dfreg.fillna(value=-99999, inplace = True)

#Split 1% of data for forcasting
forecast_out = int(math.ceil(0.01 * len(dfreg)))

forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'],1))

X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(dfreg['label'])
y = y[:-forecast_out]


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

# Linear Regression
print('Dimension of X',X.shape)
print('Dimension of y',y.shape)



clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train,y_train)


# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)
    
# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)


KNeighborsRegressor(algorithm='auto', leaf_size=30,metric='minkowski',
                   metric_params=None,n_jobs=1,n_neighbors=2,p=2,
                   weights='uniform')

confidencereg = clfreg.score(X_test,y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test,y_test)

print("The linear regression confidence is ",confidencereg)
print("The quadratic regression 2 confidence is ",confidencepoly2)
print("The quadratic regression 3 confidence is ",confidencepoly3)
print("The knn regression confidence is ",confidenceknn)

forecast_set = clfknn.predict(X_lately)
dfreg['Forecast'] = np.nan
print(forecast_set, confidenceknn, forecast_out)


last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
  next_date = next_unix
  next_unix += datetime.timedelta(days=1)
  dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
                                                                       
                                                   
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


df.tail()


#Moving Average code
#data = df.sort_index(ascending=True, axis= 0)
#new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])


#for i in range(0,len(data)):
#  new_data['Date'][i] = data['Date'][i]
#  new_data['Close'][i] = data['Close'][i]
  

#train = new_data[:987]
#valid = new_data[987:]

#print(valid.shape)

#preds = []
#for i in range(0,valid.shape[0]):
#    a = train['Close'][len(train)-365+i:].sum() + sum(preds)
#    b = a/248
#    preds.append(b)


#plt.figure(figsize=(16,8))
#valid['Predictions'] = 0
#valid['Predictions'] = preds
#plt.plot(train['Close'])
#plt.plot(valid[['Close','Predictions']])


