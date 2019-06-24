# Example Regression
import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
df = quandl.get('WIKI/GOOGL')

# Selecting the features to be used by the regression classifier (not the best features)
# Volume refers to how many trades occured that day, related to volatility
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# From the feature selected create new valuables insights.
# Adj. High - Adj. Close, can tell us the percentage volatility. 
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100 

# Daily percentage change =  new minus the old divided by the old times a 100 
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100 


# Defining the features that we will care about and will be used for our classifier. 
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Define the forecast collumn 
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# We are trying to predict the next 10% of prices
forecast_out = int(math.ceil(0.1 * len(df)))

# Create the label collumn 
df['label'] = df[forecast_col].shift(-forecast_out)

# Features
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
# Slice array from forecast_out until the end, this will get the last 10% of numbers in the array, is the data that we are going to use to predict
X_lately = X[-forecast_out:]
# Slice array from 0 until forecast_out, more simple, the 90% of the array used to train the model
X = X[:-forecast_out]


df.dropna(inplace=True)
# Labels
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

# Create a new collumn and filling with non-values
df['Forecast'] = np.nan

# Creating date collumn to show in the graph 
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# Loop through the objects
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # Explanation about df.loc[next_date], the dates are the index of the data. So when we say this we are reference the data of this date. if any doubt we can print the df.head and see. For the precited values you can print df.tail()
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
