# Example Regression
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

# Selecting the features to be used by the regression classifier
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
forecast_out = int(math.ceil(0.01 * len(df)))

# Create the label collumn 
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())

# Features
X = np.array(df.drop(['label'], 1))
# Labels
y = np.array(df['label'])

X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
