# Example Regression
import pandas as pd
import quandl

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

print(df.head())