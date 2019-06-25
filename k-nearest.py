import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors
import pandas as pd

# Reading database as csv file
df = pd.read_csv('breast-cancer-wisconsin.data')
# Replacing missing values for a really big negative number, this will create outlayers, that our algorithm will recognize and will handle in a proper way
df.replace('?',-99999, inplace=True)
# Droping Id collumn, since this does not have any valuable information
df.drop(['id'], 1, inplace=True)

# Storing the features values inside of X
X = np.array(df.drop(['class'], 1))
# Storing the classes values inside of y
y = np.array(df['class'])

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
# Measuring accuracy of the classifier
accuracy = clf.score(X_test, y_test)
print(accuracy)

# Creating new instances to predict
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
# Reshaping np array into a format accepted by scikitlearn
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)