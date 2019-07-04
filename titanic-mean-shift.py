import numpy as np
from sklearn.cluster import MeanShift, KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd
import matplotlib.pyplot as plt


'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''


# https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
df = pd.read_excel('titanic.xls')

original_df = pd.DataFrame.copy(df)
df.drop(['body','name'], 1, inplace=True)
df.fillna(0,inplace=True)

def handle_non_numerical_data(df):
    
    # handling non-numerical data: must convert.
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        #print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            
            column_contents = df[column].values.tolist()
            #finding just the uniques
            unique_elements = set(column_contents)
            # great, found them. 
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x+=1
            # now we map the new "id" vlaue
            # to replace the string. 
            df[column] = list(map(convert_to_int,df[column]))

    return df

df = handle_non_numerical_data(df)
df.drop(['ticket','home.dest'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

#  Getting some attributes from our clf object, for example, how much cluster the algorithm found
labels = clf.labels_
cluster_centers = clf.cluster_centers_

# New column to our original dataframe and populate with empty values
original_df['cluster_group']=np.nan

# Iterate through the labels and populate the labels to the empty column
for i in range(len(X)):
    # With iloc we can access the index of the dataframe
    original_df['cluster_group'].iloc[i] = labels[i]


# Checking the survival rates for each of the groups we happen to find
n_clusters_ = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
    #print(temp_df.head())

    survival_cluster = temp_df[  (temp_df['survived'] == 1) ]

    survival_rate = len(survival_cluster) / len(temp_df)
    #print(i,survival_rate)
    survival_rates[i] = survival_rate
    
print(survival_rates)

# We realize that our algorithm classified the dataset of titanic into groups that reflects their class in the boat. With group zero with predominant people from second class, group one with predominant people of first class, and group two with third class passengers

# Print characteristics of passengers of group 1 of our mean shift algorithm
print(original_df[ (original_df['cluster_group']==0) ].describe())
# Print characteristics of passengers of group 2 of our mean shift algorithm
print(original_df[ (original_df['cluster_group']==1) ].describe())
# Print characteristics of passengers of group 3 of our mean shift algorithm
print(original_df[ (original_df['cluster_group']==2) ].describe())

# Checking the suvival rate of first class passengers within group 0
cluster_0 = (original_df[ (original_df['cluster_group']==0) ])
cluster_0_fc = (cluster_0[ (cluster_0['pclass']==1) ])
print(cluster_0_fc.describe())