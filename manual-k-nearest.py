# Manual k-nearest algorithm, based on Euclidean distance
import numpy as np
import warnings
from math import sqrt
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            # Only works in features that are bi-dimensional
            # euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )
            # The same as the line above, but wrote in a different way, using numpy
            # euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            # Simplified version of getting euclidean distance using numpy function 
            # Calculating euclidean distances from each point of group to the point that we want to predict
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            # Append the distance in the list of distances, that way in the end we can sort the array of distances, get the first 3 and check which class most repited.
            distances.append([euclidean_distance, group])
    
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence


df = pd.read_csv("breast-cancer-wisconsin.data")
# Replacing missing values for a really big negative number, this will create outlayers, that our algorithm will recognize and will handle in a proper way
df.replace('?',-99999, inplace=True)
# Droping Id collumn, since this does not have any valuable information
df.drop(['id'], 1, inplace=True)
# Making sure that all dataset are numbers
full_data = df.astype(float).values.tolist()

random.shuffle(full_data)
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
# Selecting the first 80% of data to train
train_data = full_data[:-int(test_size*len(full_data))]
# Selecting the last 20% of data to test
test_data = full_data[-int(test_size*len(full_data)):]


for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        else:
            print(confidence)
        total += 1
print('Accuracy:', correct/total)