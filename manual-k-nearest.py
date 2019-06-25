# Manual k-nearest algorithm, based on Euclidean distance
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1],s=100,color=i)
# Rewriting in one line
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]

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
    return vote_result

result = k_nearest_neighbors(dataset, new_features)
print(result)