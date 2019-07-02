import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')

#ORIGINAL:

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])


# Plotting dataset 
# plt.scatter(X[:, 0],X[:, 1], s=150, linewidths = 5, zorder = 10)
# plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)

# Gettings coordinates from centroids
centroids = clf.cluster_centers_
# Getting the labels that the cluster algorithm created
labels = clf.labels_

colors = ["g.","r."]
# Graphing dataset and centroids
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
plt.show()