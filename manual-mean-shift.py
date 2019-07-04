import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()

colors = 10*["g","r","c","b","k"]

class Mean_Shift:
    def __init__(self, radius=4):
        self.radius = radius
    
    
    def fit(self, data):
        centroids = {}
        # Every point in the dataset is a centroid
        for i in range(len(data)):
            centroids[i] = data[i]
        
        # Loop until centroids converge
        while True:
            new_centroids = []
            # Loop through centroids
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                # Loop through all points in the dataset
                for featureset in data:
                    # Check if point it is within the bandwith of the centroid
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwidth.append(featureset)
                # Calculating the new position of the centroid, it is the average point of all points within the bandwidth.
                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))
            # Eliminating duplicated centroids
            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            # Redefining the new centroids
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            
            for i in centroids:
                # Check if there centroids changed, if not, the centroids converged
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
                
            if optimized:
                break

        self.centroids = centroids


clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

plt.scatter(X[:,0], X[:,1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()