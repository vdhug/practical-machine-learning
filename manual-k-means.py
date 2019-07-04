import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])

colors = 10*["g","r","c","b","k"]
plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter


    def fit(self,data):

        self.centroids = {}
        # Initializing the centroids to be the first k values, for example, if the dataset is X = [ [1,2], [4,5], [8,6], [4,4] ] and k (the number of centroids) is equal to two, this will set the centroids to be [1,2] and [4,5] initially
        for i in range(self.k):
            self.centroids[i] = data[i]
        
        # Initializing optimization
        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                # Calculating distances of dataset points to the centroids
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                # Measuring for what centroid the point belongs
                classification = distances.index(min(distances))
                # Appending the point to the centroid
                self.classifications[classification].append(featureset)
            
            # Storing old centroids to measure the percentual of change between the new centroid and the older one
            prev_centroids = dict(self.centroids)

            # Redefine the new centroids
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                # Check if the centroids moved more than the tolerance
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    # Print the percentage change
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False
            
            if optimized:
                break
    

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

# Testing k-means algorithm
clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

# Testing algorithm with unknown data
unknowns = np.array([[1,3],
                    [8,9],
                    [0,3],
                    [5,4],
                    [6,4],])

for unknown in unknowns:
   classification = clf.predict(unknown)
   plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)
  
plt.show()