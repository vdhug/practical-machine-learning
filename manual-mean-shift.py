import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

'''
1. Start at every datapoint as a cluster center

2. take mean of radius around cluster, setting that as new cluster center

3. Repeat #2 until convergence.

'''

X, y = make_blobs(n_samples=15, centers=3, n_features=2)



# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8 ],
#               [8, 8],
#               [1, 0.6],
#               [9,11],
#               [8,2],
#               [10,2],
#               [9,3],])

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()

colors = 10*["g","r","c","b","k"]

class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step = 100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step


    def fit(self, data):

        if self.radius == None:
            # Finding a decente overall radius to have and steps
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        centroids = {}
        # Every point in the dataset is a centroid
        for i in range(len(data)):
            centroids[i] = data[i]
        
        # Defining weights and reversing 
        weights = [i for i in range(self.radius_norm_step)][::-1]  
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
                    # if np.linalg.norm(featureset-centroid) < self.radius:
                    #     in_bandwidth.append(featureset)
                    distance = np.linalg.norm(featureset-centroid)
                    # Adding distance for the first iteration where the centroid it is the point itself
                    if distance == 0:
                        distance = 0.00000000001
                    
                    # Defining weight 
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1

                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bandwidth +=to_add
                # Calculating the new position of the centroid, it is the average point of all points within the bandwidth.
                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))
            # Eliminating duplicated centroids
            uniques = sorted(list(set(new_centroids)))

            to_pop = []

            for i in uniques:
                for ii in [i for i in uniques]:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        #print(np.array(i), np.array(ii))
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

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

        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []
            
        for featureset in data:
            #compare distance to either centroid
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            #print(distances)
            classification = (distances.index(min(distances)))

            # featureset that belongs to that cluster
            self.classifications[classification].append(featureset)


    def predict(self, data):
        #compare distance to either centroid
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification


clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1], marker = "x", color=color, s=150, linewidths = 5, zorder = 10)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()