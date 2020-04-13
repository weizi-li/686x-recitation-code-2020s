import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

### prepare the data
x, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
# plt.scatter(x[:,0], x[:,1])
# plt.show()

### determine the optimal K
dis_all = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='random', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    dis_all.append(kmeans.inertia_) #sum of squared distances of samples to their closest cluster center.
plt.plot(range(1, 20), dis_all)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.show()

### use the optimal K for clustering the data
kmeans = KMeans(n_clusters=4, max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(x)
c1 = np.where(pred_y == 0)[0]
c2 = np.where(pred_y == 1)[0]
c3 = np.where(pred_y == 2)[0]
c4 = np.where(pred_y == 3)[0]
plt.scatter(x[c1,0], x[c1,1], c='green')
plt.scatter(x[c2,0], x[c2,1], c='blue')
plt.scatter(x[c3,0], x[c3,1], c='orange')
plt.scatter(x[c4,0], x[c4,1], c='cyan')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='red')
plt.show()
