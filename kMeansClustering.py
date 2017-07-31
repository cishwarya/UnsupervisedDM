# Python program for Data Clustering via KMeans Algorithm
# Data Mining Project - 2
# 
# 25th Jun, 2017
# @author Waqar Alamgir <wajrcs@gmail.com>
# @author Laridi Sofiane <sofyeeen@gmail.com>
# @author Ishwarya Chandrasekaran <cishwarya@gmail.com>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from time import time
from sklearn.cluster import KMeans

# Load data
input_file = "data/glass.data"
fig_dir = 'clusters-output/'

columnNames = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "GlassType"]
dataframe = pd.read_csv(input_file, names=columnNames)
sample_size=214
# End index is exclusive
X = np.array(dataframe.ix[:, 1:10])
y = np.array(dataframe['GlassType'])

n_samples, n_features = X.shape
n_labels = len(np.unique(y))

print("n_labels: %d, \t n_samples: %d, \t n_features: %d" % (n_labels, n_samples, n_features))

# Preprocessing

# PCA decomposition
pca = PCA() 
pca.fit(X)

exvar = pca.explained_variance_
print("Variance of all features after PCA decomposition", exvar)

plt.plot(exvar)
plt.savefig(fig_dir+"PCA_KMeans.png")

pca.n_components = 2
X_reduced = pca.fit_transform(X)
print("Data after PCA decomposition \n",X_reduced)

binarizer = Binarizer(threshold=1.1)
X_binarized = binarizer.transform(X)  

normalizer = Normalizer(norm='l1').fit(X)
X_normalized=normalizer.transform(X)  

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_featureSelected=sel.fit_transform(X)

print(79 * '_') #prints line

# Calculate the purity score for the given cluster assignments and ground truth classes
def purity_score(clusters, classes):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes
    
    :param clusters: the cluster assignments array
    :type clusters: numpy.array
    
    :param classes: the ground truth classes
    :type classes: numpy.array
    
    :returns: the purity score
    :rtype: float
    """
    
    A = np.c_[(clusters,classes)]

    n_accurate = 0.

    for j in np.unique(A[:,0]):
        z = A[A[:,0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]

# Performance calculation
def bench_k_means(estimator,data):
	t0 = time()
	estimator.fit(data)
	
	print ("Labels of instances after performing clustering:\n",estimator.labels_)
	print("Count of Labels",len(estimator.labels_))
	clusters = estimator.cluster_centers_
	print("Clusters:\n",clusters)
	cohesion = estimator.inertia_ / len(data)
	print("\nEvaluation Metrics:")
	print("Cohesion:  %.3f" %(cohesion))
	#print('% 9s' % 'init'
	print('time   inertia    homogenity   completeness  v-measure     ARI     AMI   silhouetteScore')
	print('%.2fs   %i          %.3f         %.3f     %.3f        %.3f    %.3f    %.3f'
		  % ((time() - t0), estimator.inertia_,
			 metrics.homogeneity_score(y, estimator.labels_),
			 metrics.completeness_score(y, estimator.labels_),
			 metrics.v_measure_score(y, estimator.labels_),
			 metrics.adjusted_rand_score(y, estimator.labels_),
			 metrics.adjusted_mutual_info_score(y,  estimator.labels_),
			 metrics.silhouette_score(data, estimator.labels_,
									  metric='euclidean',
									  sample_size=sample_size)))
	purityScore=purity_score(y,estimator.labels_)
	print("Purity score",purityScore)
	print("###############################################################################")


# Plot Visualization
def plotResults(kmeans,name,X_reduced):

	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
	y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Obtain labels for each point in mesh. Use last trained model.
	Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1)
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
	           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
	           cmap=plt.cm.Paired,
	           aspect='auto', origin='lower')

	plt.plot(X_reduced[:, 0], X_reduced[:, 1], 'k.', markersize=2)
	# Plot the centroids as a white X
	centroids = kmeans.cluster_centers_
	plt.scatter(centroids[:, 0], centroids[:, 1],
	            marker='x', s=169, linewidths=3,
	            color='w', zorder=10)
	plt.title('K-means clustering on Glass Identification dataset (PCA redcued data)\nCentroids are marked with white cross')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.savefig(fig_dir+ "KMeans_result.png")

reducedX = [X_reduced,X_normalized,X_binarized,X_featureSelected]
name =['PCA reduced','Normalized','Binarized','Low variance feature removed']
initClusterCenterList =['k-means++','random']
Nameindex=0

for X in reducedX:
	for initClusterCenter in initClusterCenterList:
		print("Perform Kmeans\nParameter settings and data used: ",initClusterCenter+","+name[Nameindex])
		kmeans = KMeans(n_clusters=n_labels,init=initClusterCenter)
		bench_k_means(kmeans,data=X)
	Nameindex=Nameindex+1

#Plot Kmeans with PCA reduced data

kmeans = KMeans(n_clusters=n_labels)
kmeans.fit(X_reduced)
plotResults(kmeans,'PCA reduced data',X_reduced)

