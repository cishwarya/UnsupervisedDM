# Python program for Data Clustering via Agglomerative Algorithm
# Data Mining Project - 2
# 
# 25th Jun, 2017
# @author Waqar Alamgir <wajrcs@gmail.com>
# @author Laridi Sofiane <sofyeeen@gmail.com>
# @author Ishwarya Chandrasekaran <cishwarya@gmail.com>

from sklearn.neighbors import kneighbors_graph
import time as time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score
from time import time

#Load data
input_file = "data/glass.data"
fig_dir = 'clusters-output/'

columnNames = ["ID", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "GlassType"]
dataframe = pd.read_csv(input_file, names=columnNames)
sample_size=214

# End index is exclusive
X = np.array(dataframe.ix[:, 1:10])
y = np.array(dataframe['GlassType'])

# Preprocessing

# PCA decomposition
pca = PCA() 
pca.fit(X)

exvar =pca.explained_variance_
print("Variance of all features after PCA decomposition", exvar)

plt.plot(exvar)
plt.savefig(fig_dir+"PCA_Agglomerative.png")

pca.n_components = 2
X_reduced = pca.fit_transform(X)
print("Data after PCA decomposition \n",X_reduced)

binarizer = Binarizer(threshold=1.1)
X_binarized = binarizer.transform(X)  #.961 with kmeans

normalizer = Normalizer(norm='l1').fit(X)
X_normalized=normalizer.transform(X)  #.980 with kmeasn pca -highest

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_featureSelected=sel.fit_transform(X)

print("Compute structured hierarchical clustering")
st = time()

# Algo for agglomerative clustering
def agglomerative(X,linkageType, affinityType, name):
	print("Perform Agglomerative Clustering\nParameter settings and data used",name,"\n")
	ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity,
	                               linkage=linkageType,affinity=affinityType).fit(X)
	elapsed_time = time() - st
	label = ward.labels_
	print("Elapsed time: %.2fs" % elapsed_time)
	print("Labels:\n",label)
	print("Number of points: %i" % label.size)
	return label
	print(79 * '_') 

# Metrics

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
def evaluate_cluster(data,agglomerative_labels):
	print("\nEvaluation Metrics:")
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, agglomerative_labels))
	print("Completeness: %0.3f" % metrics.completeness_score(y, agglomerative_labels))
	print("V-measure: %0.3f" % metrics.v_measure_score(y, agglomerative_labels))
	print("Adjusted Rand Index: %0.3f"
	      % metrics.adjusted_rand_score(y, agglomerative_labels))
	print("Adjusted Mutual Information: %0.3f"
	      % metrics.adjusted_mutual_info_score(y, agglomerative_labels))
	print("Silhouette Coefficient: %0.3f"
	      % metrics.silhouette_score(data, agglomerative_labels))
	purityScore=purity_score(y,agglomerative_labels)
	print("Purity score",purityScore)
	print("###############################################################################")

# Plot Visualization
def plotResult(label):
	fig = plt.figure()
	ax = p3.Axes3D(fig)
	ax.view_init(7, -80)
	for l in np.unique(label):
	    ax.plot3D(X_featureSelected[label == l, 0], X_featureSelected[label == l, 1], X_featureSelected[label == l, 2],
	              'o', color=plt.cm.jet(float(l) / np.max(label + 1)))
	#plt.title('With connectivity constraints (time %.2fs)' % elapsed_time)
	plt.title('Agglomerative clustering')
	plt.savefig(fig_dir+"Agglomerative_results.png")

linkageList = ['complete','ward','average']
affinityList = ['euclidean','manhattan']
reducedX = [X_reduced,X_normalized,X_binarized,X_featureSelected]
name =['PCA reduced','Normalized','Binarized','Low variance feature removed']
Nameindex=0

# Agglomerative function called below code
for X in reducedX:
	for linkage in linkageList:
		for affinity in affinityList:
			connectivity = kneighbors_graph(X, n_neighbors=10) # include_self=False
			if linkage == 'ward':
				affinity = 'euclidean'
				agglomerativeLabels = agglomerative(X,linkage,affinity,linkage+','+affinity+','+name[Nameindex])
				evaluate_cluster(X,agglomerativeLabels)
			else:
				agglomerativeLabels = agglomerative(X,linkage,affinity,linkage+','+affinity+','+name[Nameindex])
				evaluate_cluster(X,agglomerativeLabels)
	Nameindex=Nameindex+1

# Plot Results - Settings and data used: Average, Euclidean & normalized
agglomerativeLabels = agglomerative(X_normalized,'average','euclidean','average, euclidean & normalized')
plotResult(agglomerativeLabels)




