# Python program for Data Clustering via KMeans Algorithm
# Data Mining Project - 2
# 
# 25th Jun, 2017
# @author Waqar Alamgir <wajrcs@gmail.com>
# @author Laridi Sofiane <sofyeeen@gmail.com>
# @author Ishwarya Chandrasekaran <cishwarya@gmail.com>

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from time import time
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import scale

# Load data
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
plt.savefig(fig_dir+"PCA_DBSCAN.png")

pca.n_components = 2
X_reduced = pca.fit_transform(X)
print("Data after PCA decomposition \n",X_reduced)

binarizer = Binarizer(threshold=1.1)
X_binarized = binarizer.transform(X)  

normalizer = Normalizer(norm='l1').fit(X)
X_normalized=normalizer.transform(X)  

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_featureSelected=sel.fit_transform(X)

X_stdScaled = StandardScaler().fit_transform(X)

X_robustScaled = RobustScaler().fit_transform(X)

X_scaled = scale(X)

print(79 * '_') #prints line

def dbscan(eps,minSamples,X,name):
  print("Perform DBSCAN Clustering\nParameter settings and data used:",name)
  db = DBSCAN(eps=eps, min_samples=minSamples).fit(X)
  labels = db.labels_
  evaluationMeasure(labels,X)
  #print("Predicted labels",labels)

def purity_score(y_true, y_pred):
    """Purity score

    To compute purity, each cluster is assigned to the class which is most frequent 
    in the cluster [1], and then the accuracy of this assignment is measured by counting 
    the number of correctly assigned documents and dividing by the number of documents.abs

    Args:
        y_true(np.ndarray): n*1 matrix Ground truth labels
        y_pred(np.ndarray): n*1 matrix Predicted clusters
    
    Returns:
        float: Purity score
    
    References:
        [1] https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    """
    # matrix which will hold the majority-voted labels
    y_labeled_voted = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_labeled_voted[y_pred==cluster] = winner
    
    return accuracy_score(y_true, y_labeled_voted)

def evaluationMeasure(labels,X):
  # Number of clusters in labels, ignoring noise if present.
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  #print('Estimated number of clusters: %d' % n_clusters_)
  print("\nEvaluation Metrics:")
  print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
  print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
  print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
  print("Adjusted Rand Index: %0.3f"
        % metrics.adjusted_rand_score(y, labels))
  print("Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(y, labels))
  print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, labels))
  purityScore=purity_score(y,labels)
  print("Purity score",purityScore)
  print("###############################################################################")


def plot(X,labels):
  # Black removed and is used for noise instead.
  unique_labels = set(labels)
  colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
  for k, col in zip(unique_labels, colors):
      if k == -1:
          # Black used for noise.
          col = [0, 0, 0, 1]

      class_member_mask = (labels == k)

      xy = X[class_member_mask & core_samples_mask]
      plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
               markeredgecolor='k', markersize=14)

      xy = X[class_member_mask & ~core_samples_mask]
      plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
               markeredgecolor='k', markersize=6)
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  plt.title('Estimated number of clusters: %d' % n_clusters_)
  plt.savefig(fig_dir+'DBSCAN_output.png')

reducedX = [X_stdScaled,X_robustScaled,X_scaled,X_reduced]
epsList = [1.07, 1.08]
minSamplesList= [7, 8]
name =['Std scaled','X_robustScaled','X_scaled','PCA reduced']
nameIndex=0

for X in reducedX:
  for eps in epsList:
    for minSamples in minSamplesList:
      dbscan(eps,minSamples,X,str(eps)+','+str(minSamples)+','+name[nameIndex])
  nameIndex=nameIndex+1

# Plot results for robustScaled data
db = DBSCAN(eps=1.07, min_samples=8).fit(X_robustScaled)
labels = db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
plot(X_robustScaled,labels)
