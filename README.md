# Unsupervised Learning - Clustering algorithm Implementation on Glass Identification Dataset

Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/glass/

In this project, following clustering algorithms are implemented using Python sklearn library on the Glass identification dataset.
1. KMeans
2. Agglomerative Clustering and
3. DBSCAN

Data files are located inside the project ./data/ directory so no one has to download and provide data source. Source code is in Python and compatible with both Python 2.7.x and 3.x. Each script can be executed from the project directory.
Software requiremnt:
Python2.7.x or Python3.x with scikit-learn.
Scikit-learn requires:
1. Python (>= 2.6 or >= 3.3),
2. NumPy (>= 1.6.1),
3. SciPy (>= 0.9).

Uni-variate Analysis:
-----------------------
Execution:
python ./uni-variate-analysis.py

Output:
Histogram and Boxplot will be generated in ./uni-variate-analysis/

Bivariate Analysis:
-----------------------
Execution:
python ./correlation-matrix.py

Output:
Plot will be generated in ./correlation-matrix/

Execution of Clustering algorithms:
--------------------------------------
python3 kMeansClustering.py
python3 agglomerativeClustering.py
python3 dbscan.py

Output:
For each clustering algorithm following are displayed as output.
1. Clusters
2. Silhouette Coefficient
3. Homogenity
4. Completeness
5. V-measure
6. Adjusted Rand Index
7. Adjusted Mutual Information
8. Cluster Visualization

All Visualization plots have generated in ./clusters-output/ directory.
