# Python program for Correlation Matrix
# Data Mining Project - 2
# 
# 25th Jun, 2017
# @author Waqar Alamgir <wajrcs@gmail.com>
# @author Laridi Sofiane <sofyeeen@gmail.com>
# @author Ishwarya Chandrasekaran <cishwarya@gmail.com>

# Importing packages
import numpy as np
import pandas as pd
import os
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Directories settings
input_file = 'data/glass.data'
project_dir = ''

# Defining global variables here
labels = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

# Comma delimited is the default
range_data = [i for i in range(1,10)]
df = pd.read_csv(input_file, header=0, usecols=range_data)
df_target = pd.read_csv(input_file, header=0, usecols=[10])

# Remove the non-numeric columns
df_data = df._get_numeric_data()

# Plots correlation matrix
def plot_correlation_matrix(cm, title='Correlation Matrix', labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(df_data.corr())
    plt.title(title)
    fig.colorbar(cax)
    if labels:
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)

    plt.savefig(project_dir+'correlation-matrix/confusion_matrix.png')
    plt.show()

# Plot the Correlation Matrix
plot_correlation_matrix(df_data, 'Correlation Matrix', labels)