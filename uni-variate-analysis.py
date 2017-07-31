# Python program for Uni-variat Analysis
# Data Mining Project - 2
# 
# 25th Jun, 2017
# @author Waqar Alamgir <wajrcs@gmail.com>
# @author Laridi Sofiane <sofyeeen@gmail.com>
# @author Ishwarya Chandrasekaran <cishwarya@gmail.com>

# Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as pl
import matplotlib
import numpy as np

# Directories settings
input_file = 'data/glass.data'
project_dir = ''

# Defining global variables here
labels = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type of glass']
labelsWithInfo = [
	['RI', 0.001, 30, 1.5100],
	['Na', 0.5, 20, 10],
	['Mg', 0.5, 10, 0],
	['Al', 0.5, 10, 0],
	['Si', 0.5, 15, 65],
	['K', 0.5, 15, 0],
	['Ca', 0.5, 25, 0],
	['Ba', 0.5, 10, 0],
	['Fe', 0.1, 6, 0],
	['Type of glass', 0, 0, 0]
]
# dataset = pd.read_csv(input_file, sep=',', names=labels)
range_data = [i for i in range(1,11)]
dataset = pd.read_csv(input_file, sep=',', names=labels, usecols=range_data)

# This methods generates histogram
def histogram(label, project_dir, dataset, bin_width, num_bins, range_value):
	dataset.hist(column=label, figsize=(7,7), color="blue", bins=num_bins, range=(range_value, range_value+(bin_width*num_bins)))
	plt.xlabel(label)
	plt.ylabel('Value')
	plt.savefig(project_dir+'uni-variate-analysis/hist_'+label+'.png')

# Looping through all lables and generate histograms
for v in labelsWithInfo :
	if (v[0] != 'Type of glass'):
		histogram(v[0], project_dir, dataset, v[1], v[2], v[3])

# Generating boxplot here
dataset.plot(kind='box', figsize=(12, 8))
plt.savefig(project_dir+'uni-variate-analysis/boxplot.png')
