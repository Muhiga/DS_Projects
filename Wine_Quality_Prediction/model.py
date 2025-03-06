import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('winequalityN.csv')

#Basic Checks
print(f'The shape of the data: is {df.shape}')
print(f'Columns in the data: {df.columns}')
print(df.head())
print(df.tail())
# Lets find the number of non null values in each column, the dtypes and the  memory usage 
print(df.info())
# Summary statistics for the the numerical columns in the dataset
print(df.describe().T)
#check for the missing values
print(f'Number of null values per column: {df.isnull().sum()}')

#EXPLORATORY DATA ANALYSIS
def plots(data, output_dir, hue=None):
	 # Create directories for plots
	distplot_dir = os.path.join(output_dir, "distplots")
	boxplot_dir = os.path.join(output_dir, "boxplots")

	os.makedirs(distplot_dir, exist_ok=True)
	os.makedirs(boxplot_dir, exist_ok=True)
#we create distplot and boxplots for each columns
	for column in data.columns:
		if pd.api.types.is_numeric_dtype(data[column]):
			plt.figure(figsize=(12,8))
			sns.histplot(data[column], kde=True)
			plt.title(f'Distribution plot for {column}')
			plt.xlabel(column)
			plt.ylabel('Frequency')
			plt.savefig(os.path.join(distplot_dir, f'{column}_distplot.png'))
			plt.close()

			plt.figure(figsize=(12,8))
			sns.boxplot(data=data, x=column, showfliers=True)
			plt.title(f'Boxplot plot for {column}')
			plt.xlabel(column)
			plt.savefig(os.path.join(boxplot_dir, f'{column}_boxplot.png'))
			plt.close()
#we create a correlation heatmap
	if 'Unnamed: 0' in data.columns:
		data=data.drop('Unnamed: 0', axis=1)
		plt.figure(figsize=(12,8))
		sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
		plt.title('Correlation Heatmap')
		plt.savefig(os.path.join(output_dir, 'heatmap.png'))
		plt.close()
#plot a pairplot
	plt.figure(figsize=(15,10))
	sns.pairplot(data, hue='quality')
	plt.title('Pairplot')
	plt.savefig(os.path.join(output_dir, 'Pairplot.png'))
	plt.close()
#plt a countplot	
	plt.figure(figsize=(12,8))
	sns.countplot(x=data['quality'])
	plt.title('Countplot of the Target Column (Outcome)')
	plt.ylabel('Frequency')
	plt.savefig(os.path.join(output_dir, 'quality_countplot.png'))
	plt.close()

output_directory = 'plots'

if __name__=='__main__':
	plots(df, output_directory)

