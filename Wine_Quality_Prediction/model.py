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
from sklearn.preprocessing import LabelEncoder


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
#handle missing values
for column in df.columns:
	if df[column].isnull().sum() > 0:
		df[column] = df[column].fillna(df[column].median())
#check of missing vales have ben handled
print(f'Number of null values per column in the imputed  data: {df.isnull().sum()}')

#change the target column into  0,1 binary
df['Best_Quality'] = [1 if x > 5 else 0 for x in df.quality]

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
	sns.heatmap(data.select_dtypes(include=['number']).corr(), annot=True, fmt='.2f', cmap='coolwarm')
	plt.title('Correlation Heatmap')
	plt.savefig(os.path.join(output_dir, 'heatmap.png'))
	plt.close()
#plot a pairplot
	plt.figure(figsize=(15,10))
	sns.pairplot(data, hue='Best_Quality', palette='bright')
	plt.title('Pairplot')
	plt.savefig(os.path.join(output_dir, 'Pairplot.png'))
	plt.close()
#plt a countplot	
	plt.figure(figsize=(12,8))
	sns.countplot(x=data['Best_Quality'])
	plt.title('Countplot of the QUALITY column)')
	plt.ylabel('Frequency')
	plt.savefig(os.path.join(output_dir, 'quality_countplot.png'))
	plt.close()

output_directory = 'plots'

if __name__=='__main__':
	plots(df, output_directory)

#MODEL DEVELOPMENT
# Use label encoder to convert the categorical data to numerical data
lc=LabelEncoder()
df['type']=lc.fit_transform(df['type'])
print(df.head(10))

print(df.head(10))
print(f'Number of null values per column: {df.isnull().sum()}')

#spliting the data
x=df.drop(['quality', 'Best_Quality'], axis=1)
y=df['Best_Quality']
x_train,x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
print(f'x_train shape = {x_train.shape}\ny_train shape = {y_train.shape}\nx_test shape = {x_test.shape}\ny_test shape = {y_test.shape}')


#Normalize the data by scalling
scaling=MinMaxScaler()
x_train=scaling.fit_transform(x_train)
x_test=scaling.transform(x_test)

model=[LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
for i in range(3):
	model[i].fit(x_train,y_train)
	print(f'{model[i]}: ')
	print('Training accuracy: ',metrics.roc_auc_score(y_train, model[i].predict(x_train)))
	print('Validation accuracy: ', metrics.roc_auc_score(y_test, model[i].predict(x_test)))
	print()

#From the above accuracies we can say that Logistic Regression and SVC() classifier perform better on the validation data with less difference between the validation and training data.

# Let’s plot the confusion matrix,accuracy score, precision score and classification report for the validation data using the Logistic Regression model
print('Confusion matrix: \n' , metrics.confusion_matrix(y_test,model[1].predict(x_test)))
print('Accuracy score: \n' , metrics.accuracy_score(y_test,model[1].predict(x_test)))
print('Precision score: \n' , metrics.precision_score(y_test,model[1].predict(x_test)))
print('Classification Report: \n' , metrics.classification_report(y_test,model[1].predict(x_test)))
