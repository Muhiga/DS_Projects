import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

#load the dataset
df=pd.read_csv('HousingData.csv')

#check for mising values
print(df.isnull().sum())

#drop the null values
df.dropna(inplace=True)

print(df.head())
print(df.columns)
print(df.dtypes)
print(df.describe())
print(df.info())
print(df.duplicated().sum())

#EDA
# Define the directory where the plots will be saved
eda_plots = os.path.join(os.getcwd(), 'eda_plots')

# Create the directory if it doesn't exist
if not os.path.exists(eda_plots):
    os.makedirs(eda_plots)

for column in df.columns:
	plt.figure(figsize=(12,8))
	sns.histplot(df[column], kde=True, hue=None)
	plt.title(f'Distribution plot for {column}')
	plt.xlabel(column)
	plt.ylabel('Frequency')
	plt.savefig(os.path.join(eda_plots, f'{column}_distplot.png'))
	plt.close()

	plt.figure(figsize=(12,8))
	sns.boxplot(df[column],showfliers=True)
	plt.title(f'Boxplot For {column}')
	plt.xlabel(column)
	plt.savefig(os.path.join(eda_plots, f'{column}_boxplot.png'))
	plt.close()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Map')
plt.savefig(os.path.join(eda_plots, 'heatmap.png'))
plt.close()

plt.figure()
sns.pairplot(df, hue=None)
plt.savefig(os.path.join(eda_plots,'pairplot.png'))
plt.close()


#feature selection
x=df.drop('MEDV',axis=1)
y=df['MEDV']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
print(f'Training set shape: {x_train.shape}')
print(f'Test set shape: {x_test.shape}')
print(f'Y Train set shape: {y_train.shape}')
print(f'Y Test set shape: {y_test.shape}')

#model building
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#Lets print the first five values of the y_pred
print(f'First 5 predicted values: {y_pred[:5]}') 

#model evaluation
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print(f'The Mean squared error: {mse}')
print(f'The Mean absolute error: {mae}')
print(f'The root mean squared error: {rmse}')
print(f'The r squared: {r2}')
