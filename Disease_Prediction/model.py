#imporing the necessary libraries
import pandas as pd
import seaborn as sns
import numpy as no
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode

data = pd.read_csv('Disease_Prediction.csv')
print(data.head())
print(data.tail())
print(f'Shape of data: {data.shape}')
print('Null values\n', data.isnull().sum())
print('PROGNOSIS \n', data['prognosis'].value_counts())
#encoding the target value into numerical
encoder=LabelEncoder()
data['prognosis']=encoder.fit_transform(data['prognosis'])
print('Encoded data\n',data.head(7))
#split the data into training and testing
x=data.drop('prognosis',axis=1)
y=data['prognosis']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=24)
print(f'x_train: {x_train.shape}\ny_train: {y_train.shape}\nx_test: {x_test.shape}\ny_test: {y_test.shape}')