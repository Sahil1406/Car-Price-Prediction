# IMPORTING ALL THE DEPENDENCIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression    
from sklearn import metrics 


# IMPORTING THE DATA 

dataset = pd.read_csv('car data.csv')

# print(dataset.head()) --> Prints first 5 rows of the dataset
# print(dataset.shape) --> Prints the no. rows and column
# print(dataset.describe()) --> Prints the mathematical values of dataset
# print(dataset.isnull().sum()) --> Prints how many missing values are there in the dataset

# CHECKING THE CATEGORICAL DATA

# print(dataset.Fuel_Type.value_counts())
# print(dataset.Seller_Type.value_counts())
# print(dataset.Transmission.value_counts())


# ENCODING THE CATEGORICAL DATA

dataset.replace({'Fuel_Type': {'Petrol':0, 'Diesel':1, 'CNG':2}, 'Seller_Type': {'Dealer':0, 'Individual':1}, 'Transmission': {'Manual':0, 'Automatic':1}}, inplace=True) # --> Label Encoding again


# print(dataset.head())

# SPLIITING THE DATA FROM THE LABELS AND UNNECESARRY COLUMNS

X = dataset.drop(['Car_Name', 'Selling_Price'], axis=1) 
Y = dataset['Selling_Price']


# SPLITTING THE DATA INTO TRAINING AND TEST DATA

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# TRAINING THE MODEL

# linear regression model

lin_reg = LinearRegression()

lin_reg.fit(X_train, Y_train)


# EVALUATION

print("Linear Regression")

training_data_prediction = lin_reg.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R Square Error of Training Data: ", error_score)

testing_data_prediction = lin_reg.predict(X_test)
error_score = metrics.r2_score(Y_test, testing_data_prediction)
print("R Square Error of Testing Data: ", error_score)


# lasso regression

las_reg = Lasso()
las_reg.fit(X_train, Y_train)

# EVALUATION

print("Lasso Regression")

training_data_prediction = las_reg.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R Square Error of Training Data: ", error_score)

testing_data_prediction = las_reg.predict(X_test)
error_score = metrics.r2_score(Y_test, testing_data_prediction)
print("R Square Error of Testing Data: ", error_score)