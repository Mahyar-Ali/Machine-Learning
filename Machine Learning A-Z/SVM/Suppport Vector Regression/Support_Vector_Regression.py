#python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:31:40 2020

@author: mahyar
"""
#import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Feature Scaling because it is required for SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1)).flatten() #It is because y is a vector.
# Flatten method is used to convert the results back into an array.


#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#Predicting a new result and accounting for the Scaling
x= sc_X.transform(np.array([[6.5]]))
y_pred = regressor.predict(x)
y_pred = sc_y.inverse_transform(y_pred)

#Visualising the Results
plt.scatter(X,y,color='yellow')
plt.plot(X,regressor.predict(X),color='orange')
plt.title("Level vs Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()