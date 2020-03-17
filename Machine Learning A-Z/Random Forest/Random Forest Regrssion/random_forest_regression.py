#python3 Random Forest Regression
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:43:59 2020

@author: mahyar
"""
#import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2]
y = dataset.iloc[:,2]

#Fitting the dataset to the Random Forset Regression 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state=0) #n_estimators = no of trees
regressor.fit(X,y)

#Predicting new results
y_pred = regressor.predict(np.array([[6.5]]))

#Visualising the results
X_grid = np.arange(1,10,0.001)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='brown')
plt.plot(X_grid,regressor.predict(X_grid),color='red')
plt.title("Position vs Salary (random Forest)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()