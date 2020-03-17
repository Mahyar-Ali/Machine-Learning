#python3 Decision Tree Regression
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:27:32 2020

@author: mahyar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataet
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2]
y = dataset.iloc[:,2]

'''Fitting Decision Tree Regression to the dataset'''
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#Predicting a new result
y_pred = regressor.predict(np.array([[6.5]])) 

#Visualising the results 
X_grid = np.arange(1, 10, 0.01) 
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='magenta')
plt.plot(X_grid,regressor.predict(X_grid),color='black')
plt.title("Level vs Salary")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()