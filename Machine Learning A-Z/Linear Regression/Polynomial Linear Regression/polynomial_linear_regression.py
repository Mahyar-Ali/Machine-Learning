#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:17:46 2020

@author: mahyar
"""

#importing Libraries
import pandas as pd

#importing Dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#In this case according to the context of the problem we are trying to make very accurate predictions
#Because there are only ten levels and we want to predict for level 6.5 and want to consider high bias
#because we will not be using it for levels greater than 10.
'''We will not split the data into test and train set'''

'''Fitting Polynomial Linear Regression'''
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly,y)

#Visualising the Polynomial Regression Model
import matplotlib.pyplot as plt
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_poly.predict(X_poly),color='yellow')
plt.title("Level vs Salary")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#Predicting a new result based on Polynomial Linear Regression
lin_reg_poly.predict(poly_reg.fit_transform([[6.5]]))

















