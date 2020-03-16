#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:26:09 2020

@author: mahyar
"""
#import Libraries
import numpy as py
import pandas as pd
import matplotlib.pyplot as plt 

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size =0.2,random_state=0)


#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#To predict new values based on the fitting
y_predict = regressor.predict(X_test)

#Visualizing the Training set result, predicted line
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title ("Salary vs Experience")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary (Dollars)")
plt.show()

#Visualizing how it performs on the test set
plt.scatter(X_test,y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title ("Salary vs Experience")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary (Dollars)")
plt.show()

#Bonus
plt.scatter(X_test,y_predict,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title ("Salary vs Experience")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary (Dollars)")


