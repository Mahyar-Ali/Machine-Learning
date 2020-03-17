#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:21:22 2020

@author: mahyar
"""

import pandas as pd

#Import the Dataset 
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Dealing with the Categorial Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])

onehotencoder= OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:,1:]

#Dividing the dataset into test and training sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0) 

#Fitting the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Evaluating the test set
y_predict = regressor.predict(X_test)
