#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:13:35 2020

@author: mahyar
"""
#importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dividing data into Dependent and Independent Variables
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,3].values

#Dealing with the missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Dealing with the Categorial Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting data into train set and test set.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test  = standardscaler.transform(X_test) 
# we are using the same value of mean
# and standard deviation because if we normalize both differently then we may end
#up on different scales. The reason why we don't use original X set to set mean
#and deviation is because in this way both test and train set will have a relation.

#Do not scale the training and test sets using different scalars: this could lead 
#to random skew in the data.

#Moreover Andrew Ng also did like this.It depends on you but remember they must
#be on the same scale.There would not be much relation between test and training set
#in this way but If you use the whole dataset to figure out the feature mean and 
#variance[imp], you're using knowledge about the distribution of the test set to
#set the scale of the training set - 'leaking' information.

#==============================================================#





