#python3 PCA Kernel
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:55:00 2020

@author: mahyar
"""

#python3 logistic regression
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:10:46 2020

@author: mahyar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,2:-1].values
y = dataset.iloc[:,4].values

#Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

'''Applying Kernel PCA'''
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel ='rbf')
X_train = kpca.fit_transform(X_train,y_train)
X_test  = kpca.transform(X_test)


#Fittting Logistic Regression to our Dataset
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

#Predicting new results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Visulaising the results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
