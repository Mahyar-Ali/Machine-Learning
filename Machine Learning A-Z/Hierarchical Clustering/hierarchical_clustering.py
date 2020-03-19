# python3 Hierarchical Clustering
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:46:27 2020

@author: mahyar
"""
#IMporting Libraries
import numpy as py
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,3:].values

#Using the Dendrogram to find the optimal Number of clusters

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward')) #The ward method tries to minimize the
															#variance within each cluster
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Eucledian Distance")	
plt.show();												

#fitting the Hierarchical Clustering to our data set
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X);

#Visualisind the results
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=50,c = 'red' ,label = 'careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=50,c = 'blue' ,label = 'standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=50,c = 'green' ,label = 'target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=50,c = 'orange' ,label = 'careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=50,c = 'magenta' ,label = 'sensible')
plt.title("Clusters of Clients")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()  
