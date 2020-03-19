#python3 k_means_clustering
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:29:43 2020

@author: mahyar
"""

#Importing the Libraries

import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,3:].values

#Using the Elbow Method To Decide the Number of Cluster
from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
	kmeans = KMeans(n_clusters= k,init = 'k-means++',max_iter=300,n_init = 10,random_state = 0) #n_init number of times we perform k means 
	kmeans.fit(X)
	wcss.append(kmeans.inertia_) #To compute within clusters sum of squares
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.show()

#Applying K_Means to the right Number of Clusters
k_means = KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=k_means.fit_predict(X)

#Visualising the Results
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=10,c='red',label='carefull')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=10,c='blue',label='standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=10,c='green',label='target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=10,c='orange',label='careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=10,c='yellow',label='sensible')
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=50,c='black')
plt.title("Clusters of Clients")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()   


