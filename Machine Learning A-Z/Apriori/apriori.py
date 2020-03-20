#python3 Apriori
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:06:00 2020

@author: mahyar
"""

#Importing the libraries
import pandas as pd

#importing the dataset
dataset =pd.read_csv("Market_Basket_Optimisation.csv",header = None)  #Do not consider first row as header

transactions=[]
for i in range(0,7501):
	temp =[]
	for j in range(0,20):
		temp.append(str(dataset.values[i,j]))
	transactions.append(temp)

#Training Apriori on our dataset
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2,min_lift=3,min_length=2)
'''min_length=2 means that we only want to find relation between two object, like 
two movies. In short, we are only recommending one product.
min_support ~ 0.0028 because we want minimum three transactions of those products
which should be considered. i.e., 3*7/7500 --> We have data of one week.
--If we set a very high confidence then it will associate those items which are
bought frequently without having any relation between them
for this case min_lift is set to 3 but it is your choice according to the business model.'''

results = list(rules)
results_list = []
for i in range(0, len(results)):
results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nInfo:\t' + str(results[i][2]))



	 