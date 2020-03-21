# python3 Thompson Sampling
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:09:50 2020

@author: mahyar
"""
#Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import random

#Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv") 

#Implementing the Thompson Sampling Algorithm
N = 10000
d = 10
ads_Selected = []
sum_rewards_0 = [0]*d
sum_rewards_1 = [0]*d
total_reward = 0
for n in range (0,N):
	ad=0
	max_random = 0
	for i in range(0,d):
		random_beta = random.betavariate(sum_rewards_1[i]+1,sum_rewards_0[i]+1)
		if(random_beta>max_random):
			max_random = random_beta
			ad = i
	ads_Selected.append(ad)
	if (dataset.values[n,ad]==1):
		sum_rewards_1[ad]+=1
	else:
		sum_rewards_0[ad]+=1
		
total_reward = sum(sum_rewards_1)
#Visualising the results
plt.hist(ads_Selected)
plt.title("Histogram of Ads Selections")
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()  
