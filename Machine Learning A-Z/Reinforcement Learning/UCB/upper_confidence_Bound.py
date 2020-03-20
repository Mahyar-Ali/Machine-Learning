#python3 Upper Confidence Bound
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:26:58 2020

@author: mahyar
"""

#Importing the Libraries
import pandas as pd
import math
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

#Impementing the Upper Confidence Bound Algorithm
N = 10000 #Number of Rounds
ads_Selected = []
no_of_ads = 10
no_of_selections = [0]*no_of_ads
sum_of_rewards   = [0]*no_of_ads
total_reward = 0

for n in range (0,N):
	upper_confidence = [0]*no_of_ads #To Calculate Confidence of each ad in each round
	for i in range (0,10):
		if(no_of_selections[i]>0):
			average_reward = sum_of_rewards[i]/no_of_selections[i]
			delta_i = math.sqrt((3/2)*(math.log(n+1)/no_of_selections[i]))
			upper_confidence[i]=average_reward+delta_i
		else:
			upper_confidence[i] = 1e40 #To display the ad if no_of_selections_i == 0

	'''To Find the ad with maximum UCB, displaying it, adding its reward,incrementing its selection count'''
	maximum_upper_confidence,ad = max(upper_confidence),upper_confidence.index(max(upper_confidence))
	ads_Selected.append(ad)
	no_of_selections[ad] = no_of_selections[ad]+1
	sum_of_rewards[ad] += dataset.values[n,ad]

total_reward = sum(sum_of_rewards)

#Visualising the results
plt.hist(ads_Selected)
plt.title("Histogram of Ads Selections")
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()  
