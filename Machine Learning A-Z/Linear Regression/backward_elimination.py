#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:30:09 2020

@author: mahyar
"""

'''Backward Elimination by Observing as you go'''
import numpy as np 
# X : Matrix of independent variables or dataset
# y : matrix of dependent variables
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis =1)

import statsmodels.api as sm
X_optim = X[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y,exog=X_optim).fit()
regressor_OLS.summary()
#See the summary and decide which variable to remove by folloeing the 
#Backward Elimination

'''End'''

'''Backward Elimination : automatically '''
import statsmodels.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

'''end'''