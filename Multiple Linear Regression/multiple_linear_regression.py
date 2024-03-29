#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:50:35 2019

@author: natigvahabov
"""

# Multiple Linear Regression

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading data
dataset = pd.read_csv('50_Startups.csv')

# dependent, independent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# converting categorical variable to integer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy variable trap
X = X[:, 1:]

# train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# fit, predict with LinearRegression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# backward elimination
import statsmodels.regression.linear_model as sm

def backwardElimination(X, sl):
    epochs = len(X[0])
    for i in range(0, epochs):
        regressor_OLS = sm.OLS(y, X).fit()
        varWithMaxPValue = max(regressor_OLS.pvalues).astype(float)
        if varWithMaxPValue > sl:
            for j in range(0, epochs - 1):
                if (regressor_OLS.pvalues[j].astype(float) == varWithMaxPValue):
                    X = np.delete(X, j, 1)
    regressor_OLS.summary()
    return X

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4]]
X_Modeled = backwardElimination(X_opt, SL)


