#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:30:56 2019

@author: natigvahabov
"""

# Random Forest Regression

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('../Polynomial Regression/Position_Salaries.csv')

# splitting dependent, independent variables
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# random forest regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state =42)
regressor.fit(X, y)

# predicting
y_pred = regressor.predict([[6.5]])

# plotting decision tree
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Bluff or Truth')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()