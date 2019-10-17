#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:44:22 2019

@author: natigvahabov
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# getting data with pandas
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]
 
# train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting y test values
y_pred = regressor.predict(X_test)

# visualizing train result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title = "Year of Experience & Salary [Train]"
plt.xlabel = "Years of Experience"
plt.ylabel = "Total amount of Salary(year)"
plt.show()

# visualizing train result
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title = "Year of Experience & Salary [Test]"
plt.xlabel = "Years of Experience"
plt.ylabel = "Total amount of Salary(year)"
plt.show()