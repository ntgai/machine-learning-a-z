#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 09:38:10 2019

@author: natigvahabov
"""

# Polynomial Linear Regression
# Linearity means, can we write function in linear form: y = b0x0+b1x1+b2x1^2+ .. +bnx1^n

# importing librarires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# geting dataset
dataset = pd.read_csv('Position_Salaries.csv')

# dependent, independent variables
X = dataset.iloc[:, 1:2].values # we need to X become matrix, y vector
y = dataset.iloc[:, 2].values

# linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)

# multiple linear regression
mult_reg = LinearRegression()
mult_reg.fit(X_poly, y)

# visualizing
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.plot(X, mult_reg.predict(poly_reg.fit_transform(X)), color='green')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')