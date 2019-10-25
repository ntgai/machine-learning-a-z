#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:11:04 2019

@author: natigvahabov
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading dataset
dataset = pd.read_csv('../Polynomial Regression/Position_Salaries.csv')

# dependent, independent variables
X = dataset.iloc[:, 1:2].values # we need to X become matrix, y vector
y = dataset.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# predicting value by SVR
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# visualizing
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()