#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:06:49 2019

@author: natigvahabov
"""

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# independent, dependent variables
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# KNN regressor
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# calculating accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)

'''
array([[59,  3],
       [ 4, 34]])
'''