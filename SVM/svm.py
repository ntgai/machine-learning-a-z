#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:36:00 2019

@author: natigvahabov
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# dependent, independet variables
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.25, random_state = 42)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# svm
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 42)
classifier.fit(X_train, y_train)

# predicting test results
y_pred = classifier.predict(X_test)

# evaluating model result, 95% correct answer
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)