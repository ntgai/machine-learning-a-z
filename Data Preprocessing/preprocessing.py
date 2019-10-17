# -*- coding: utf-8 -*-
# Data Preprocessing Template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading dataset
dataset = pd.read_csv('Data.csv')

# splitting independent, dependent variables (iloc)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# missing data (SimpleImputer)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categorical data with LabelEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

'''
Do we need to scale dummy variables? for some case yes for some case not
Do we need to scale y variable? in this case not, because it is categorical, in regression maybe
'''