#%%
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from constant import *
#%%
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
#%%
## Data Preparation
cat_features = list(df_train.columns[1:7])
cat_features.append(FAILURE)
cat_features.remove(LOADING)
num_features = list(df_train.columns[7:])
num_features.append(LOADING)
num_features.remove(FAILURE)
#%%
## combine the train and test data for preparation
df = pd.concat([df_train, df_test], axis=0)
#%%
## fill na, the missing value are just in numerical feature, we filled with the grouped median of training set for each product
for feature in num_features:
    df[feature] = df[feature].fillna(df_train.groupby([PRODUCT_CODE])[feature].transform(np.median))
#%%
## one-hot encoding nominal categorical variables with dummy variables
for feature in cat_features[0:-1]:
    df = pd.get_dummies(df, columns=[feature])
    df = df.drop([df.columns[-1]], axis=1)
#%%
## standardize all the numerical variables by training set for better regression results
scaler = StandardScaler()
scaler.fit(df_train[num_features])
df[num_features] = scaler.transform(df[num_features])
#%%
##split the data back into train and test sets
df_train = df.iloc[0: df_train.shape[0]].copy()
df_test = df.iloc[df_train.shape[0]:].copy()

X_train = df_train.drop([ID, FAILURE], axis=1).copy()
y_train = df_train[FAILURE].copy()
#%%
if __name__ == '__main__':
    print(X_train.describe())
    print(df_test.describe())
#%%
