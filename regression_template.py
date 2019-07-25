# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 20:35:08 2019

@author: Sadanand Vishwas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib


def modelRegression(X_trainset, X_testset, y_trainset, y_testset,
                     regressor, name):
    # Fitting the model to classifier
    regressor.fit(X_trainset, y_trainset)
    
    # Predicting the Results
    y_pred = regressor.predict(X_testset)
    y_pred = y_pred.apply(lambda x: 1 if x>=.5 else 0)
#    Making the training set
    from sklearn.metrics import accuracy_score, mean_absolute_error
    from sklearn.model_selection import cross_val_score
    # Applying K-Fold cross validation
    accuracies = cross_val_score(estimator=regressor, X=X_testset,
                                 y=y_testset, cv=10, n_jobs=-1)
    print("accuracy of "+name+" with cv is :",accuracies.mean())
    print("accuracy of "+name+" is :",accuracy_score(y_testset, y_pred))
    print("MAE of "+name+" is :",mean_absolute_error(y_testset, y_pred))
    # for the actual dataset
# =============================================================================
#     X_nan = (data_nan.drop(['shot_id_number'], axis=1,
#                           inplace=False)).iloc[:,:]
#     
#     y_nan_pred = regressor.predict(X_nan)
#     result_df = pd.DataFrame({'shot_id_number':data_nan['shot_id_number'],
#                               'is_goal':y_nan_pred}, 
#      columns=['shot_id_number','is_goal'])
#     
#     result_df.to_csv(name+"_prediction.csv", index=False)
# =============================================================================
  
# Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True, n_jobs=-1)

modelRegression(X_train, X_test, y_train, y_test, regressor,
                name="linear_regression")

# Gradiant Boosting Regression Model
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators=100, 
                                      min_samples_leaf=100, max_depth=8)

modelRegression(X_train, X_test, y_train, y_test, regressor,
                name="gradient_regression")

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', gamma=0.5)

modelRegression(X_train_scaled, X_test_scaled,
                y_train, y_test, regressor, name="svr_regression")

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=8 ,min_samples_leaf=100)

modelRegression(X_train, X_test, y_train, y_test, regressor,
                name="decision_regression")

# RandomForest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200,max_depth=8, min_samples_leaf=100,
                           n_jobs=-1)

modelRegression(X_train, X_test, y_train, y_test, regressor,
                name="randomforest_regression")

from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)

modelRegression(X_train, X_test, y_train, y_test, regressor,
                name="knn_regression")


