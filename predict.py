# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 23:12:30 2019

@author: Sadanand Vishwas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from sklearn.externals import joblib

## Read dataset
predict_data = pd.read_csv("data/predict_set.csv")

X_data = predict_data.drop(['shot_id_number'], axis=1, inplace=False).iloc[:,:]
# load the standerd scaler
sc = joblib.load("models/"+"standard_scaler.sav")
# Scale the data
X_data = sc.transform(X_data)

# function to predict the result and save it
def predict(X_test, filename):
    classifier = joblib.load("models/"+filename+".sav")
    y_pred = classifier.predict(X_test)
    
    # save the predicted results
    result_df = pd.DataFrame({'shot_id_number':predict_data['shot_id_number'],
                              'is_goal':y_pred},
    columns=['shot_id_number','is_goal'])
    result_df.to_csv("results/"+filename+"_prediction.csv", index=False)
    

# load the models and predict the results
predict(X_data, "logistic_regression_classifier")
predict(X_data, "knn_classifier")
predict(X_data, "svm_classifier")
predict(X_data, "decision_tree_classifier")
predict(X_data, "random_forest_classifier")
    
print("The model executed successfully and predicted results are stored.")
