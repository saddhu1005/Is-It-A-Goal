# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 20:35:08 2019

@author: Sadanand Vishwas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib

#Read the preprocessed dataset
data = pd.read_csv("data/modified_data.csv")
data_fr = data.dropna(axis=0, inplace=False)
data_nan = data[data.isnull().any(axis=1)]
data_nan.drop(['is_goal'], axis=1, inplace=True)

# save the unknown(the data for which we need to predict) dataset
data_nan.to_csv("data/predict_set.csv", index=False)

# Split the variables
y = data_fr['is_goal'].to_numpy()
X = (data_fr.drop(['is_goal', 'shot_id_number'], axis=1,
                        inplace=False)).iloc[:,:]

# Split the Train Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# save the standerd scaler 
joblib.dump(sc, "models/standard_scaler.sav")

# function to fit and train the model

def modelClassification(X_trainset, X_testset, y_trainset, y_testset,
                classifier, name):
    # Fitting the model to classifier
    classifier.fit(X_trainset, y_trainset)
    
    # Predicting the Results
    y_pred = classifier.predict(X_testset)

#    Printing the accuracy of model
    from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
    cm = confusion_matrix(y_testset, y_pred)
    print("Confusion Matrix:",cm)
    print("Accuracy of "+name+" is :",accuracy_score(y_testset, y_pred))
    print("MAE of "+name+" is :",mean_absolute_error(y_testset, y_pred))
    
    # Save the trained classifier
    joblib.dump(classifier, "models/"+name+"_classifier.sav")

    
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(n_jobs=-1)

modelClassification(X_train_scaled, X_test_scaled, y_train, y_test, classifier,
            name="logistic_regression")

# Fitting KNN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)

modelClassification(X_train_scaled, X_test_scaled, y_train, y_test, classifier, name="knn")

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier=SVC(kernel='rbf', class_weight='balanced', gamma=0.5)

modelClassification(X_train_scaled, X_test_scaled, y_train, y_test, classifier, name="svm")


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=8, criterion = 'entropy',min_samples_leaf=20)

modelClassification(X_train_scaled, X_test_scaled, y_train, y_test, classifier, name="decision_tree")

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, max_depth = 8, 
                                    criterion = 'entropy',min_samples_leaf=25,
                                    n_jobs=-1)
modelClassification(X_train_scaled, X_test_scaled, y_train,
            y_test, classifier, name="random_forest")

print("The Models are trained and saved successfully.")

