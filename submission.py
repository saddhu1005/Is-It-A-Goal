# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:59:18 2019

@author: Sadanand Vishwas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('data/data.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)
#  'match_id', 'team_id'
data.describe()
data.apply(lambda x:sum(x.isnull()))
data.apply(lambda x:len(x.unique()))
data['shot_id_number'] = data.index+1

data.fillna({'remaining_min':data['remaining_min'].mean(),
             'power_of_shot':data['power_of_shot'].mean(),
             'remaining_sec':data['remaining_sec'].mean(),
             'distance_of_shot':data['distance_of_shot'].mean(),
             'location_x':data['location_x'].mean(),
             'location_y':data['location_y'].mean(),
             'remaining_min.1':data['remaining_min.1'].mean(),
             'power_of_shot.1':data['power_of_shot.1'].mean(),
             'remaining_sec.1':data['remaining_sec.1'].mean(),
             'distance_of_shot.1':data['distance_of_shot.1'].mean(),
             'knockout_match.1':data['knockout_match.1'].mean()},inplace=True)

vars=['knockout_match','area_of_shot','shot_basics', 'range_of_shot', 'team_name',
      'date_of_game', 'home/away', 'type_of_shot', 'type_of_combined_shot',
      'lat/lng', 'game_season']

for var in vars:
    data[var].fillna(method='ffill', inplace=True)
    
data['type_of_combined_shot'].fillna(method='bfill', inplace=True)

data['home_or_away'] = data['home/away'].apply(lambda x:
    'AWA' if x[5:6] == '@' else 'HOM')
    
data['time_min.1'] = data['remaining_min.1'] + data['remaining_sec.1'].apply(lambda x:
    x if x==0 else x/60)
    
times = [i for i in range(2, 131, 2)]
start_time = [i for i in range(0, 129, 2)]
def imputeTime(cols):
    time = cols[0]
    for i,time_i in enumerate(times):
        if float(time)<=float(time_i):
            return str(start_time[i])+'-'+str(time_i)


data['remaining_time'] = data[['time_min.1']].apply(imputeTime, axis=1).astype(str)

data.drop(['time_min.1','location_y', 'shot_basics', 'lat/lng','power_of_shot.1','distance_of_shot.1',
           'knockout_match.1','distance_of_shot.1', 'range_of_shot', 'type_of_shot',
           'match_event_id', 'team_name', 'team_id', 'match_id', 'date_of_game',
           'home/away', 'remaining_min', 'remaining_min.1', 'remaining_sec',
           'remaining_sec.1'],
    axis=1,inplace=True)

data.apply(lambda x:sum(x.isnull()))
data.apply(lambda x:len(x.unique()))

# Create broad seasons of 4 years each
seasons = ['2000','2004', '2008','2012','2016']
start_seasons = ['1996','2001','2005', '2009', '2013']
def imputSeason(cols):
    season=cols[0]
    for i,year in enumerate(seasons):
        if year>=season[:4]:
            return start_seasons[i]+'-'+year[-2:]

data['game_season_broad'] = data[['game_season']].apply(imputSeason, axis=1).astype(str)
data.drop(['game_season'],axis=1, inplace=True)

# Label Encoding    
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

vars = ['area_of_shot', 'home_or_away',
        'type_of_combined_shot',
       'game_season_broad', 'remaining_time']

for var in vars:
    data[var]=le.fit_transform(data[var])

# Correalation table(matrix)
cor = data.corr( method='pearson')

# OneHotEncoding
data=pd.get_dummies(data, columns=vars)

# Save modified data
data.to_csv("modified_data.csv", index=False)
# Read dataset
data_all = pd.read_csv("modified_data.csv")

data_fr = data_all.dropna(axis=0, inplace=False)
data_nan = data_all[data.isnull().any(axis=1)]
data_nan.drop(['is_goal'], axis=1, inplace=True)
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

def modelClassification(X_trainset, X_testset, y_trainset, y_testset,
                classifier, name):
    # Fitting the model to classifier
    classifier.fit(X_trainset, y_trainset)
    
    # Predicting the Results
    y_pred = classifier.predict(X_testset)

#    Making the training set
    from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
    cm = confusion_matrix(y_testset, y_pred)
    print(cm)
    print("Accuracy of "+name+" is :",accuracy_score(y_testset, y_pred))
    print("MAE of "+name+" is :",mean_absolute_error(y_testset, y_pred))
    # for the actual dataset
    X_nan = (data_nan.drop(['shot_id_number'], axis=1,
                          inplace=False)).iloc[:,:]
    
    y_nan_pred = classifier.predict(X_nan)
    result_df = pd.DataFrame({'shot_id_number':data_nan['shot_id_number'],
                              'is_goal':y_nan_pred},
    columns=['shot_id_number','is_goal'])
    result_df.to_csv("results/"+name+"_classifier_prediction_1.csv", index=False)



# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(n_jobs=-1)

modelClassification(X_train_scaled, X_test_scaled, y_train, y_test, classifier,
            name="logistic_regression")

# Fitting KNN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)

modelClassification(X_train_scaled, X_test_scaled, y_train, y_test, classifier, name="knn")

# =============================================================================
# # Fitting SVM to the Training set
# from sklearn.svm import SVC
# classifier=SVC(kernel='rbf',probability=True, class_weight='balanced')
# 
# model_train(X_train_scaled, X_test_scaled, y_train, y_test, classifier, name="svm")
# 
# =============================================================================

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=8, criterion = 'entropy',min_samples_leaf=20)

modelClassification(X_train_scaled, X_test_scaled, y_train, y_test, classifier, name="decision_tree")


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 110, max_depth = 10, 
                                    criterion = 'entropy',min_samples_leaf=25,
                                    n_jobs=-1)
modelClassification(X_train_scaled, X_test_scaled, y_train,
            y_test, classifier, name="random_forest")


print("The model is ran and executed successfully.")