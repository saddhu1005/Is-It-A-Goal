# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:00:19 2019

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
    
#data['time_min'] = data['remaining_min'] + data['remaining_sec'].apply(lambda x:
#    x if x==0 else x/60)
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
        'type_of_combined_shot', 'game_season_broad', 'remaining_time']

for var in vars:
    data[var]=le.fit_transform(data[var])

# Correalation table(matrix)
cor = data.corr( method='pearson')

#data.drop(['distance_of_shot', 'location_x'], axis=1, inplace=True)
#data.drop(['type_of_shot', 'time_min.1'], axis=1, inplace=True)
#vars = ['area_of_shot', 'range_of_shot', 'home_or_away',
#        'type_of_combined_shot',
#       'game_season_broad']
#vars = [ 'home_or_away', 'type_of_combined_shot',
#       'game_season_broad','area_of_shot', 'remaining_time']

# OneHotEncoding
data=pd.get_dummies(data, columns=vars)

## Save modified data
data.to_csv("data/modified_data.csv", index=False)
print("Data is cleaned and preprocessing is now done")
