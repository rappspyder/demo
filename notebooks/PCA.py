# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 08:58:59 2020

@author: srappold
"""


import pandas as pd
import numpy as np

def split_join(df, field_name, split_string):
    temp_df = df.drop(field_name, axis=1).join(
            df[field_name].str.split(split_string, expand=True).stack()
            .reset_index(level=1, drop=True).rename(field_name))
    return temp_df

#data file to be loaded
raw_data_file = "DATA/PPR Raw.csv"

#uses Pandas to load dataset
try:
    df = pd.read_csv(raw_data_file, thousands=',', float_precision=2)
    print('File loaded')
except:
    print('Error: Data file not found')
    
df = df.dropna(subset=['AutoKey'])


# distributing the dataset into two components X and Y 
X = df.iloc[:, 0:13].values 
y = df.iloc[:, 13].values 


# Splitting the X and Y into the 
# Training set and Testing set 
from sklearn.model_selection import train_test_split 
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# performing preprocessing part 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
  
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 