# -*- coding: utf-8 -*-
"""
@author: srappold
"""

import pandas as pd
from sklearn import preprocessing 

#convert objects / non-numeric data types into numeric
def convertStringColsToInts(all_data):
    for f in all_data.columns:
        if all_data[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(all_data[f].values)) 
            all_data[f] = lbl.transform(list(all_data[f].values))
    return all_data
    
def split_join(df, field_name, split_string):
    temp_df = df.drop(field_name, axis=1).join(
            df[field_name].str.split(split_string, expand=True).stack()
            .reset_index(level=1, drop=True).rename(field_name))
    return temp_df

def rmissingvaluecol(dff,threshold):
    #Remove Missing Value Columns
    #dff = Data Frame passed into 
    #threshold = numeric value to determine percentage of missing values acceptable
    l = []
    l = list(dff.drop(dff.loc[:,list((100*(dff.isnull().sum()/len(dff.index))>=threshold))].columns, 1).columns.values)
    print("# Columns having more than %s percent missing values:"%threshold,(dff.shape[1] - len(l)))
    print("Columns:\n",list(set(list((dff.columns.values))) - set(l)))
    return l

def obfuscateName(df):
    for i, row in df.iterrows():
        #print(row)
        print(i)
    #two ways to do this.  
#    for i, row in df.iterrows():
#    if <something>:
#        row['ifor'] = x
#    else:
#       row['ifor'] = y

#   df.ix[i]['ifor'] = x
    
    #df[columnName] = df.apply(lambda row: x if something else y, axis=1)

from faker import Faker

#data file to be loaded
raw_data_file = "DATA/PPR Raw.csv"

#uses Pandas to load dataset
try:
    raw = pd.read_csv(raw_data_file, thousands=',', float_precision=2)
    print('File loaded')
except:
    print('Error: Data file not found')
    
 #Data Analysis that should be done prior to upload into AWS   
df = raw.dropna(subset=['AutoKey'])

#remove spaces from column names
df.columns = df.columns.str.replace(' ', '')

df['Organization/Suborganization'] = df['Organization/Suborganization'].str.slice(start=4, stop=8)
df['Travel_Location'] = df['Location/Destination(firstonly)'].str.upper()
#CHANGE TITLES
df['FullName'] = df['TravelerFirstName'].str.upper() + ' ' + df['TravelerLastName'].str.upper()
#df.set_index('FullName', inplace=True)

dfNames = pd.DataFrame(df['FullName'].unique().tolist())
dfNames.columns = ['FullName']
fake = Faker()
dfNames['Obfuscate'] = fake.name()
#dfNames['Obfuscate'] = dfNames.Obfuscate.apply(fake.name())

for i, row in dfNames.iterrows():
    row['Obfuscate'] = fake.name()
    
dfNames.to_csv('DATA/fullNames.csv', sep=',')
#Obfuscate values

#START DATA PROFILING
#import pandas_profiling
#pfr = pandas_profiling.ProfileReport(df)
#pfr.to_file("example.html")
    

bad_columns = rmissingvaluecol(df, 100)
df = df[bad_columns]

#PRINTS OUT THE NUMBER OF ROWS THEN COLUMNS
print(df.shape)

#PRINT OUT THE 1ST ROW OF INFORATION
print(df.head(0))

#PRINT OUT THE AVERAGE OF EACH STATISTICAL FIELD
print(df.mean())

print("Dimensions of the dataset: ")
print(df.dtypes)

print("Summary statistics of dataset: ")
print(df.describe())

print("Unique values for each column:")
print(df.nunique())