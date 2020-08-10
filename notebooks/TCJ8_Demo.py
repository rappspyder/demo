# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:52:02 2019

@author: srappold
"""

import pandas as pd
import numpy as np
#from wordcloud import WordCloud, STOPWORDS
from sklearn import preprocessing 
from faker import Faker


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

def getElbowPoint(df):
    # THIS SECTION IS TO GET THE ELBOW (DETERMINE THE BEST NUMBER OF CLUSTERS TO USE)
    #elbow_data = df.iloc[:, [3,4,12]].values # this was selecting specific columns from the previous version
    elbow_data = df._get_numeric_data().dropna(axis=1)

    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(elbow_data)
        wcss.append(kmeans.inertia_)
    
    #Plotting the results onto a line graph, allowing us to observe 'The elbow'
    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS') #within cluster sum of squares
    return plt.show()
    # END ELBOW SECTION

def rmissingvaluecol(dff,threshold):
    #Remove Missing Value Columns
    #dff = Data Frame passed into 
    #threshold = numeric value to determine percentage of missing values acceptable
    l = []
    l = list(dff.drop(dff.loc[:,list((100*(dff.isnull().sum()/len(dff.index))>=threshold))].columns, 1).columns.values)
    print("# Columns having more than %s percent missing values:"%threshold,(dff.shape[1] - len(l)))
    print("Columns:\n",list(set(list((dff.columns.values))) - set(l)))
    return l

def score_in_percent (a,b):
    return (sum(a==b)*100)/len(a)

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
    
def removeCorrAndMissing(df):
    #REMOVE COLUMNS WITH 100% OF VALUES MISSING
    bad_columns = rmissingvaluecol(df, 100)
    df = df[bad_columns]
    
    #REMOVE COLUMNS WITH 95% CORRELATION TO OTHER FIELDS
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
    # Drop features 
    df.drop(to_drop, axis=1, inplace=True)
    return 1   

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

#df['Organization/Suborganization'] = df['Organization/Suborganization'].str.slice(start=4, stop=8)
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

dfLocs = pd.DataFrame(df['Travel_Location'].unique().tolist())
dfLocs.columns = ['TravelLocation']
dfLocs['ObsLocation'] = ''
for i, row in dfLocs.iterrows():
    row['ObsLocation'] = fake.city()
dfLocs.to_csv('DATA/Locations.csv', sep=',')   
#Obfuscate values

#START DATA PROFILING
import pandas_profiling
#import timeit
pfr = pandas_profiling.ProfileReport(df)
#timeit(pfr.to_file("example.html"))
pfr.to_file("example.html")
    
#print(df.sample(5))

#print(orig(df.copy()))
#print(faster(df.copy()))

#Updates NAN with 'Not Filled'
#df = df.replace(np.nan, 'Not Filled')

removeCorrAndMissing(df)

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

################################################
#GRAPHING
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="TripPurpose", data=df)
plt.show()

sns.countplot(x="TripType", data=df.where(df["TripType"] != "Temporary Duty Travel (Routine)"))
plt.show()

# Display the generated image:

################################################
#CLUSTERING
df_clustering = convertStringColsToInts(df.copy())
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=4, random_state=1) # number of clusters should be represented by getElbowPoint function
good_columns = df_clustering._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_

df_clustering['cluster'] = labels

#PLOT THESE CLUSTERS
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()

#Scatter plot
sns.lmplot('Organization/Suborganization', 'TotalDaysTDY', 
           data=df, 
           fit_reg=False, 
           hue="TripType",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('Clusters')
plt.xlabel('Org ID')
plt.ylabel('Total Days TDY ')
plt.show()

#TRY K-MEANS variants
# Convert the salary values to a numpy array
expenses = df['TotalTripExpenses'].values

# For compatibility with the SciPy implementation
expenses = expenses.reshape(-1, 1)
expenses = expenses.astype('float64')
# Import kmeans from SciPy
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
    
# Specify the data and the number of clusters to kmeans()
centroids, avg_distance = kmeans(expenses, 4, 300)
# Get the groups (clusters) and distances
groups, cdist = vq(expenses, centroids)
plt.scatter(expenses, np.arange(0,len(expenses)), c=groups)
plt.xlabel('Trip Expenses in (USD)')
plt.ylabel('Indices')
plt.show()


################################################
#NOW LETS SPLIT THIS INTO A TRAIN AND TEST SET
#train = df.sample(frac=0.8, random_state=1)
#test = df.loc[~df.index.isin(train.index)]

from sklearn.model_selection import train_test_split
train, test = train_test_split(df_clustering, test_size = 0.2)






#FEATURE IMPORTANCE
predictor_columns = ["TotalDaysTDY", "AdvanceAmount", "TripType", "Directorate", "TripPurpose"]

data = df.copy()
X = train.copy()  #independent columns
y = train['TotalTripExpenses']    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(test,test['TotalTripExpenses'].astype('int'))
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=300)
clf.fit(X, y.astype('int'))
print(clf.predict(test))

from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(y))
print(utils.multiclass.type_of_target(y.astype('int')))




# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8] #get last one


#UNIVATRIATE LINEAR REGRESSION
#PREDICT 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[["TotalTripExpenses"]], train["TotalDaysTDY"])
predictions = lr.predict(test[["TotalTripExpenses"]])

#CALCULATION SUMMARY STATS FOR THE MODEL
import statsmodels.formula.api as sm
model = sm.ols(formula='TotalDaysTDY + TotalTripExpenses + ExpensesbyLOA', data=train)
fitted = model.fit()
print(fitted.summary())

#RANDOM FOREST MODEL
from sklearn.ensemble import RandomForestRegressor
predictor_columns = ["TotalDaysTDY", "AdvanceAmount", "TotalTripExpenses"]
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5)
rf.fit(train[predictor_columns], train["TotalTripExpenses"])
predictions = rf.predict(test[predictor_columns])

#NOW THAT WE HAVE CLACULATED TWO MODELS, LETS CALCULATE THE MSE
from sklearn.metrics import mean_squared_error
print(predictions)
mean_squared_error(test["TotalTripExpenses"], predictions)

#### END OF COURSE #######

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization

# Saving feature names for later use
feature_list = list(train.columns)

# Export the image to a dot file
export_graphviz(tree, 
                out_file = 'tree.dot', 
                feature_names = feature_list, 
                rounded = True, 
                precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

#df.query('Username == "Scott Rappold"')


#NEW Decision Tree
all_data = convertStringColsToInts(df_clustering.dropna(subset=['AutoKey']))
Y_train = all_data['TotalTripExpenses']   
all_data.drop(['TotalTripExpenses'], axis=1, inplace=True)
        
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Prepare train and test for prediction
num_train = len(train)
Y_train = Y_train[:num_train]
X_train = all_data[:num_train]
X_test = all_data[num_train:]     

# create validation set
X_train, X_cv, y_train, y_cv = train_test_split( X_train, Y_train, test_size = 0.3, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                               max_depth=7, min_samples_leaf=5)
clf_gini.fit(X_train, y_train.astype('int'))


# Graphviz is used to build decision trees
from sklearn.tree import export_graphviz
from sklearn import tree

# This statement builds a dot file.
cols = list(X_train.columns.values)
tree.export_graphviz(clf_gini, out_file='tree.dot',feature_names  = cols)  
y_pred = clf_gini.predict(X_cv)


score_in_percent(y_pred,y_cv)
