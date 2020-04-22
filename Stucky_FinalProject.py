#%%
# Spencer Stucky
# Data Mining Final Project

#%%
# Standard quick checks
def dfChkBasics(dframe, valCnt = False): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(f'\n{cnt}: describe(): ')
  cnt+=1
  print(dframe.describe())

  print(f'\n{cnt}: dtypes: ')
  cnt+=1
  print(dframe.dtypes)

  try:
    print(f'\n{cnt}: columns: ')
    cnt+=1
    print(dframe.columns)
  except: pass

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1

# examples:
# dfChkBasics(df)
#%%
import os
import numpy as np
import pandas as pd
import geopandas as gdp
import datetime as dt
#%%
#load data and edit "Charge column" since that column has weird numbers & letter conventions 

os.chdir('/Users/spencerstucky/Documents/Documents/GWU School Work/DATS 6103/GWU_classes/DATS_6103_DataMining/DM_Final Project')
balti = pd.read_csv('Balti_data.csv', dtype={"Charge":str})

dfChkBasics(balti, True)

#%%
# Spencer Code
print(balti.head)
# convert cleaned dataset into csv
#balti.to_csv(r'/Users/spencerstucky/Documents/Documents/GWU School Work/DATS 6103/GWU_classes/DATS_6103_DataMining/Balti_Clean.csv', index = False)

# %%
# EDA
from pandas import value_counts
print(type(balti))
del balti['ArrestLocation']
del balti['IncidentLocation'] 
del balti['Location 1'] 
del balti['Unnamed: 5']
del balti['Unnamed: 6']
dfChkBasics(balti)

#%%
balti.describe()
balti.head()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
# EDA for non-location based analyses using balti dataframe

#%%
# Boxplots

sns.boxplot(balti['Charge'])

#%%
# Heatmap

# Finding the relationships between the variables
plt.figure(figsize=(20,10))
c = balti.corr()
sns.heatmap(c,cmap="Charge",cannot=True)

#%%
# Violin Plot

#%%
# Stacked histogram of age and gender
age_1 = balti.Age[ balti.Sex=='M']
age_2 = balti.Age[ balti.Sex=='F']

legend = ['Men', 'Women']
plt.hist([age_1,age_2], label=['Male', 'Female'], edgecolor='black', stacked=True)
plt.yticks([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000], ['0', '5k', '10k', '15k', '20k', '25k', '30k', '35k'])
plt.title('Age and Gender Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency of Arrest')
plt.legend()
plt.show()

#%%
# stacked histogram of gender and season
sex_1 = balti.Season[ balti.Sex=='M']
sex_2 = balti.Season[ balti.Sex=='F']

legend = ['African-American', 'Caucasian']
plt.hist([sex_1, sex_2], label=['Male', 'Female'], color = ['orange', 'yellow'], edgecolor='black', stacked=True)
plt.yticks([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000], ['0', '5k', '10k', '15k', '20k', '25k', '30k', '35k'])
plt.title('Season and Gender Stacked Histogram')
plt.xlabel('Season')
plt.ylabel('Frequency of Arrest')
plt.legend()
plt.show()

#%%
# stacked histogram of age and season
szn_1 = balti.Age[ balti.Season=='Fall']
szn_2 = balti.Age[ balti.Season=='Winter']
szn_3 = balti.Age[ balti.Season=='Spring']
szn_4 = balti.Age[ balti.Season=='Summer']

legend = ['Fall', 'Winter', 'Spring', 'Summer']
plt.hist([szn_1,szn_2, szn_3, szn_4], label=['Fall', 'Winter', 'Spring', 'Summer'], color = ['red','blue', 'green', 'orange'], edgecolor='black', stacked=True)
plt.yticks([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000], ['0', '5k', '10k', '15k', '20k', '25k', '30k', '35k', '40k'])
plt.title('Season and Age Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency of Arrest')
plt.legend()
plt.show()

#%%
# subset 5 most frequent charges
print(balti.Charge.value_counts())

Charge = ['11415', '43550', '10077', '11635', '30233']
balti.Charge.isin(Charge)

balti_Charge = balti[balti.Charge.isin(Charge)]
balti_Charge.shape

balti_Charge.Charge.unique()

#%%
# Histogram of 5 Most Frequent Charges

plt.hist(balti_Charge.Charge.dropna(), bins = 5, color = 'red', edgecolor='black')
plt.yticks([0, 5000, 10000, 15000, 20000, 25000], ['0', '5k', '10k', '15k', '20k', '25k'])
plt.xticks(['11415', '43550', '10077', '11635', '30233'],['Assault', 'Poss Non Marij', 'Failure to Appear', 'Poss Marij', 'Distr. Narcotic'])
plt.ylabel('Frequency of Arrest')
plt.xlabel('Type of Charge')
plt.title('Histogram of Top Five Charges')
plt.show()

#%%
# to match district code with district
import json
district_zip = {}
for district, code in zip(balti['District'], balti['District_Code']):
  if district_zip.get(district, None) is None:
      district_zip[district] = code
  elif district_zip[district] != code:
    print("{} should be {} but was {}".format(district, district_zip[district], code))

print(json.dumps(district_zip))
#%%
# Histogram of 3 Most Frequent Charges and District
# Drop NA values in District for graphing

D1 = balti.District.dropna()[ balti.Charge=='11415']
D2 = balti.District.dropna()[ balti.Charge=='43550']
D3 = balti.District.dropna()[ balti.Charge=='11635']
# took out failure to appear (3rd most frequent charge) bc it doesnt have district associated with it
# includes 2nd Degree Assault, Poss Non-Marij, and Poss Marij
# most assaults had an unknown location, and had a very high frequency
# removed the NA values from District to match crimes with district location better

plt.hist([D1, D2, D3], bins = 9, edgecolor='black', label=['Assault', 'Poss: Non-Marij', 'Poss: Marij'])
plt.xticks([0,1,2,3,4,5,6,7,8], ['NE','Cen', 'SE', 'E', 'W', 'S', 'N', 'NW', 'SW'])
legend = ['Assault', 'Poss: Non-Marij', 'Poss: Marij']
plt.legend()
plt.ylabel('Frequency of Arrest')
plt.xlabel('District')
plt.title('Histogram of Top 3 Charges by District')
plt.show()

#%%
print('Proportion Table for Charges by Category:')

# Combined like charges for the purposes of simpler graphing and display
# Assualt 1st and 2nd degree were combined, thefts of different $ amounts were combined,
# possess of marijuna charges were combined, narcotic distribution charges were combined
# and other (mainly nonviolent charges) were comined into a "other" category.

# can we cut down number of decimals?

#proportion of assault 1st and 2nd degree
prop_1 = len(balti[balti.Charge=='11415'])
prop_2 = len(balti[balti.Charge=='11420'])
prop_3 = (prop_1+prop_2) / len(balti.Charge)
print('Prop of Assaults:', prop_3) # 25.8%

#proportion of non marij possession
prop_4 = len(balti[balti.Charge=='43550'])/len(balti.Charge)
print('Prop of poss of non marij:', prop_4) # 21.2%

#proportion of failure to appear
prop_5 = len(balti[balti.Charge=='10077'])/len(balti.Charge)
print('Prop of failure to appear:', prop_5) # 19.6%

# proportion of possession of marijuana
prop_6 = len(balti[balti.Charge=='11635'])
prop_7 = len(balti[balti.Charge=='10573'])
prop_8 = (prop_6+prop_7) / len(balti.Charge)
print('Prop of poss of marij:', prop_8) # 9.9%

# proportion of narcotic distribution and/or production
prop_9 = len(balti[balti.Charge=='30233'])
prop_10 = len(balti[balti.Charge=='20696']) 
prop_11 = len(balti[balti.Charge=="2A0696"]) 
prop_12 = (prop_9+prop_10+prop_11) / len(balti.Charge)
print('Prop of Distr Narcotic:', prop_12) # 9.4%

# proportion of theft charges
prop_13 = len(balti[balti.Charge=='10521']) 
prop_14 = len(balti[balti.Charge== '10621'])
prop_15 = (prop_13+prop_14) / len(balti.Charge)
print('Prop of theft:', prop_15) # 4.8%

#proportion of other charges
prop_16 = len(balti[balti.Charge=='10088'])
prop_17 = len(balti[balti.Charge=='11093'])
prop_18 = len(balti[balti.Charge=='20050'])
prop_19 = len(balti[balti.Charge=='20705'])
prop_20 = (prop_16+prop_17+prop_18+prop_19) / len(balti.Charge)
print('Prop of other Crimes:', prop_20) # 9.1%

#%%
# Pie Chart of Charges by Category
print('Pie Chart of Charges by Category')

labels = "Assaults", 'Non Marijuana Possession', 'Failure to Appear', 'Possesion of Marijuana', 'Manufacture/Distr. of Narcotic', 'Theft', 'Other'
sizes = [prop_3, prop_4, prop_5, prop_8, prop_12, prop_15, prop_20]
colors = ['red', 'green', 'blue', 'orange', 'yellow', 'purple', 'pink']
explode = (0.1,0,0,0,0,0,0)

plt.pie(sizes, labels=labels, explode=explode, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Pie Chart of Charges by Category')
plt.axis('equal')
plt.show()

#%%
# Encode Variables for KNN Modeling

# Adding numerical variables for modeling 
balti['Daynight_code'] = np.where(balti["Daynight"].str.contains("Day"),1,0)
balti['Sex_code'] = np.where(balti["Sex"].str.contains("F"),0,1)
balti['Season_code'] = pd.factorize(balti['Season'])[0] + 1
balti['District_code'] = pd.factorize(balti['District'])[0] + 1

print(balti.Daynight_code)
print(balti.Sex_code)
print(balti.Season_code)
print(balti.District_code)

#%%
# subset top 5 most frequent charges
print("11415:", "2nd Degree Assault", '43550:', "Non Marijuana Posession", "10077:", "Failure to Appear", "11635:", "Possession Marijuana", "30233:", "Distribute Narcotics")

Charge = ['11415', '43550', '10077', '11635', '30233']
balti.Charge.isin(Charge)

balti_Charge = balti[balti.Charge.isin(Charge)]
balti_Charge.shape
balti_Charge.Charge.unique()

#%%
# KNN Models for Top 5 Most Frequent Charges

# prepare data
# target variable is top 5 charges
# cut down number of targets to simplify model and make it more accurate
# Independent vars are avg temperature in F, precipitation (to see if weather, like rain, has a relatoinship with crime),
# day or night encoded variable, male or female encoded variable, season encoded variable, and district encoded variable.

#seed = 42

ybalti = balti_Charge['Charge']

ybalti=pd.to_numeric(ybalti)
print(ybalti.head())
print(ybalti.shape)

# left out age bc of error, can incorporate more variables that Juhne has processed into categorical

xbalti = balti_Charge[["Mean.TemperatureF", "PrecipitationIn", 'Daynight_code', 'Sex_code', 'Season_code', 'District_code']]
print(xbalti.head())
print(xbalti.shape)

print(type(xbalti))
print(type(ybalti))

# make sure length of train and test are the same

print(balti_Charge.shape)
print(xbalti.shape)
print(ybalti.shape)

#%%
#Train test split 5:1, KNN Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(xbalti, ybalti, test_size = .20, stratify=ybalti, random_state = 1000)
# reset
# y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

print('x_train1 type',type(x_train))
print('x_train1 shape',x_train.shape)
print('x_test1 type',type(x_test))
print('x_test1 shape',x_test.shape)
print('y_train1 type',type(y_train))
print('y_train1 shape',y_train.shape)
print('y_test1 type',type(y_test))
print('y_test1 shape',y_test.shape)

#%%
# KNN Model: Train-Test Split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

kval = 10
knn_split = KNeighborsClassifier(n_neighbors=kval)
knn_split.fit(x_train, y_train)
print(f'KNN Train Score: {knn_split.score(x_train,y_train)}')
print(f'KNN Test Score: {knn_split.score(x_test,y_test)}')
print(confusion_matrix(y_test, knn_split.predict(x_test)))
print(classification_report(y_test, knn_split.predict(x_test)))

# Accuracy of model is 53% , which is a decent, but could be better.

# Cross Val on KNN
knn_split_cv = cross_val_score(knn_split, x_train, y_train, cv=10, scoring='accuracy')
print(f'\n KNN Cross Val Score: {knn_split_cv}\n')
np.mean(knn_split_cv)
# Cross Val Score is 52.5%. Our model is 52% accurate at explaining test or "unseen" data.

#%%
# Standard KNN Model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=kval)
knn.fit(xbalti, ybalti)
y_pred = knn.predict(xbalti)
print('Target Prediction', y_pred)

knn_score = knn.score(xbalti, ybalti)
print('KNN Score:', knn_score)
print(confusion_matrix(ybalti, knn.predict(xbalti)))
print(classification_report(ybalti, knn.predict(xbalti)))
# KNN accuracy is 59%, better than train-test split model.
# This is our best KNN model.

# Cross Val on Standard KNN
knn_cv = cross_val_score(knn, xbalti, ybalti, cv=5, scoring='accuracy')
print(f'\n KNN Cross Val Score: {knn_cv}\n')
print('CV Mean Score:')
np.mean(knn_cv)
# Cross Val score is 45%

# %%
# KNN CrossVal Model
knn_cv = KNeighborsClassifier(n_neighbors=kval) # instantiate with n value given

from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(knn_cv, xbalti, ybalti, cv=5)
print(cv_results) 
np.mean(cv_results)

# KNN cross val mean result is 45%
# Hence, this is a less accurate method of how our model will perform on test data
# and the least accurate knn model out of those tried.

# %%
# Graphs on KNN
