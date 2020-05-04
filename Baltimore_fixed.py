# %%
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
import datetime as dt
#%%

#load data and edit "Charge column" since that column has weird numbers & letter conventions 

os.chdir('./data/')
balti = pd.read_csv('Balti_data.csv', dtype={"Charge":str})

dfChkBasics(balti, True)

#%%
#Head of orignal data set
print("Head of the data")
balti.head()

#%% Value Counts of Non Null Values in original data set
print("Baltimore Not Null Value Counts")
balti_counts=balti.notnull().sum(axis=0)
print(balti_counts)
#%%
#  Value Counts of  Null Values in original data set
print("Baltimore Null Value Counts")
balti_null=balti.isnull().sum(axis=0)
print(balti_null)

#%% # Function to get percents of missing values 
def missing_values_table(df):
    # Utility function, identify missing data and show percentages.
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\nThere are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
    return mis_val_table_ren_columns


#%% 
# Calling Percent of missing Values Function 
missing_values_table(balti)
    


#%%      
#add season column based on day of arrest column 
def determine_season(day):
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    # winter = everything else

    if day in spring:
        season = 'Spring'
    elif day in summer:
        season = 'Summer'
    elif day in fall:
        season = 'Fall'
    else:
        season = 'Winter'
    return season

def create_season(row):
# for index, row in balti.iterrows():
    if type(row["ArrestDate"]) is not str:
        return "Unknown"
    else:
        the_date = dt.datetime.strptime(row["ArrestDate"], "%m/%d/%y")
        return determine_season(the_date.timetuple().tm_yday)
balti['Season'] = balti.apply (lambda row: create_season(row), axis=1)


#%%
# add year column 
balti['year'] = pd.DatetimeIndex(balti['ArrestDate']).year
balti.head()


#%%
def determine_daynight(timestamp):
    import time
    if ':' in timestamp:
        t = time.strptime(timestamp, "%H:%M")
    elif '.' in timestamp:
        t = time.strptime(timestamp, "%H.%M")
    else:
        t = time.strptime(timestamp, "%H")
    hour = t.tm_hour
    daytime_hours = range(7,18) 
    
    if hour in daytime_hours:
        daynight= "Day"
    else: 
        daynight="Night"
    return daynight
 
       
def create_daynight(row):
# for index, row in balti.iterrows():
    if type(row["ArrestTime"]) is not str:
        return "Unknown"
    else:
        try:
            return determine_daynight(row["ArrestTime"])
        except Exception as e:
            print("Daytime conversion found an error: {}".format(e))
            return "Unknown"
balti['Daynight'] = balti.apply (lambda row: create_daynight(row), axis=1)
balti.head()
#%%
balti[balti["ArrestTime"]=="12:05"]

# %%
#weather dataset 
wt= pd.read_csv('Balti_weather.csv')
wt.head()
# %%
#join weather data set with crime data set 
balti = balti.join(wt.set_index("Date"), on="ArrestDate")
balti.head()
#%%
#Maybe we can use Description for our Analysis? Checking the data shows it is messy. This function can clean it up a little
def trim_charge_desc(row):
# for index, row in balti.iterrows():
    if type(row["ChargeDescription"]) is not str:
        # print("Broken, but unchanged cell value: {}:{}".format(type(row["ChargeDescription"]), row["ChargeDescription"]))
        return "Unknown"
    elif "||" in row["ChargeDescription"]:
        new_value = row["ChargeDescription"].split("||", 1)[0].strip()
        # print("Amended cell value: {}".format(new_value))
        return new_value
    else:
        # print("Unchanged cell value: {}".format(row["ChargeDescription"]))
        return row["ChargeDescription"].strip()
    #balti["ChargeDescription"] = balti["ChargeDescription"]
balti['ChargeDescriptionClean'] = balti.apply (lambda row: trim_charge_desc(row), axis=1)

#%%
#Dictionary shows many spelling variations and irregularities between "Charge" & ChargeDescription" 
charge_dict = {}
for charge, desc in zip(balti["Charge"], balti["ChargeDescriptionClean"]):
    # print(charge, desc)
    if charge_dict.get(charge) is None:
        charge_dict[charge] = {}
    charge_dict[charge][desc] = charge_dict[charge].get(desc, 0) + 1
print("Size of charge dict: {}".format(len(list(charge_dict.keys()))))

print(charge_dict)
import json
print(json.dumps(charge_dict, indent='\t'))


#%%
#Value Counts Confirms This with multiple spellings of "Failure to Appear" in top few Charge Descriptions
print("\n15 Most Frequent Charge Description in Data Set")
chargedesc_count=balti["ChargeDescriptionClean"]
chargedesc_countval=chargedesc_count.value_counts()
print(chargedesc_countval.head(10))

#It still looks messy so we will need to use other columns instead for analysis 
 # %%
 # We will use "Charge" as the column to determine which crimes are the 15 most frequent causes of arrest since this column is more clean than the "Charge Description" or Incident Offense" columns 
 # To faciliate the use of this column we should drop null values for this column and remove spaces to aid in easier analysis
balti=balti.dropna(subset = ['Charge'])
balti["Charge"] = balti["Charge"].str.replace(' ', '')
    
    
#%%
#value counts dictionary of "Charge" column to see number of occurances of crimes codes are the in the data set
chargenumdesc=balti['Charge']
#chargenumdesc_total={}
#for row in chargenumdesc: 
 #   chargenumdesc_total[row] = chargenumdesc_total.get(row, 0) + 1
#print(chargenumdesc_total)    

# Value counts of top charges of the 15 most frequent causes of arrest in the data set
print("\n15 Most Frequent Charge Codes in Data Set")

chargenum_counts=chargenumdesc.value_counts()
print(chargenum_counts.head(15))


# %%
# Since the data was messy and the Charge codes did not always match in Charge Description we used the Baltimore Government website to look up the exact crime each code stands for. 
#http://doc.dpscs.state.md.us/CJIS_DNA_Search/CJIScodelist_Arrest.asp?z_IP_CJ_CODE=LIKE&x_IP_CJ_CODE=0233&Submit=
# 15 most common charges for Arrest: code & desc 
#11415 - assualt 2nd degree 1
#43550- possess not marihuana: 2
#10077 Failure to appear: 3
#11635 poss Marijuana: 4
#30233 Cds:P W/I Dist:Narc: 5 
#10573	Cds: Possession-Marihuana: 6 
#10088 Violation of probation: 7
#11420  Asslt-First Degree: 8
#2A0696 Att-Cds Manuf/Dist-Narc: 9 
#10521 Theft Less Than $100.00: 10
#10621 Theft: Less $1,000 Value: 11 
#11093 General Prostitution: 12
#20050 Dis.Erly Conduct :13
#20696 Cds Manuf/Dist-Narc :14
#20705 Armed Robbery :15

charge_to_desc_dict = {
    "11415": "assualt 2nd degree 1",
    "43550": "possess not marihuana",
    "10077": "Failure to appear",
    "11635": "poss Marijuana",
    "30233": "Cds:P W/I Dist:Narc",
    "10573": "Cds: Possession-Marihuana",
    "10088": "Violation of probation",
    "11420": "Asslt-First Degree",
    "2A0696": "Att-Cds Manuf/Dist-Narc",
    "10521": "Theft Less Than $100.00",
    "10621": "Theft: Less $1,000 Value",
    "11093": "General Prostitution",
    "20050": "Dis.Erly Conduct",
    "20696": "Cds Manuf/Dist-Narc",
    "20705": "Armed Robbery",
}
# Strip out all arrests that are not part of the top fifteen charge numbers
useful_charge_codes = [x for x in charge_to_desc_dict.keys()]
balti= balti[balti['Charge'].isin(useful_charge_codes)]

# Unify their charge descriptions.
for k, v in charge_to_desc_dict.items():
    balti.loc[balti.Charge == k,'ChargeDescriptionClean']=v

# We should have ~100k rows.
len(balti)


#%% 
#Check dataset to make sure it still looks good
print("\n15 Most Frequent Charge Codes in Data Set")
chargenumdesc=balti['Charge']
chargenum_counts=chargenumdesc.value_counts()
print(chargenum_counts.head(15))

print("\n15 Most Frequent Charge Description in Data Set")
chargedesc_count=balti["ChargeDescriptionClean"]
chargedesc_countval=chargedesc_count.value_counts()
print(chargedesc_countval.head(15))


missing_values_table(balti)

#looks good

# %%
# Checking to see if Charge Description is also cleaned up 

chargetrim=balti['ChargeDescriptionClean']
chargedesc_total={}
for row in chargetrim: 
    chargedesc_total[row] = chargedesc_total.get(row, 0) + 1
# print(chargedesc_total)
print(json.dumps(chargedesc_total, indent='\t'))

#%%
#make sure data sorted correctly 
ctr1 = 0
charge_dict1 = {}
for charge, desc in zip(balti["Charge"], balti["ChargeDescriptionClean"]):
    # print(charge, desc)
    if charge_dict1.get(charge) is None:
        charge_dict1[charge] = {}
    charge_dict1[charge][desc] = charge_dict1[charge].get(desc, 0) + 1
    ctr1 += 1
    
print("Iterated size: {}".format(ctr1))
    
print("Size of charge dict: {}".format(len(list(charge_dict1.keys()))))


# print(charge_dict1)

print(json.dumps(charge_dict1, indent='\t'))

# %%
#Drop Columns that will not be used for analysis 
balti=balti.drop(columns=['IncidentOffense', 'ChargeDescription'])

#%%
missing_values_table(balti)
# a lot of location information is missing. We should not totally disregard this data but for location based analysis we should drop.
#%%
#Making Second DataFrame to use for Location 
balti_loca=balti.dropna(subset = ['Longitude',"Latitude","Location 1"])
#print(balti_loca)


#%%
#adding numerical variables for modeling 
balti["Daynight_Code"] = np.where(balti["Daynight"].str.contains("Day"),1,0)
balti["Sex_code"] = np.where(balti["Sex"].str.contains("F"),0,1)
balti['Season_code'] = pd.factorize(balti['Season'])[0] + 1
balti['District Code'] = pd.factorize(balti['District'])[0] + 1


# %%
#top 5 balti dataframe for possible analysis of these charges
top_5_balti=balti[(balti.Charge=="11415") | (balti.Charge=="43550") | (balti.Charge== "10077") | (balti.Charge=="11635") | (balti.Charge== "30233")]
print(top_5_balti)




#%%
#Bar Graph 
#function to explore bar graphs of "Charges" against other variables 
def bar_chart(feature):
    balti_chart = []
    balti_char_indexes = []
    for k, v in charge_to_desc_dict.items():
        balti_chart.append( balti[balti['Charge']==k][feature].value_counts() )
        balti_char_indexes.append(v)
    balti_chart = pd.DataFrame(balti_chart)
    balti_chart.index = balti_char_indexes
    balti_chart.plot(kind='bar', stacked=True, figsize=(10,5))

    
bar_chart("Sex")
bar_chart("Season")
bar_chart("Daynight")


#bar_chart()

#%%
# Bar graph that only looks at most common charge against other variables    
def make_bar_chart1(feature):
    balti_chart = [balti[balti['Charge']=="11415"][feature].value_counts() ]
    balti_chart = pd.DataFrame(balti_chart)
    balti_chart.index = ["assault 2nd degree"]
    balti_chart.plot(kind='bar', stacked=False, figsize=(10,5))
make_bar_chart1("District")



#%%

#proportion for an analysis Spencer plans to run 
x3 = len(balti[balti.Charge == "11415"])
x4 =len(balti[balti.Charge == "11420"])
x5=x3+x4
x6=(x5)/len(balti.Charge)
print(x6)
#%%
print(balti.head())
#balti["Neighborhood"]
#%%
balti_loca=balti_loca.dropna()
#%%
import plotly.express as px
#%%
balti_loca=balti.dropna(subset = ['District'])
#%%

px.set_mapbox_access_token("pk.eyJ1IjoibWNraW1wYyIsImEiOiJjazlhOW82MXAyMGhvM2dtc3AxeHVlbDA3In0.FmRE6YvGKR5CsJo4IgK9eg")
##df = px.data.carshare()
print("Charges-Male")
fig1 = px.scatter_mapbox(balti_loca[balti_loca["Sex"]=="M"], lat="Latitude", lon="Longitude", color="ChargeDescriptionClean", size_max=1, zoom=10.2)
fig1.show()
#%%
print("Charges-Female")
fig2 = px.scatter_mapbox(balti_loca[balti_loca["Sex"]=="F"], lat="Latitude", lon="Longitude", color="ChargeDescriptionClean", color_continuous_scale=px.colors.cyclical.IceFire, size_max=1, zoom=10.2)
fig2.show()

#fig3 = px.scatter_mapbox(balti_loca[(balti_loca["Sex"]=="M") & (balti_loca["year"]=="2013")], lat="Latitude", lon="Longitude", size_max=1, zoom=10.2)

#fig3 = px.scatter_mapbox(balti_loca[(balti_loca["Charge"]=="11415") & (balti_loca["Sex"]=="M")& (balti_loca["Daynight"]=="Day")], lat="Latitude", lon="Longitude", color="District", color_continuous_scale=px.colors.cyclical.IceFire, size_max=1, zoom=10.2)
#%%
print("Charges Yearly-Male")
fig3 = px.scatter_mapbox(balti_loca[(balti_loca["Charge"]!="zn") & (balti_loca["Sex"]=="M")], lat="Latitude", lon="Longitude", color="year", size_max=1, zoom=10.2)
fig3.show()
#%%
print("Charges Yearly_ Female")
fig4 = px.scatter_mapbox(balti_loca[(balti_loca["Charge"]!="zn") & (balti_loca["Sex"]=="F")], lat="Latitude", lon="Longitude", color="year", size_max=1, zoom=10.2)
fig4.show()
#%%
print("Charges by Day or Night-Female")
fig5 = px.scatter_mapbox(balti_loca[(balti_loca["Charge"]!="zn") & (balti_loca["Sex"]=="F")], lat="Latitude", lon="Longitude", color="Daynight", color_continuous_scale=px.colors.cyclical.IceFire, size_max=1, zoom=10.2)
fig5.show()
#%%
print("Charges by Day or Night-Male")
fig6 = px.scatter_mapbox(balti_loca[(balti_loca["Charge"]!="zn") & (balti_loca["Sex"]=="M")], lat="Latitude", lon="Longitude", color="Daynight", color_continuous_scale=px.colors.cyclical.IceFire, size_max=1, zoom=10.2)
fig6.show()
#fig = px.scatter_mapbox(balti, lat="Latitude", lon="Longitude",     color="ChargeDescriptionClean", hover_data='ChargeDescriptionClean', size_max=1, zoom=10)
#fig.show()
#fig1.show()
#fig2.show()
#fig3.show()



#%%
import seaborn as sns
sns.set(style="ticks", palette="pastel")



#boxplot age vs charge 
sns.boxplot(x="Age", y="ChargeDescriptionClean",
            data=balti)
sns.despine(offset=10, trim=True)




#%%
# subset top 5 most frequent charges
print("11415:", "2nd Degree Assault", '43550:', "Non Marijuana Posession", "10077:", "Failure to Appear", "11635:", "Possession Marijuana", "30233:", "Distribute Narcotics")

Charge = ['11415', '43550', '10077', '11635', '30233']
balti.Charge.isin(Charge)

balti_Charge = balti[balti.Charge.isin(Charge)]
balti_Charge.shape
balti_Charge.Charge.unique()
balti_Charge=balti_Charge.dropna()
print(balti_Charge.head())

#SOME VISUALIZATION CHARTS/EDA
balti['Sex'].value_counts().plot(kind='barh')
balti['District'].value_counts().plot.pie(subplots=True, figsize=(9, 15))
balti['Age'].value_counts().plot.pie(subplots=True, figsize=(9, 15))
balti['ChargeDescriptionClean'].value_counts().plot(kind='barh')
balti['Daynight'].value_counts().plot.pie(subplots=True, figsize=(9, 15))
balti['Season'].value_counts().plot.pie(subplots=True, figsize=(9, 15))
sns.barplot(x='Sex', y='Age', hue='Daynight', data=balti,saturation=0.8)
sns.barplot(x='District', y='Age', hue='Sex', data=balti,saturation=0.8)
sex_groups =balti.groupby('Sex').groups
male_ind = sex_groups['M']
female_ind = sex_groups['F']
plt.hist([balti.loc[male_ind,'Age'],balti.loc[female_ind,'Age']], 20, density = 1, histtype = 'bar', stacked=True, label = balti['Sex'].unique())
plt.xlabel('Age')
plt.title('Age by Sex')
plt.legend()
plt.show()
sns.heatmap(pd.crosstab(balti['Sex'],balti['ChargeDescriptionClean'], normalize=True),vmin=0,vmax=1,annot=True,cbar_kws={'label': 'Fraction'})
pd.crosstab(balti['Sex'],balti['ChargeDescriptionClean'])

#COMPUTATION OF CHI2
#DROPPING VARIABLES THAT WE DO NOT WANT
mybaltichi=balti.drop(['Arrest', 'Age','Race', 'ArrestDate', 'ArrestTime', 'ArrestLocation', 'IncidentOffense', 'IncidentLocation', 'Charge', 'ChargeDescription', 'Neighborhood','Post','Longitude','Latitude','Location 1','Max.TemperatureF','Mean.TemperatureF','Min.TemperatureF','PrecipitationIn','Unnamed: 5','Unnamed: 6'], axis = 1) 

#DEFINING THE FUNCTION

from scipy.stats import chi2, chi2_contingency
from itertools import combinations


def get_corr_mat(df, f=chi2_contingency):
    columns = df.columns
    dm = pd.DataFrame(index=columns, columns=columns)
    for var1, var2 in combinations(columns, 2):
        cont_table = pd.crosstab(df[var1], df[var2], margins=False)
        chi2_stat = f(cont_table)[0]
        dm.loc[var2, var1] = chi2_stat
        dm.loc[var1, var2] = chi2_stat
    dm.fillna(0, inplace=True)
    return dm

get_corr_mat(mybaltichi)


#%%
# KNN Models

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
#%%
# left out age bc of error, can incorporate more variables that Juhne has processed into categorical

xbalti = balti_Charge[['Age', "Mean.TemperatureF", "PrecipitationIn", 'Daynight_Code', 'Sex_code', 'Season_code', 'District Code']]
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

# Accuracy of model is 47% , which is a decent, but could be better.
#%%
# Cross Val on KNN
knn_split_cv = cross_val_score(knn_split, x_train, y_train, cv=10, scoring='accuracy')
print(f'\n KNN Cross Val Score: {knn_split_cv}\n')
np.mean(knn_split_cv)
# Cross Val Score is 46%. Our model is 46% accurate at explaining test or "unseen" data.

#%%
# Standard KNN Model


knn = KNeighborsClassifier(n_neighbors=kval)
knn.fit(xbalti, ybalti)
y_pred = knn.predict(xbalti)
print('Target Prediction', y_pred)

knn_score = knn.score(xbalti, ybalti)
print('KNN Score:', knn_score)
print(confusion_matrix(ybalti, knn.predict(xbalti)))
print(classification_report(ybalti, knn.predict(xbalti)))
# KNN accuracy is 57%, better than train-test split model.
# This is our best KNN model.
#%%
# Cross Val on Standard KNN
knn_cv = cross_val_score(knn, xbalti, ybalti, cv=5, scoring='accuracy')
print(f'\n KNN Cross Val Score: {knn_cv}\n')
print('CV Mean Score:')
np.mean(knn_cv)
# Cross Val score is 42%

# %%
# KNN CrossVal Model
knn_cv = KNeighborsClassifier(n_neighbors=kval) # instantiate with n value given

from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(knn_cv, xbalti, ybalti, cv=5)
print(cv_results) 
np.mean(cv_results)

# KNN cross val mean result is 42%
# Hence, this is a less accurate method of how our model will perform on test data
# and the least accurate knn model out of those tried.

# %%
# Graphs on KNN
