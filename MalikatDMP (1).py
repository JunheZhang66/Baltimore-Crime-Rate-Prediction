#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

balti=pd.read_csv(r"C:\Users\Malikat Coulibaly\Downloads\bpd-arrests-1.csv")


# In[44]:


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


# In[45]:


dfChkBasics(balti)


# In[46]:


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


# In[47]:


import datetime as dt
def create_season(row):
 for index, row in balti.iterrows():
    if type(row["ArrestDate"]) is not str:
        return "Unknown"
    else:
        the_date = dt.datetime.strptime(row["ArrestDate"], "%m/%d/%y")
        return determine_season(the_date.timetuple().tm_yday)

balti['Season'] = balti.apply (lambda row: create_season(row), axis=1)


# In[48]:


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
 


# In[49]:


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


# In[50]:


wt=pd.read_csv(r"C:\Users\Malikat Coulibaly\Downloads\Balti_weather.csv")
balti = balti.join(wt.set_index("Date"), on="ArrestDate")


# In[51]:


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


# In[52]:


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


# In[53]:


print("\n15 Most Frequent Charge Description in Data Set")
chargedesc_count=balti["ChargeDescriptionClean"]
chargedesc_countval=chargedesc_count.value_counts()
print(chargedesc_countval.head(10))


# In[54]:


balti=balti.dropna(subset = ['Charge'])
balti["Charge"] = balti["Charge"].str.replace(' ', '')


# In[55]:


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


# In[56]:


balti.head()


# In[57]:


#Invidual
balti['Sex'].value_counts().plot(kind='barh')


# In[58]:


balti['Race'].value_counts().plot.pie(subplots=True, figsize=(9, 15))


# In[59]:


balti['Age'].hist()


# In[60]:


balti['ChargeDescriptionClean'].value_counts().plot(kind='barh')


# In[61]:


balti['Daynight'].value_counts().plot.pie(subplots=True, figsize=(9, 15))


# In[62]:


balti['Season'].value_counts().plot.pie(subplots=True, figsize=(9, 15))


# In[ ]:


balti['District'].value_counts().plot(kind='barh')


# In[ ]:


for col in balti.columns: 
    print(col) 


# In[ ]:


# Grouped
sns.barplot(x='Sex', y='Age', hue='District', data=balti,saturation=0.8)


# In[ ]:


sns.barplot(x='Race', y='Age', hue='Sex', data=balti,saturation=0.8)


# In[ ]:


sex_groups =balti.groupby('Sex').groups
male_ind = sex_groups['M']
female_ind = sex_groups['F']
plt.hist([balti.loc[male_ind,'Age'],balti.loc[female_ind,'Age']], 20, density = 1, histtype = 'bar', stacked=True, label = balti['Sex'].unique())
plt.xlabel('Age')
plt.title('Age by Sex')
plt.legend()
plt.show()


# In[ ]:


sns.heatmap(pd.crosstab(balti['Sex'],balti['ChargeDescriptionClean'], normalize=True),vmin=0,vmax=1,annot=True)


# In[ ]:


pd.crosstab(balti['Sex'],balti['ChargeDescriptionClean'])


# In[ ]:


sns.barplot(x='Race', y='Age', hue='District', data=balti,saturation=0.8)


# In[ ]:


# plot data
fig, ax = plt.subplots(figsize=(15,7))
# use unstack()
balti.groupby(['District','ChargeDescriptionClean']).count()['Race'].unstack().plot(ax=ax)


# In[ ]:


balti.h


# In[72]:


# TESTING CORR FOR CAT  VARIABLES
def chi_square_of_df_cols(df, col1, col2):
    df_col1, df_col2 = df[col1], df[col2]
    cats1, cats2 = categories(df_col1), categories(df_col2)

    def aux(is_cat1):
        return [sum(is_cat1 & (df_col2 == cat2))
                for cat2 in cats2]

    result = [aux(df_col1 == cat1)
              for cat1 in cats1]

    return scs.chi2_contingency(result)


# In[ ]:


for col in balti.columns: 
    print(col) 
    df.drop(['C', 'D'], axis = 1) 


# In[90]:


mybaltichi=balti.drop(['Arrest', 'Age','Race', 'ArrestDate', 'ArrestTime', 'ArrestLocation', 'IncidentOffense', 'IncidentLocation', 'Charge', 'ChargeDescription', 'Neighborhood','Post','Longitude','Latitude','Location 1','Max.TemperatureF','Mean.TemperatureF','Min.TemperatureF','PrecipitationIn','Unnamed: 5','Unnamed: 6'], axis = 1) 


# In[93]:


mybaltichi.head()


# In[74]:


chi_square_of_df_cols(mybaltichi)


# In[92]:


mybaltichi=mybaltichi.dropna()


# In[94]:


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


# In[95]:


len(mybaltichi)


# In[ ]:




