#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

balti=pd.read_csv(r"C:\Users\Malikat Coulibaly\Downloads\bpd-arrests-1.csv")


# In[3]:


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


# In[4]:


dfChkBasics(balti)


# In[5]:


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


# In[6]:


import datetime as dt
def create_season(row):
 for index, row in balti.iterrows():
    if type(row["ArrestDate"]) is not str:
        return "Unknown"
    else:
        the_date = dt.datetime.strptime(row["ArrestDate"], "%m/%d/%y")
        return determine_season(the_date.timetuple().tm_yday)

balti['Season'] = balti.apply (lambda row: create_season(row), axis=1)


# In[7]:


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
 


# In[8]:


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


# In[9]:


wt=pd.read_csv(r"C:\Users\Malikat Coulibaly\Downloads\Balti_weather.csv")
balti = balti.join(wt.set_index("Date"), on="ArrestDate")


# In[10]:


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


# In[13]:


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


# In[14]:


print("\n15 Most Frequent Charge Description in Data Set")
chargedesc_count=balti["ChargeDescriptionClean"]
chargedesc_countval=chargedesc_count.value_counts()
print(chargedesc_countval.head(10))


# In[15]:


balti=balti.dropna(subset = ['Charge'])
balti["Charge"] = balti["Charge"].str.replace(' ', '')


# In[16]:


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


# In[17]:


balti.head()


# In[18]:


#Invidual
balti['Sex'].value_counts().plot(kind='barh')


# In[19]:


balti['Race'].value_counts().plot.pie(subplots=True, figsize=(9, 15))


# In[20]:


balti['Age'].hist()


# In[21]:


balti['ChargeDescriptionClean'].value_counts().plot(kind='barh')


# In[22]:


balti['Daynight'].value_counts().plot.pie(subplots=True, figsize=(9, 15))


# In[23]:


balti['Season'].value_counts().plot.pie(subplots=True, figsize=(9, 15))


# In[24]:


balti['District'].value_counts().plot(kind='barh')


# In[25]:


for col in balti.columns: 
    print(col) 


# In[26]:


# Grouped
sns.barplot(x='Sex', y='Age', hue='Race', data=balti,saturation=0.8)


# In[27]:


sns.barplot(x='Race', y='Age', hue='Sex', data=balti,saturation=0.8)


# In[28]:


sex_groups =balti.groupby('Sex').groups
male_ind = sex_groups['M']
female_ind = sex_groups['F']
plt.hist([balti.loc[male_ind,'Age'],balti.loc[female_ind,'Age']], 20, density = 1, histtype = 'bar', stacked=True, label = balti['Sex'].unique())
plt.xlabel('Age')
plt.title('Age by Sex')
plt.legend()
plt.show()


# In[29]:


sns.heatmap(pd.crosstab(balti['Sex'],balti['ChargeDescriptionClean'], normalize=True),vmin=0,vmax=1,annot=True,cbar_kws={'label': 'Fraction'})


# In[30]:


pd.crosstab(balti['Sex'],balti['ChargeDescriptionClean'])


# In[31]:


sns.barplot(x='Race', y='Age', hue='District', data=balti,saturation=0.8)


# In[39]:


# plot data
fig, ax = plt.subplots(figsize=(15,7))
# use unstack()
balti.groupby(['District','ChargeDescriptionClean']).count()['Race'].unstack().plot(ax=ax)


# In[45]:


balti.corr()


# In[47]:


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




