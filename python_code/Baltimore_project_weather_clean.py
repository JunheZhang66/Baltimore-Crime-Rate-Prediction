

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
import geopandas as gpd
import datetime as dt
#%%

#load data and edit "Charge column" since that column has weird numbers & letter conventions 

os.chdir('/Users/paulinemckim/Desktop')
balti = pd.read_csv('Balti_data.csv', dtype={"Charge":str})

dfChkBasics(balti, True)

#%%
#Head of orignal data set
print("Head of the data")
balti.head()

#%% Value Counts of Non Null Values in original data set
print("Baltimore Not Null Value Counts")
balti_counts=balti.notnull().sum(axis=0)

#  Value Counts of  Null Values in original data set
print("n/Baltimore Null Value Counts")
balti_null=balti.isnull().sum(axis=0)

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

#It still looks messy so we will need to use other columns instead for analysis 
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
#%%
# %%
wt= pd.read_csv('Balti_weather.csv')
wt.head()
# %%

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

#%%
missing_values_table(balti)

# %%
# Checking to see if Charge Description is also cleaned up 

chargetrim=balti['ChargeDescriptionClean']
chargedesc_total={}
for row in chargetrim: 
    chargedesc_total[row] = chargedesc_total.get(row, 0) + 1
# print(chargedesc_total)
print(json.dumps(chargedesc_total, indent='\t'))

#%%

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
# a lot of ocation information is missing. We should not totally disregard this data but for location based analysis we should drop.
#%%
#Making Second DataFrame to use for Location 
balti_loca=balti.dropna(subset = ['Longitude',"Latitude","Location 1"])
print(balti_loca)

# %%
#check to see if drop of location info was successful 
missing_values_table(balti_loca)


#%%
#Bar Graph 
def bar_chart(feature):
    balti_chart = []
    balti_char_indexes = []
    for k, v in charge_to_desc_dict.items():
        balti_chart.append( balti[balti['Charge']==k][feature].value_counts() )
        balti_char_indexes.append(v)
    balti_chart = pd.DataFrame(balti_chart)
    balti_chart.index = balti_char_indexes
    balti_chart.plot(kind='bar',stacked=True, figsize=(10,5))

bar_chart("Sex")
bar_chart("Season")
bar_chart("Daynight")
#bar_chart("Max.TemperatureF")
#bar_chart()


#%%
x3 = len(balti[balti.Charge == "11415"])
x4 =len(balti[balti.Charge == "11420"])
x5=x3+x4
x6=(x5)/len(balti.Charge)
print(x6)

