# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 09:57:52 2021

@author: DEVANAND R
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel(r'C:\Users\DEVANAND R\Desktop\TNBC Project\TNBC_survival.xlsx')

#DATA PREPROCCESING

df.Stage.value_counts()
df["Tumor Size"].value_counts()
df["Tumor Size"] = df["Tumor Size"].str.replace("cm", "")
df['Tumor Size'] = df["Tumor Size"].str.replace(">", "")
df['Tumor Size'] = df['Tumor Size'].fillna(0)

df[["Tumor", "Tumor_area"]] = df["Tumor Size"].str.split("x", expand = True)

df['Tumor'] = df['Tumor'].fillna(0)
df['Tumor_area'] = df['Tumor_area'].fillna(0)

df['Tumor_area'] = df["Tumor"].astype(float) * df['Tumor_area'].astype(float)

df["Tumor Size"] = df["Tumor Size"].fillna(0)

df["Tumor Size"] = df["Tumor Size"].str.replace("x", "   ")


t = df["Tumor Size"].astype(str).tolist()

for i in range(len(df["Tumor Size"])):
    
    if len(t[i]) >3:
        
        df.iloc[i,3] = 0
        
df.columns
df['Outcome_time'] =  df['Outcome_time'].fillna(df['Outcome_time'].median())
df.drop('Tumor', axis=1, inplace=True)
df1 = df

df["Chemo given initially"] = df["Chemo given initially"].str.lstrip()
df["Chemo given initially"] = df["Chemo given initially"].str.rstrip()

df['Treatment given on relapse'] = df['Treatment given on relapse'].str.lstrip()
df['Treatment given on relapse'] = df['Treatment given on relapse'].str.rstrip()




#EDA

#figure 1
sns.catplot("Stage", hue= 'Survival ', data=df, kind="count")
plt.title("Survival  vs Stage of Cancer")

sns.catplot("Stage", hue= 'relapse', data=df, kind="count")
plt.title("Relpase  vs Stage of Cancer")

#figure  2

sns.catplot('Surgery', hue= 'Survival ', data=df, kind="count")
plt.title("Survival  vs Surgery")

sns.catplot('Surgery', hue= 'relapse', data=df, kind="count")
plt.title("Relapse vs Surgery")

#figure 3

sns.catplot('Age', hue= 'relapse', data=df, kind="count")
plt.title("Relapse  vs Age")


sns.catplot('Age', hue= 'Survival ', data=df, kind="count")
plt.title("Survival  vs Age")

#figure 4

sns.catplot('Tumor Size', hue= 'Survival ', data=df, kind="count")
plt.title("Survival  vs HPE")

#figure 5

sns.catplot('Chemo given initially', hue= 'relapse', data=df, kind="count")
plt.xticks(rotation=45)
plt.title("Relapse  vs chemo given intially")

sns.catplot('Chemo given initially', hue= 'Survival ', data=df, kind="count")
plt.xticks(rotation=45)
plt.title("Survvial  vs chemo given intially")


df['Treatment given on relapse'].value_counts()
df['Chemo given initially'].value_counts()

df.columns

survived = df[df['Survival '] == 'Alive']
not_survived = df[df['Survival '] == 'Died']

#function for sorting survival values wrt alive and dead

def survival_values(data):
    
    survival_values = []
    survival_counts = []
    survival = pd.DataFrame()
    col = data.columns
    for i in col:
        survival_values.append(data[i].value_counts().index[0])
        survival_counts.append(data[i].value_counts().iloc[0])
    
    survival['values'] = survival_values
    survival['count'] = survival_counts

    survival['values'] = data.columns.astype(str) + ' : ' + survival['values'].astype(str)
    
    return survival


survived_values = survival_values(survived)
not_survived_values =  survival_values(not_survived)

sns.catplot(y="values", x ='count' ,kind = 'bar' ,data=survived_values , orient='h',  )
plt.title('Most repeated parameters in Survived patients')

sns.catplot(y="values", x ='count' ,kind = 'bar' ,data=not_survived_values, orient='h',  )
plt.title('Most repeated values in not survived patients')


relapse = df[df['relapse'] ==  1]
non_relapse = df[df['relapse'] == 0]

#the above function is used for collecting relapse and non relapse data

relapsed_values = survival_values(relapse)
non_relapsed_values = survival_values(non_relapse)

sns.catplot(y="values", x ='count' ,kind = 'bar' ,data=relapsed_values , orient='h',  )
plt.title('Most repeated parameters in relapsed patients')

sns.catplot(y="values", x ='count' ,kind = 'bar' ,data=non_relapsed_values, orient='h',  )
plt.title('Most repeated values in non relapsed patients')






