# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:51:08 2021

@author: DEVANAND R
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_excel(r'C:\Users\DEVANAND R\Desktop\TNBC Project\DATA 70 pts AR + TNBC.xls')
df.head()


df['Grade'] = df['Grade'].replace(['III','II','I'],[3,2,1])
df.head()

#SORTING BASED ON GRADES

df_grade3 = df[df.Grade == 3]
df_grade2 = df[df.Grade == 2]
df_grade1 = df[df.Grade == 1]



N = df['Grade'].value_counts()

#CATEGORICAL PLOT BASED ON GRADE

sns.catplot(x="Menaupausalstatus", kind = 'count' , data=df, hue='Grade')

sns.catplot(x="Histology", kind = 'count' , data=df, hue='Grade')

sns.catplot(x="NS1", kind = 'count' , data=df, hue='Grade')

sns.countplot(x ="Histology" ,data=df, hue='Grade')


def grade_values(data):
    
    grade_values = []
    grade_counts = []
    grade = pd.DataFrame()
    col = data.columns
    for i in col:
        grade_values.append(data[i].value_counts().index[0])
        grade_counts.append(data[i].value_counts().iloc[0])
    
    grade['values'] = grade_values
    grade['count'] = grade_counts

    grade['values'] = data.columns.astype(str) + ' : ' + grade['values'].astype(str)
    
    return grade

grade1 = grade_values(df_grade1)   
grade2 = grade_values(df_grade2)
grade3 = grade_values(df_grade3)

#plotting based on grade values

sns.catplot(y="values", x ='count' ,kind = 'bar' ,data=grade1, orient='h',  )
sns.catplot(y="values", x ='count' ,kind = 'bar' ,data=grade2, orient='h',  )
sns.catplot(y="values", x ='count' ,kind = 'bar' ,data=grade3, orient='h',  )


#EDA Based on Stages of cancer

sns.catplot(x="Menaupausalstatus", kind = 'count' , data=df, hue='Stage')

sns.catplot(x="Histology", kind = 'count' , data=df, hue='Stage')

sns.catplot(x="NS1", kind = 'count' , data=df, hue='Stage')

sns.countplot(x ="Histology" ,data=df, hue='Grade')

sns.catplot(y="values", x ='count' ,kind = 'bar' ,data=grade2, orient='h',  )

plt.hist(df['age'])
plt.title('Age distribution')

plt.hist(df['Nodalstatus'])
plt.title("Nodal Status distribution")

#relation between RT AND Stage


sns.catplot(x="Stage", kind = 'count' , data=df, hue='RT')
plt.title("Radiation Therapy vs Stage")

#LVI AND STAGE

sns.catplot(x="Stage", kind = 'count' , data=df, hue='LVI')
plt.title("LVI vs Stage")