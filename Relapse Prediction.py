# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:31:55 2021

@author: DEVANAND R
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_excel(r'C:\Users\DEVANAND R\Desktop\TNBC Project\TNBC_survival.xlsx')

#DATA PREPROCCESING

df.Stage.value_counts()
df["Tumor Size"].value_counts()
df["Tumor Size"] = df["Tumor Size"].str.replace("cm", "")
df['Tumor Size'] = df["Tumor Size"].str.replace(">", "")
df['Tumor Size'] = df['Tumor Size'].fillna(0)

df[["Tumor Size", "Tumor_area"]] = df["Tumor Size"].str.split("x", expand = True)

df['Tumor_area'] = df['Tumor_area'].fillna(0)

#converitng area to length

df['Tumor Size'] =np.sqrt((df["Tumor Size"].astype(float)*df["Tumor Size"].astype(float)) + ( df['Tumor_area'].astype(float) * df['Tumor_area'].astype(float)))

df["Tumor Size"] = df["Tumor Size"].fillna(0)


df.drop('Tumor_area', axis=1, inplace=True)
df1 = df

#white space cleaning

df["Chemo given initially"] = df["Chemo given initially"].str.lstrip()
df["Chemo given initially"] = df["Chemo given initially"].str.rstrip()

df['Treatment given on relapse'] = df['Treatment given on relapse'].str.lstrip()
df['Treatment given on relapse'] = df['Treatment given on relapse'].str.rstrip()
df['relapse'].hist()

#label encoding for stage

df['Stage'] = df['Stage'].replace(['Ia ','Ic ','IIa ','IIb ','IIC ', 'IIIa ','IIIc ','IV '],[1,2,3,4,5,6,7,8])


sns.heatmap(df.corr(), annot=True) #Corealtion


#removing unwanted oclumns

df.drop(['Treatment given on relapse','Survival ', 'event', 'relapse_time', 'Outcome_time'],axis=1, inplace=True)

df  = pd.get_dummies(df,columns = ['HPE', 'Surgery', 'Chemo given initially', ])
df.columns



df['Tumor Size'] = df['Tumor Size'].astype(int)  #rounding tumor length values
#normalizing the data

from sklearn.preprocessing import MinMaxScaler
x = df.values 
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_normalised = pd.DataFrame(x_scaled)
df_normalised.columns = df.columns

#df_normalised.drop('Age',axis=1, inplace=True)

X = df.drop('relapse', axis=1)
Y = df['relapse']


#splitting data

from sklearn.model_selection import train_test_split
x, x_test, y, y_test = train_test_split(X , Y, test_size = 0.3, stratify = Y)

#SMOTE oversampling

from collections import Counter
from imblearn.over_sampling import SMOTE

counter  = Counter(y)
smt =SMOTE()

x_train, y_train = smt.fit_resample(x, y)



#Naive Bayes Model using normal data

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

model_nave = MB()
model_nave.fit(x, y)

# Evaluation on Test Data
pred_nave = model_nave.predict(x_test)

print(classification_report(y_test, pred_nave))
confusion_matrix(y_test, pred_nave)


#KNN

from sklearn.neighbors import KNeighborsClassifier
acc=[]
for i in range(1,15,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train, y_train)
    test_acc = accuracy_score(y_test, neigh.predict(x_test))
    train_acc = accuracy_score(y, neigh.predict(x))
    acc.append([train_acc, test_acc])



# train accuracy plot 
plt.plot(np.arange(1,15,2),[i[0] for i in acc],"ro-",label="train accuracy")

# test accuracy plot
plt.plot(np.arange(1,15,2),[i[1] for i in acc],"bo-", label = "test accuracy")
plt.legend()

#from plotting the accuracy vs KNN graph. We can conclude the optimum K value as 5

model_knn = KNeighborsClassifier(n_neighbors=5)

model_knn.fit(x_train,y_train)
pred_KNN = model_knn.predict(x_test)
accuracy_score(y_test,pred_KNN)

print(classification_report(y_test, pred_KNN))

confusion_matrix(y_test,pred_KNN)


#logistics regression using normal data

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x,y)              

pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve

accuracy_score(y_test,pred) 

model.score(x,y) 

confusion_matrix(y_test,pred)

#XGBOOST using Smot sampled data

df = df.astype(int)

import xgboost as xgb
xgb_clf = xgb.XGBClassifier(learning_rate = 0.1, random_state = 42)


#Implementation of XG-Boost using grid Search CV

param_test1 = {'n_estimators': range(1,32,2) ,'max_depth': range(3,20,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x, y)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(y_test, cv_xg_clf.predict(x_test))
grid_search.best_params_

accuracy_score(y, cv_xg_clf.predict(x))

confusion_matrix(y_test, grid_search.predict(x_test))
 

#SVM using SMOT Sampled data
 
from sklearn.svm import SVC

model_linear = SVC(kernel = "linear")
model_linear.fit(x_train, y_train)
pred_test_linear = model_linear.predict(x_test)

accuracy_score(y_test, pred_test_linear )
pd.crosstab(y_test, pred_test_linear )

#saving model for delpoyment

import pickle
file = open('xgb.pkl', 'wb')
pickle.dump(xgb, file)




