# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:22:51 2021

@author: DEVANAND R
"""

# Survival Analytics

import pandas as pd

df = pd.read_excel(r'C:\Users\DEVANAND R\Desktop\TNBC Project\TNBC_survival.xlsx')

survival = df[["relapse", "Outcome_time", "event"]]
survival.isna().sum()
survival.dropna(inplace=True)

#KaplanMeierFitter model to survival analysis

from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()

# fitting KaplanMeierFitter model
kmf.fit(survival["Outcome_time"], event_observed= survival["event"], label= 'survival')

# survival plot
kmf.plot()

relapse_1 = survival[survival["relapse"] == 1]
relapse_0 = survival[survival["relapse"] == 0]

# Plotting the Survival plot based on relapse and non relapse cases
kmf.fit(relapse_1["Outcome_time"], event_observed= relapse_1["event"], label = "relapse")
ax = kmf.plot()
kmf.fit(relapse_0["Outcome_time"], event_observed= relapse_0["event"], label = "No relapse")
kmf.plot(ax=ax)