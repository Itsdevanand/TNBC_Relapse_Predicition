# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 23:07:45 2021

@author: DEVANAND R
"""

from pywebio.platform.flask import webio_view

from pywebio import STATIC_PATH

from flask import Flask, send_from_directory

from pywebio.input import *
from pywebio.output import *

import pandas as pd

import  pickle
import numpy as np
model = pickle.load(open('xgb.pkl', 'rb')) #loading model

app = Flask(__name__)

def predict():
    
    age = input("enter age", type = NUMBER)
    tumorsize = input("tumor size in length cm",type = FLOAT)
    
    stage = select("Select cancer stage",['Ia','Ic','IIa','IIb','IIC', 'IIIa','IIIc','IV'])
    if stage == 'Ia':
        stage = 1
    if stage == 'Ic':
        stage = 2
    if stage == 'IIa':
        stage = 3
    if stage == 'IIb':
        stage = 4
    if stage == 'IIc':
        stage = 5
    if stage == 'IIIa':
        stage = 6
    if stage == 'IIIc':
        stage = 7
    if stage == 'IV':
        stage = 8
    
    surgery_level = input("enter surgery level(1,2,3)", type = NUMBER)
    HPE = select('Select surgery type', ['AGCT','JGCT','SCST','androgen secreting tumor'])
    
    if HPE == 'AGCT':
        AGCT = 1
        JGCT = 0
        SCST = 0
        AST  = 0
    if HPE == 'JGCT':
        AGCT = 0
        JGCT = 1
        SCST = 0
        AST  = 0
    if HPE == 'SCST':
        AGCT = 0
        JGCT = 0
        SCST = 1
        AST  = 0
    if HPE == 'androgen secreting tumor':
        AGCT = 0
        JGCT = 0
        SCST = 0
        AST  = 1
    surgery = select('Select surgery type', ['Surgery_Complete Surgery','Surgery_FSO', 'Surgery_Incomplete Surgery', 'Surgery_Optimal surgery'])
    if surgery == 'Surgery_Complete Surgery':
        Surgery_Complete_Surgery = 1
        Surgery_FSO = 0
        Surgery_Incomplete_Surgery = 0
        Surgery_Optimal_surgery = 0
    if surgery == 'Surgery_FSO':
        Surgery_Complete_Surgery = 0
        Surgery_FSO = 1
        Surgery_Incomplete_Surgery = 0
        Surgery_Optimal_surgery = 0
    if surgery == 'Surgery_Incomplete Surgery':
        Surgery_Complete_Surgery = 0
        Surgery_FSO = 0
        Surgery_Incomplete_Surgery = 1
        Surgery_Optimal_surgery = 0
    if surgery == 'Surgery_Optimal surgery':
        Surgery_Complete_Surgery = 0
        Surgery_FSO = 0
        Surgery_Incomplete_Surgery = 0
        Surgery_Optimal_surgery = 1
    chemo = select('Select type of chemo given to the patient', ['6 x CDDP +Ctx',
       '6xCDDP+ Ctx ; 6xCDDP + Etop',
       '6xCDDP+ VCR ; 6xCDDP +VP16 ;oral endoxan',
       '6xEP', 'BEP',
       'CDDP + CTx 6 cycles alone',
       'CDDP+Bleo +Vinb 3 cycles',
       'PEB 4 cycles', 'No chemo given'])
    if chemo == '6 x CDDP +Ctx':
        a = 1
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0
        h = 0
        i = 0
    if chemo == '6xCDDP+ Ctx ; 6xCDDP + Etop':
        a = 0
        b = 1
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0
        h = 0
        i = 0
    if chemo == '6xCDDP+ VCR ; 6xCDDP +VP16 ;oral endoxan':
        a = 0
        b = 0
        c = 1
        d = 0
        e = 0
        f = 0
        g = 0
        h = 0
        i = 0
    if chemo == '6xEP':
        a = 0
        b = 0
        c = 0
        d = 1
        e = 0
        f = 0
        g = 0
        h = 0
        i = 0
    if chemo == 'BEP':
        a = 0
        b = 0
        c = 0
        d = 0
        e = 1
        f = 0
        g = 0
        h = 0
        i = 0
    if chemo == 'CDDP + CTx 6 cycles alone':
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 1
        g = 0
        h = 0
        i = 0
    if chemo == 'CDDP+Bleo +Vinb 3 cycles':
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 1
        h = 0
        i = 0
    if chemo == 'PEB 4 cycles':
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0
        h = 1
        i = 0
    if chemo == 'No chemo given':
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0
        h = 0
        i = 1
        
    #converting to data frame
    
    pred_df = pd.DataFrame([[age,stage,tumorsize,surgery_level,AGCT,JGCT,SCST,AST,
              Surgery_Complete_Surgery,Surgery_FSO,Surgery_Incomplete_Surgery,Surgery_Optimal_surgery,
              a,b,c,d,e,f,g,h,i]])
    pred_df = pred_df.astype(int)
    

    prediction = model.predict(pred_df)
    
    if prediction == 1:
        put_text("Relapse can occur")
    if prediction == 0:
        put_text("relapse will not ccour")
        
        
df.columns.value_counts().sum()
    
    
 
    
app.add_url_rule('/tool', 'webio_view', webio_view(predict), methods=['GET','POST','OPTIONS'])

app.run(host = 'localhost', port=80) #running on local host



