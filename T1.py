# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:16:36 2023

@author: alikh
"""
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression


#Data Frames

data = pd.read_csv('Admission_Predict.csv')
data_new = pd.read_csv('Admission_Predict_Ver1.1.csv')
new = data_new.tail(100)


#Variables that were used

X = data.drop('Chance of Admit ', axis = 1)
y = data['Chance of Admit ']
X_test = new.drop('Chance of Admit ', axis = 1)
y_test = new['Chance of Admit ']

#Processes 

lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X_test)
diff   = np.empty(len(y_pred),dtype = float)


for i in range(len(y_pred)):
    diff[i]=y_pred[i]-y_test[i+400]


Predicted_vs_actual = pd.DataFrame({'predicted':y_pred,'actual':y_test,'difference':diff})
print(Predicted_vs_actual)