#uploaded = files.upload()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import math

data_heart= pd.read_csv("heart_change.csv")
data_heart.head(10)
x = data_heart.drop("HeartDisease",axis=1)
y= data_heart["HeartDisease"]

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state=42)
logmodel=LogisticRegression()
#logmodel.fit(trainX, trainY)
sc=StandardScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)
logmodel.fit(trainX_scaled, trainY)
predictions = logmodel.predict(testX_scaled)
classification_report(testY,predictions)
confusion_matrix(testY,predictions)
accuracy_score(testY, predictions)
logmodel.score(testX_scaled,testY)
#Datos tienen que ser un array 2D
array = [[57,1,2,164,176,0,0,115,1,1.2,0]]
print(array)

sc=StandardScaler()
scaler = sc.fit(trainX)
array_scaled = scaler.transform(array)
print(logmodel.predict(array_scaled))

