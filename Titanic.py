# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:18:27 2019

@author: Akshay
"""
#C:\Users\Akshay\Desktop\titanic

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import math

######import#######

ip=pd.read_csv("/Users/Akshay/Desktop/titanic/train.csv",encoding="utf8")

df=pd.DataFrame(ip)
#print(df.head(10))
#print("#Pass: " +str(len(df.index)))

#####Analysis########

#sb.countplot(x="Survived", data=df)
#sb.countplot(x="Survived", hue="Sex", data=df)
#sb.countplot(x="Survived", hue="Embarked", data=df)
#df["Fare"].plot.hist(bins=10, figsize=(10,5))
#df.info()

#####Data Cleean/Wrangle#######

nul=df.isnull()
#print(nul)
total_nan=nul.sum()
#print(total_nan)
df.drop("Cabin", axis=1, inplace=True)
#print(df.head())
df.dropna(inplace=True)
res=df.isnull().sum()
#print(res)
#print(df.head())

sex=pd.get_dummies(df["Sex"], drop_first=True)
#print(sex)
embarked=pd.get_dummies(df["Embarked"], drop_first=True)
#print(embarked)
pcl=pd.get_dummies(df["Pclass"], drop_first=True)


df=pd.concat([df,sex,embarked,pcl],axis=1)
#print(df.head())
df.drop(["Pclass","Sex", "Embarked", "PassengerId", "Name", "Ticket"], axis=1, inplace=True)
#print(df.head())


#########Model_building#########

X=df.drop("Survived", axis=1)
y=df["Survived"]

from sklearn.model_selection import train_test_split as tts

X_train,X_test,y_train,y_test=tts(X,y, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression

LRmodel=LogisticRegression()
LRmodel.fit(X_train,y_train)




########Prediction#######
pred=LRmodel.predict(X_test)

from sklearn.metrics import classification_report as cr

cross_mat =cr(y_test,pred)
#print(cross_mat)

from sklearn.metrics import confusion_matrix as cm

matrix=cm(y_test,pred)
#print(matrix)

from sklearn.metrics import accuracy_score as score

acc_score=score(y_test,pred)
print(acc_score)
 
