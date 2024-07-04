# AIML

import numpy as np
import pandas as pd

data1=pd.read_csv(r'C:\Users\HP\Desktop\jupyterpractice\Machine learning\pima-indians-diabetes.csv')
data1.head()

data1.info()
data1.describe()

y=data1['class']
x=data1.drop('class', axis=1)

from scipy.stats import zscore
x=x.apply(zscore)
x.head()
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

x_train.shape, x_test.shape
y_train.shape, y_test.shape

from sklearn.tree import DecisionTreeClassifier

dt_model=DecisionTreeClassifier(random_state=42)
dt_model

dt_model.fit(x_train, y_train)

y_pred=dt_model2.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score, classification_report

accuracy_score(y_test,y_pred)

report=classification_report(y_test,y_pred)
report
