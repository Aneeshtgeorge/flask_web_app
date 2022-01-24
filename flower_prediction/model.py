## This model is used to predict the species of the flower.
## importing the necessory libraries
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
## read the excel file
df1=pd.read_csv('iris1.csv')

## assigning x and y
x=df1.drop('Classification',axis=1)
y=df1['Classification']

## data is spilited
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.25)

## K=6
model1=KNeighborsClassifier(n_neighbors=6,metric='minkowski')
m=model1.fit(x_train,y_train)

## saving the model to disk
pickle.dump(model1,open('model.pkl','wb'))
