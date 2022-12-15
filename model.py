import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv("dataset_train.csv")

df.dropna(inplace=True)
# convert string date to datetime
df['Birthday'] = pd.to_datetime(df['Birthday'])
# separate datetime into day, month, year features
df['Birth_day'] = df['Birthday'].dt.day
df['Birth_month'] = df['Birthday'].dt.month
df['Birth_year'] = df['Birthday'].dt.year
df = df.drop(columns=['Birthday'])

X=df[['Best Hand','Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying','Birth_day','Birth_month','Birth_year']]
y=df['Hogwarts House']
print(X.columns)
print(X.shape)
ct = LabelEncoder()
    
X['Best Hand']= ct.fit_transform(X["Best Hand"])
#print(X.shape)
#X['Best Hand']=x.reshape(-1,1)






classifier = LogisticRegression(random_state=0)
classifier.fit(X, y)
pickle.dump(classifier, open("model.pkl", "wb"))