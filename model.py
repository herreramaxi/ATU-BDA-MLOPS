from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

data=pd.read_csv('spam.csv')
data['Spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)

X_train,X_test,y_train,y_test=train_test_split(data.Message,data.Spam,test_size=0.25)

model=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])

model.fit(X_train,y_train)

dump(model, "SpamModel.pkl")