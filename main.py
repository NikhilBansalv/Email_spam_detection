import numpy as np
import pandas as pd
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib as jb

data = pd.read_csv('emails.csv')
print(data.head())

data.drop_duplicates(inplace=True)
data.reset_index(inplace=True)
data.drop("index",axis=1,inplace=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'^subject\s*','',text,flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', '', text)
    return text

data['text'] = data['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['spam']

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=43,test_size=0.2)
#model = RandomForestClassifier(n_estimators=100,n_jobs=-1,max_depth=None)
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

jb.dump(model, 'spam_model.pkl')
jb.dump(vectorizer, 'vectorizer.pkl')
print("Vectorizer and model saved successfully.")