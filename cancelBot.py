import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

data = pd.read_csv('data.csv')
X = data['tweet'].head(1000)
y = data['hate_speech'].head(1000)


cv = CountVectorizer()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


features = cv.fit_transform(X_train)

model = svm.SVC()
model.fit(features, y_train)

features = cv.transform(X_test)

print(accuracy_score(y_test, model.predict(features)))





