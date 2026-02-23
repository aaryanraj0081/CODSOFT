import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import tfidfvectorizer
from sklearn.preprocessing import labelencoder
from sklearn.svm import linearsvc
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv("train_data.txt", sep=':::', names=['id', 'title', 'genre', 'description'], engine='python')
test_sol_data = pd.read_csv("test_data_solution.txt", sep=':::', names=['id', 'title', 'genre', 'description'], engine='python')

train_data['description'] = train_data['description'].fillna(" ")
test_sol_data['description'] = test_sol_data['description'].fillna(" ")

t_v = tfidfvectorizer(stop_words='english', max_features=10000)
x_train = t_v.fit_transform(train_data['description'])
x_test = t_v.transform(test_sol_data['description'])

le = labelencoder()
y_train = le.fit_transform(train_data['genre'])
y_test = le.transform(test_sol_data['genre'])

model = linearsvc()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
