import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import tfidfvectorizer
from sklearn.preprocessing import labelencoder
from sklearn.svm import linearsvc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv("train_data.txt", sep=':::', names=['id', 'title', 'genre', 'description'], engine='python')
test_data = pd.read_csv("test_data.txt", sep=':::', names=['id', 'title', 'genre', 'description'], engine='python')
test_sol_data = pd.read_csv("test_data_solution.txt", sep=':::', names=['id', 'title', 'genre', 'description'], engine='python')

print(train_data.head())
print(train_data.shape)

plt.figure(figsize=(10,10))
sns.countplot(data=train_data, y='genre', order=train_data['genre'].value_counts().index, palette='viridis')
plt.show()



train_data['description'] = train_data['description'].fillna("unknown")
test_sol_data['description'] = test_sol_data['description'].fillna("unknown")

t_v = tfidfvectorizer(stop_words='english', max_features=10000)

x_train = t_v.fit_transform(train_data['description'])
x_test = t_v.transform(test_sol_data['description'])

le = labelencoder()
y_train = le.fit_transform(train_data['genre'])
y_test = le.transform(test_sol_data['genre'])



model = linearsvc()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("accuracy of the model:")
print(accuracy_score(y_test, y_pred))

print("full report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

def predict_genre(text):
    text_transformed = t_v.transform([text])
    prediction = model.predict(text_transformed)
    return le.inverse_transform(prediction)[0]

s1 = "a movie where police cashes the criminal and shoot him"
print("test 1 result:")
print(predict_genre(s1))

s2 = "a movie where person cashes a girl too get marry with him but girl refuses him."
print("test 2 result:")
print(predict_genre(s2))

s3 = "space travelers find a new planet with aliens and lasers"
print("test 3 result:")
print(predict_genre(s3))