import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

plt.style.use('default')

#load data
Location = 'movies.csv'
data = pd.read_csv(Location)

#display first 5 rows
print(data.head(), "\n")

# ----------- ? (Done by Karanraj) ------------

# ----------- KNN (Done by Theo) ------------
print("\n--------------KNN---------------")
X = data[['rt_audience_score',  'rt_score', 'imdb_rating', 'rank_in_year']]
y = data['rated_r']
#fit the model
knn = KNeighborsClassifier(4)
knn.fit(X, y)

#predict
knn_preds = knn.predict(X)

print(knn_preds)
#confusion matrix
conf = confusion_matrix(y, knn_preds, labels=knn.classes_)
print('Confusion matrix:\n')
print(knn.classes_)
print(conf)

#setting up
TN = conf[0][0]
TP = conf[1][1]
FP = conf[0][1]
FN = conf[1][0]

#accuracy score
acc = accuracy_score(y, knn_preds)
print('\nThe accuracy is: ', acc)

#precision
precision = TP / (TP + FP)

#recall
recall = TP / (TP + FN)

#f score
fscore = precision * recall / (precision + recall)

print("Precision: ", precision)
print("Recall: ", recall)
print("F Score: ", fscore)