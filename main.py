import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#load data
Location = 'College.csv'
data = pd.read_csv(Location)

#display first 5 rows
data.head()
print("\n")

# ----------- ? (Done by Karanraj) ------------


# ----------- LDA (Done by Bhavitavya) ------------


# ----------- KNN (Done by Theo) ------------
X = [['Lag1', 'Lag2', 'Volume']]
y = ['Direction']

#
knn = KNeighborsClassifier(5)
knn.fit(X, y)