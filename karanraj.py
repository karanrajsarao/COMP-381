import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, accuracy_score

plt.style.use('default')

#load data
Location = 'movies.csv'
data = pd.read_csv(Location)

#display first 5 rows
print(data.head(), "\n")

# ----------- ? (Done by Karanraj) ------------
# Select relevant features and define the target variable

