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

# ----------- LDA (Done by Bhavitavya) ------------
# Select relevant features and define the target variable
X = data[['rt_audience_score', 'rt_score', 'imdb_rating', 'rank_in_year']]  
y = data['rated_r']  

# Initialize and fit the LDA model
lda = LinearDiscriminantAnalysis()  
lda.fit(X, y)  

# Use the trained model to make predictions
lda_preds = lda.predict(X)  

# Evaluate the model using a confusion matrix
conf = confusion_matrix(y, lda_preds)  
print('Confusion matrix:\n')
print(conf)

# Calculate True Negatives, True Positives, False Positives, and False Negatives
TN = conf[0][0]  
TP = conf[1][1]  
FP = conf[0][1]  
FN = conf[1][0]  

# Calculate the accuracy of the model
acc = accuracy_score(y, lda_preds)  
print('\nThe accuracy is: ', acc)

# Calculate precision (positive predictive value)
precision = TP / (TP + FP)  

# Calculate recall (sensitivity or true positive rate)
recall = TP / (TP + FN)  

# Calculate the F-score
fscore = 2 * (precision * recall) / (precision + recall)  

# Print the results
print("Precision: ", precision)
print("Recall: ", recall)
print("F Score: ", fscore)
