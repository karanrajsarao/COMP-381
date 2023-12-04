import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

plt.style.use('default')
#load data
Location = 'movies.csv'
df = pd.read_csv(Location)
#display first 5 rows
print(df.head(), "\n")
# Defining predictors and target variable
X_movie = df[['imdb_rating', 'rt_score', 'audience_freshness', 'rt_audience_score', 'length', 'rank_in_year', 'rating']]
Y_movie = df['revenue']

# One-hot encoding the 'rating' column
X_movie = pd.get_dummies(X_movie, columns=['rating'], drop_first=True)

# Spliting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_movie, Y_movie, test_size=0.5, random_state=123)

# Creating a linear regression model which is the Base Model
linmod_base = LinearRegression()
linmod_base.fit(X_train, Y_train)

# Making predictions on the test set
preds_base = linmod_base.predict(X_test)

# Creating a scatterplot for the base model with actual and predicted values
plt.scatter(Y_test, preds_base, color='g')
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Base Model Scatterplot")
plt.show()

# Defining coefficients for the base model
coefs_base = linmod_base.coef_

print("Coefficients for the Base Model are:")
for i in range(len(coefs_base)):
    print(f"{X_movie.columns[i]}: {coefs_base[i]}")

# Calculating and display R-squared value and MSE value for the base model
r2_base = r2_score(Y_test, preds_base)
mse_base = mean_squared_error(Y_test, preds_base)
print(f"Base Model's R-squared value is: {r2_base}, MSE value is: {mse_base}")

# Adding additional predictors ('rank_in_year' and 'rating')
X_add = df[['imdb_rating', 'rt_score', 'audience_freshness', 'rt_audience_score', 'length', 'rank_in_year', 'rating']]
X_add = pd.get_dummies(X_add, columns=['rating'], drop_first=True)
X_train_add, X_test_add, _, _ = train_test_split(X_add, Y_movie, test_size=0.5, random_state=123)

# Creating a linear regression model which is the Additional Predictors Model
linmod_add = LinearRegression()
linmod_add.fit(X_train_add, Y_train)

preds_add = linmod_add.predict(X_test_add)

# Displaying scatterplot for the Additional Predictors Model
plt.scatter(Y_test, preds_add, color='g')
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Additional Predictors Model Scatterplot")
plt.show()

# Calculating and displaying R-squared value and MSE value for the Additional Predictors Model
r2_add = r2_score(Y_test, preds_add)
mse_add = mean_squared_error(Y_test, preds_add)
print(f"Additional Predictors Model's R-squared value is: {r2_add}, MSE value is: {mse_add}")

# Adding interaction terms imdb_rating and rt_score and creating Interaction Terms Model
X_interact = X_add.copy()
X_interact['interaction_term'] = X_add['imdb_rating'] * X_add['rt_score']
X_train_interact, X_test_interact, _, _ = train_test_split(X_interact, Y_movie, test_size=0.5, random_state=123)

# Creating a linear regression model which is the Interaction Terms Model
linmod_interact = LinearRegression()
linmod_interact.fit(X_train_interact, Y_train)

preds_interact = linmod_interact.predict(X_test_interact)

# Displaying scatterplot for the Interaction Terms Model
plt.scatter(Y_test, preds_interact, color='g')
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Interaction Terms Model Scatterplot")
plt.show()

# Calculating and displaying R-squared value and MSE value for the Interaction Terms Model
r2_interact = r2_score(Y_test, preds_interact)
mse_interact = mean_squared_error(Y_test, preds_interact)
print(f"Interaction Terms Model's R-squared value is: {r2_interact}, MSE value is: {mse_interact}")

# Adding polynomial terms (polynomial of degree 2 for 'length')
poly_degree = PolynomialFeatures(2)
polyx = poly_degree.fit_transform(X_add[['length']])

X_train_poly, X_test_poly, _, _ = train_test_split(polyx, Y_movie, test_size=0.5, random_state=123)

# Creating a linear regression model which is the Polynomial Terms Model
linmod_poly = LinearRegression()
linmod_poly.fit(X_train_poly, Y_train)

preds_poly = linmod_poly.predict(poly_degree.transform(X_test_add[['length']]))

# Displaying scatterplot for the Polynomial Terms Model
plt.scatter(Y_test, preds_poly, color='g')
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Polynomial Terms Model Scatterplot")
plt.show()

# Calculating and displaying R-squared value and MSE value for the Polynomial Terms Model
r2_poly = r2_score(Y_test, preds_poly)
mse_poly = mean_squared_error(Y_test, preds_poly)
print(f"Polynomial Terms Model's R-squared value is: {r2_poly}, MSE value is: {mse_poly}")


