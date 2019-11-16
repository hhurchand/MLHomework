# Machine Learning Homework
# Levels completed - Basic , advanced and Reach
# New estimator used : Ridge
# Also tried DecisionTreeRegressor
# Used MSRE and crossvalidation score to evaluate and compare models
# H Hurchand 15 Nov 2019


import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
boston = load_boston()

columns_names = boston.feature_names
# features(X) and target(y) are already separated
X = boston.data
Y = boston.target

# split into training and testing sets

X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.35)


lm = LinearRegression()
lm.fit(X_train,Y_train)

print(f"y-intercept : {lm.intercept_}")
print(f"Coefficients: {lm.coef_}")
print(f"Named Coeficients: {pd.DataFrame(lm.coef_, columns_names)}")

predicted_values = lm.predict(X_test)

# print performance
for (real, predicted) in list(zip(Y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {abs(real - predicted):.2f}")


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="inferno")

# Plotting difference between real and predicted values
sns.scatterplot(Y_test, predicted_values)
plt.plot([0, 50], [0, 50], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.show()

# Plotting the residuals: the error between the real and predicted values
residuals = Y_test - predicted_values
sns.scatterplot(Y_test, residuals)
plt.plot([50, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residual (difference)')
plt.show()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual (difference) Distribution')
plt.show()

# Understanding the error that we want to minimize
from sklearn import metrics
print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(Y_test, predicted_values)}")
print(f"Printing MSE error: {metrics.mean_squared_error(Y_test, predicted_values)}")
print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(Y_test, predicted_values))}")

# Training a model using multiple differents algorithms and comparing the results
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
for Model in [LinearRegression, GradientBoostingRegressor, ElasticNet, KNeighborsRegressor, Lasso, Ridge]:

    model = Model()
    cv = ShuffleSplit(n_splits=5)
    p_score = cross_val_score(model,X_train,Y_train,cv=cv)
    print(f"p_score with model {Model} = {p_score}")
#    model.fit(X_train, Y_train)
#    predicted_values = model.predict(X_test)
#    print(f"Printing RMSE error for {Model}: {np.sqrt(metrics.mean_squared_error(Y_test, predicted_values))}")


desTree = DecisionTreeRegressor(max_depth=20)
#desTree.fit(X_train,Y_train)
cv = ShuffleSplit(n_splits=5)
p_score = cross_val_score(desTree, X_train, Y_train, cv=cv)
#predicted_values_DT = desTree.predict(X_test)
print(f"p_score with model DecisionTreeRegressor = {p_score}")
#print(f"Printing RMSE error for DecisionTreeModel: {np.sqrt(metrics.mean_squared_error(Y_test, predicted_values))}")

#Learn more or give us feedback
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
cancer = datasets.load_breast_cancer()
wine = datasets.load_wine()
boston = datasets.load_boston()
diabetes = datasets.load_diabetes()

# Visualizing structure of dataset in 2D
pca = PCA(n_components=2)
proj = pca.fit_transform(iris.data)
plt.scatter(proj[:, 0], proj[:, 1], c=iris.target, edgecolors='black')
plt.colorbar()
plt.show()