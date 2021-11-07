#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from sklearn import datasets, ensemble
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/train.csv")
#test_set = pd.read_csv(io.StringIO(uploaded['test.csv'].decode('utf-8')))
#print(df)

revenue = df["revenue"]

def date_to_int(date):
  lst = date.split("/")
  return 365*int(lst[2])+30*int(lst[0])+int(lst[1])

df["Open Date"] = df["Open Date"].transform(lambda x: date_to_int(x))
#print(df["Open Date"])

cities = []
def city_to_int(city):
  global cities
  if city in cities:
    return cities.index(city)
  else:
    cities.append(city)
    return cities.index(city)

df["City"] = df["City"].transform(lambda x: city_to_int(x))
#print(df["City"])

def citygroup_to_int(city):
  if city == "Big Cities":
    return 0
  else:
    return 1

df["City Group"] = df["City Group"].transform(lambda x: citygroup_to_int(x))
#print(df["City Group"])

def type_to_int(type):
  if type == "IL":
    return 0
  else:
    return 1

df["Type"] = df["Type"].transform(lambda x: type_to_int(x))
#print(df["Type"])

X_train = df.iloc[:, 1:42]
#X_test = test_set.iloc[:, 1:42]

X, x, Y, y = train_test_split(X_train, revenue, random_state = 0)

print("X: ")
print(X)
print("x: ")
print(x)
print("Y: ")
print(Y)
print("y: ")
print(y)

#print(X_train)

lr = LinearRegression().fit(X, Y)
print("Linear:")
print(lr.score(X, Y))
print(lr.score(x, y))

lasso = Lasso().fit(X, Y)
print("Lasso alpha = 1")
print(lasso.score(X, Y))
print(lasso.score(x, y))

ridge10 = Ridge(alpha=10).fit(X, Y)
print("Ridge alpha = 10")
print(ridge10.score(X, Y))
print(ridge10.score(x, y))

ridge01 = Ridge(alpha = 0.1).fit(X, Y)
print("Ridge alpha = 0.1")
print(ridge01.score(X, Y))
print(ridge01.score(x, y))

ridge = Ridge(alpha=5000).fit(X_train, revenue)
print("Ridge alpha = 5000")
print(ridge.score(X, Y))
print(ridge.score(x, y))
#print(lasso.score(X_train, revenue))
#svm = SVM().fit(X_train, revenue)
#print(svm.score(X_train, revenue))

params = {'n_estimators': 1000,
          'max_depth': 2,
          'min_samples_split': 20,
          'learning_rate': 0.01,
          'alpha': 0.99,
          'min_samples_leaf': 1,
          'loss': 'ls'}

gb = ensemble.GradientBoostingRegressor(**params)
gb.fit(X, Y)
print("Gradient Boosting")
print(gb.score(X, Y))
print(gb.score(x, y))

#calculate RMSE
y_pred = gb.predict(x)
print("RMSE:")
print(math.sqrt(mean_squared_error(y, y_pred)))

features = X.columns
feature_importance = gb.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12,6))
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(features)[sorted_idx])
plt.title("Feature Importance (MDI)")
plt.show()

rf = ensemble.RandomForestRegressor(n_jobs=-1, n_estimators=400, oob_score=True, max_features=0.5)
rf.fit(X, Y)
print("Random forest")
print(rf.score(X, Y))
print(rf.score(x, y))

plt.plot(lr.coef_, 'o', label = "Linear")
plt.plot(ridge.coef_, 's', label = "Ridge alpha = 5000")
plt.plot(ridge10.coef_, '^', label = "Ridge alpha = 10")
plt.plot(ridge01.coef_, 'p', label = "Ridge alpha = 0.1")
plt.plot(lasso.coef_, 'v', label = "Lasso")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()