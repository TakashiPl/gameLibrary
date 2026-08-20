from library import GameLibrary
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

dataSet = pd.read_csv("data/steam_clean.csv")
is_recommended = dataSet["rating"]>=85
numeric_df = dataSet[[
    "price",
    "playtime_hours",
    "release_year",
]]
y = np.array(is_recommended).astype(int)
X = numeric_df

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y, random_state=0)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))