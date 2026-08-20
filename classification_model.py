from library import GameLibrary
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

dataSet = pd.read_csv("data/steam_games_2026.csv")
is_recommended = dataSet["Review_Score_Pct"]>=85
y = np.array(is_recommended).astype(int)
features = ["Price_USD","Total_Reviews"]
X = dataSet[features]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled,y_train)

print("\n-> Weryfikacja efektu skalowania:")
print(f"Surowe ceny (pierwsze 5): {X_train['Price_USD'].values[:5]}")
print(f"Przeskalowane ceny (pierwsze 5): {X_train_scaled[:5, 0]}")


y_pred = model.predict(X_test_scaled)
print(classification_report(y_test,y_pred))