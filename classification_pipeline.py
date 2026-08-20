from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

num_features = ["Price_USD","Total_Reviews"]
cat_features = ["All_Tags","Release_Date"]

dataSet = pd.read_csv("data/steam_games_2026.csv")
is_recommended = dataSet["Review_Score_Pct"]>=85
y = np.array(is_recommended).astype(int)
X = dataSet[num_features+cat_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),num_features),
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(classification_report(y_test,y_pred))