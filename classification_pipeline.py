import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

param_grid = {
    "classifier__max_depth": [5,10,None],
    "classifier__n_estimators": [50,100],
    "classifier__min_samples_split": [2,5],
}



dataSet = pd.read_csv("data/steam_games_2026.csv")
dataSet["Release_Date"] = pd.to_datetime(
    dataSet["Release_Date"], errors="coerce"
)
dataSet["Release_Year"] = dataSet["Release_Date"].dt.year
dataSet["Age_Years"] = 2026 - dataSet["Release_Year"]

dataSet["Age_Years"] = dataSet["Age_Years"].fillna(
    dataSet["Age_Years"].median()
)

num_features = ["Price_USD","Total_Reviews", "Age_Years"]
cat_features = ["All_Tags"]

is_recommended = dataSet["Review_Score_Pct"]>=85
y = np.array(is_recommended).astype(int)
X = dataSet[num_features+cat_features]

num_transformer = Pipeline(
    steps=[
        ("quantile", 
         QuantileTransformer(
            output_distribution="normal",random_state=42, n_quantiles=100
            )
        ),
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num",num_transformer,num_features),
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_estimators=100)),
    ]
)

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1
)



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y, random_state=42)

grid_search.fit(X_train,y_train)

print("Best parameters:",grid_search.best_params_)

best_model = grid_search.best_estimator_
best_y_pred = best_model.predict(X_test)


cm = confusion_matrix(y_test,best_y_pred, labels=grid_search.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)

disp.plot()
plt.show()

rf = grid_search.best_estimator_.named_steps['classifier']

feature_names = grid_search.best_estimator_.named_steps['preprocessor'].get_feature_names_out()

importances = pd.Series(rf.feature_importances_, index=feature_names)
print(importances.sort_values(ascending=False).head(10))