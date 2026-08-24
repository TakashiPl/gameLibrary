import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv("data/steam_games_2026.csv")
data["All_Tags"] = data["All_Tags"].fillna("")

is_recommended = (data["Review_Score_Pct"] >= 85).astype(int)

cat_features = ["All_Tags"]
num_features = ["Price_USD","Total_Reviews"]

y = is_recommended
X = data[num_features+cat_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", CountVectorizer(max_features=25), "All_Tags")
    ]
)

pipeline = Pipeline(
    steps=[
        ('transformer',preprocessor),
        ('regressor',RandomForestClassifier(random_state=42))
    ]
)

pipeline.fit(X,y)


joblib.dump(pipeline, "data/steam_model.joblib")