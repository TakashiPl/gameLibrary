import joblib
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

data = pd.read_csv("data/steam_games_2026.csv")
data["All_Tags"] = data["All_Tags"].fillna("")

num_cols = ["Price_USD","Total_Reviews"]
cat_cols = ["All_Tags"]

x = data[num_cols+cat_cols]
y = (data["Review_Score_Pct"] >= 85).astype(int)

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators",10,100)
    max_depth = trial.suggest_int("max_depth",2,20)
    min_samples_split = trial.suggest_int("min_samples_split",2,10)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat",CountVectorizer(max_features=25),"All_Tags")
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("transformer",preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
             )
        ]
    )

    scores = cross_val_score(pipeline,x,y,cv=5,scoring="accuracy")

    return scores.mean()

print("Rozpoczynam optymalizację hiperparametrów za pomocą Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("\nNajlepszy wynik (Accuracy):", study.best_value)
print("Najlepsze parametry:", study.best_params)