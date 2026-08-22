import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Steam Game Predictor API",
    description="API predicting, if game on steam will be recommended",
    version="1.0"
)

model = joblib.load("data/steam_model.joblib")

class GameInput(BaseModel):
    Price_USD: float
    Total_Reviews: int
    All_Tags: str

@app.post("/predict")
def predict_game_success(game: GameInput):
    input_data = [
        {
            "Price_USD": game.Price_USD,
            "Total_Reviews": game.Total_Reviews,
            "All_Tags": game.All_Tags
        }
    ]
    df = pd.DataFrame(input_data)

    prediction = model.predict(df)[0]

    return {
        "is_recommended": int(prediction),
        "status": (
            "Game recommended"
            if prediction == 1
            else "Game probably will not be recommended"
        )
    }