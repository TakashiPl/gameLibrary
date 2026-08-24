import joblib
import pandas as pd

model = joblib.load("data/steam_model.joblib")

data = [{
    "name": "Test_Game",
    "Price_USD": 20.0,
    "Total_Reviews": 15012,
    "All_Tags": "FPS Singleplayer Co-op Horror First-Person Action"
    }]


df = pd.DataFrame(data)

prediction = model.predict(df)

print(prediction[0])