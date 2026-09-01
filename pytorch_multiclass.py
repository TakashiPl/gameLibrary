import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

data = pd.read_csv("data/steam_games_2026.csv")

selected_genres = ["Shooter","RPG","Strategy"]

filtered_data = data[data["Primary_Genre"].isin(selected_genres)]

num_cols = ["Price_USD","Total_Reviews"]

X_Raw = filtered_data[num_cols]
scaler = StandardScaler()
X = scaler.fit_transform(X_Raw)

genres = filtered_data["Primary_Genre"]

le = LabelEncoder()
y = le.fit_transform(genres)

tensor_x = torch.tensor(X, dtype=torch.float32)
tensor_y = torch.tensor(y, dtype=torch.long)

print(tensor_x.shape)
print(tensor_y.shape)
print(tensor_y.dtype)