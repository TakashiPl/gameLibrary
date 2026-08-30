import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

data = pd.read_csv("data/steam_games_2026.csv")

num_cols = ["Price_USD","Total_Reviews"]

y = (data["Review_Score_Pct"]>=85).astype(int)

X_raw = data[num_cols]
scaler = StandardScaler()

X = scaler.fit_transform(X_raw)

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

tensor_X_train = torch.tensor(X_train,dtype=torch.float32)
tensor_y_train = torch.tensor(y_train.values,dtype=torch.float32).unsqueeze(1)

tensor_X_val = torch.tensor(X_val,dtype=torch.float32)
tensor_y_val = torch.tensor(y_val.values,dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(tensor_X_train,tensor_y_train),
    batch_size=32,shuffle=True)

val_loader = DataLoader(
    TensorDataset(tensor_X_val,tensor_y_val),
    batch_size=32,shuffle=False)


print(len(train_loader))
print(len(val_loader))

