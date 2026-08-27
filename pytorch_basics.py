import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
 
data = pd.read_csv("data/steam_games_2026.csv")
df = pd.DataFrame(data)

num_cols = ["Price_USD","Total_Reviews"]

X_df = data[num_cols]


y = (data["Review_Score_Pct"]>=85).astype(int)

tensor_x = torch.tensor(X_df.values,dtype=torch.float32)
tensor_y = torch.tensor(y,dtype=torch.float32).unsqueeze(1)

tensor_DataSet = TensorDataset(tensor_x,tensor_y)

loader = DataLoader(tensor_DataSet,batch_size=32,shuffle=True)


class SteamClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2,16)
        self.fc2 = nn.Linear(16,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return (self.sigmoid(self.fc2(self.relu(self.fc1(x)))))

model = SteamClassifier()
batch_x, batch_y = next(iter(loader))
predictions = model(batch_x)
print(model, predictions.shape)
    

