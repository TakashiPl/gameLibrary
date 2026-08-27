import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
 
data = pd.read_csv("data/steam_games_2026.csv")
df = pd.DataFrame(data)

num_cols = ["Price_USD","Total_Reviews"]

X_df = data[num_cols]


y = (data["Review_Score_Pct"]>=85).astype(int)

tensor_x = torch.tensor(X_df.values,dtype=torch.float32)
tensor_y = torch.tensor(y,dtype=torch.float32).unsqueeze(1)

tensor_DataSet = TensorDataset(tensor_x,tensor_y)

loader = DataLoader(tensor_DataSet,batch_size=32,shuffle=True)

for batch_x,batch_y in loader:
    print(batch_x.shape, batch_y.shape)

# print("--- TENSOR X (CECHY) ---")
# print(f"Kształt (Shape): {tensor_x.shape}")
# print(f"Typ (Dtype):     {tensor_x.dtype}")
# print(f"Urządzenie:      {tensor_x.device}")

# print("\n--- TENSOR Y (TARGET) ---")
# print(f"Kształt (Shape): {tensor_y.shape}")
# print(f"Typ (Dtype):     {tensor_y.dtype}")
# print(f"Urządzenie:      {tensor_y.device}")


