import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
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

# POPRAWNIE:
class SteamClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        return self.sigmoid(self.fc2(self.dropout(self.relu(self.fc1(x)))))

model = SteamClassifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
epochs = 10
for e in range(epochs):
    model.train()
    train_loss = 0.0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss/len(train_loader)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item()
            predicted_classes = (predictions >= 0.5).float()
            correct += (predicted_classes == batch_y).sum().item()
            total += batch_y.size(0)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = (correct / total) * 100
    print(
        f"Epoka {e + 1:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} |"
        f" Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%"
    )

