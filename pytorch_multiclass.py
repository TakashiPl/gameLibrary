import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn

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

class MultiClassClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,16)
        self.fc2 = nn.Linear(16,3)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MultiClassClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
epochs = 20
for e in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(tensor_x)
    loss = criterion(predictions, tensor_y)
    loss.backward()
    optimizer.step()
    if e % 5 == 0:
        print(loss.item(),f"This is the: {e} epoch")
        predicted_classes = torch.argmax(predictions,dim=1)
        accuracy = (((predicted_classes == tensor_y).sum().item())/len(tensor_y))*100
        print("Epoch:","("+str(e)+"/20)","|","Loss:","%.4f" % loss.item(),"|","Accuracy","%.2f" % accuracy + "%")
new_games = [[14.99,2000],[59.99,85000]]
scaled_new_games = scaler.transform(new_games)
tensor_new = torch.tensor(scaled_new_games,dtype=torch.float32)

model.eval()

with torch.no_grad():
    raw_preds = model(tensor_new)
    predicted_indices = torch.argmax(raw_preds,dim=1)
    predicted_genres = le.inverse_transform(predicted_indices.numpy())

print("\n--- WYNIKI PREDYKCJI DLA NOWYCH GIER ---")
print(
    f"Gra A ($14.99, 1200 recenzji) -> Przewidziany gatunek:"
    f" {predicted_genres[0]}"
)
print(
    f"Gra B ($59.99, 85000 recenzji) -> Przewidziany gatunek:"
    f" {predicted_genres[1]}"
)
