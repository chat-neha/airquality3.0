# airquality3.0

LSTM Code:
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from itertools import product

# -----------------------------
# 1. Load & preprocess data
# -----------------------------
df = pd.read_csv("data.csv")
df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
df = df.sort_values("datetime").reset_index(drop=True)

timestamps = df['datetime'].values
data = df.drop(columns=['datetime']).values.astype(float)  # all features

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# -----------------------------
# 2. Dataset
# -----------------------------
WINDOW_SIZE = 24   # past 3 days (3-hourly)
FORECAST_HORIZON = 8  # next 24h (8 x 3h)

class MultiStepDataset(Dataset):
    def __init__(self, series, window_size, horizon, target_col=0):
        self.series = series
        self.window_size = window_size
        self.horizon = horizon
        self.target_col = target_col

    def __len__(self):
        return len(self.series) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.window_size]         # (window, features)
        y = self.series[idx+self.window_size: idx+self.window_size+self.horizon, self.target_col]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

dataset = MultiStepDataset(data_scaled, WINDOW_SIZE, FORECAST_HORIZON, target_col=0)

# Train-test split (80-20)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# -----------------------------
# 3. Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=FORECAST_HORIZON):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)       # (batch, window, hidden)
        out = out[:, -1, :]         # last hidden state
        out = self.fc(out)          # (batch, horizon)
        return out

# -----------------------------
# 4. Training + evaluation
# -----------------------------
def train_model(model, train_loader, val_loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

def evaluate(model, loader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, y in loader:
            output = model(X)
            preds.append(output.numpy())
            actuals.append(y.numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    # inverse scale only target (first column)
    pm2p5_scaler = MinMaxScaler()
    pm2p5_scaler.min_, pm2p5_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
    preds_inv = pm2p5_scaler.inverse_transform(preds)
    actuals_inv = pm2p5_scaler.inverse_transform(actuals)
    return preds_inv, actuals_inv

def compute_metrics(y_true, y_pred, name=""):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    print(f"{name} - R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    return r2

# -----------------------------
# 5. Grid Search
# -----------------------------
param_grid = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "lr": [0.001, 0.0005]
}
best_score = -np.inf
best_params = None

for hidden_size, num_layers, lr in product(param_grid["hidden_size"], param_grid["num_layers"], param_grid["lr"]):
    print(f"\nTesting hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = LSTMModel(input_size=data.shape[1], hidden_size=hidden_size, num_layers=num_layers)
    train_model(model, train_loader, test_loader, epochs=20, lr=lr)

    train_preds, train_actuals = evaluate(model, train_loader)
    test_preds, test_actuals = evaluate(model, test_loader)

    r2 = compute_metrics(test_actuals, test_preds, "Test")
    if r2 > best_score:
        best_score = r2
        best_params = (hidden_size, num_layers, lr)

print(f"\nBest R²: {best_score:.4f} with params hidden_size={best_params[0]}, num_layers={best_params[1]}, lr={best_params[2]}")
