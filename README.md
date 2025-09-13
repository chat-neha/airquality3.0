# airquality3.0

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# -----------------------------
# 1. Load & preprocess data
# -----------------------------
df = pd.read_csv("data.csv")
df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
df = df.sort_values("datetime").reset_index(drop=True)

target_col = "pm2p5"   # change if needed
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[[target_col]])

def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 24
X, y = create_sequences(scaled, SEQ_LEN)

# reshape for sklearn: (samples, features)
X = X.reshape(X.shape[0], -1)
y = y.ravel()

# -----------------------------
# 2. Train/Test Split (chronological)
# -----------------------------
split_idx = int(len(X) * 0.8)  # 80% train, 20% test
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# -----------------------------
# 3. PyTorch wrapper for sklearn
# -----------------------------
class TorchTimeSeriesRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_type="lstm", hidden_size=32, num_layers=1, 
                 lr=0.001, batch_size=32, epochs=20, seq_len=SEQ_LEN, device=None):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_dim, output_dim=1):
        if self.model_type == "lstm":
            return nn.LSTM(input_dim, self.hidden_size, self.num_layers, batch_first=True)
        elif self.model_type == "gru":
            return nn.GRU(input_dim, self.hidden_size, self.num_layers, batch_first=True)
        else:
            raise ValueError("Invalid model_type")

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X = X.reshape(X.shape[0], self.seq_len, -1)
        y = y.reshape(-1, 1)

        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                                 torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        input_dim = X.shape[2]
        self.model = self._build_model(input_dim).to(self.device)
        self.fc = nn.Linear(self.hidden_size, 1).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.fc.parameters()), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                if self.model_type == "lstm":
                    out, (h, c) = self.model(xb)
                else:
                    out, h = self.model(xb)
                preds = self.fc(out[:, -1, :])
                loss = criterion(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        X = check_array(X)
        X = X.reshape(X.shape[0], self.seq_len, -1)
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            if self.model_type == "lstm":
                out, (h, c) = self.model(xb)
            else:
                out, h = self.model(xb)
            preds = self.fc(out[:, -1, :]).cpu().numpy()
        return preds.ravel()

# -----------------------------
# 4. Hyperparameter tuning (on training only)
# -----------------------------
tscv = TimeSeriesSplit(n_splits=3)

param_grid = {
    "hidden_size": [16, 32],
    "num_layers": [1, 2],
    "lr": [0.001, 0.01],
    "epochs": [20]
}

# Best LSTM
lstm_est = TorchTimeSeriesRegressor(model_type="lstm")
grid_lstm = GridSearchCV(lstm_est, param_grid, cv=tscv,
                         scoring="neg_mean_absolute_error", verbose=1)
grid_lstm.fit(X_train, y_train)

# Best GRU
gru_est = TorchTimeSeriesRegressor(model_type="gru")
grid_gru = GridSearchCV(gru_est, param_grid, cv=tscv,
                        scoring="neg_mean_absolute_error", verbose=1)
grid_gru.fit(X_train, y_train)

best_lstm = clone(grid_lstm.best_estimator_)
best_gru = clone(grid_gru.best_estimator_)
best_lstm.fit(X_train, y_train)
best_gru.fit(X_train, y_train)

# -----------------------------
# 5. Stacking
# -----------------------------
lstm_preds_train = best_lstm.predict(X_train).reshape(-1, 1)
gru_preds_train = best_gru.predict(X_train).reshape(-1, 1)

stack_X_train = np.hstack([lstm_preds_train, gru_preds_train])

meta = Ridge()
meta.fit(stack_X_train, y_train)

# -----------------------------
# 6. Test Evaluation
# -----------------------------
lstm_preds_test = best_lstm.predict(X_test).reshape(-1, 1)
gru_preds_test = best_gru.predict(X_test).reshape(-1, 1)
stack_X_test = np.hstack([lstm_preds_test, gru_preds_test])

stacked_preds_test = meta.predict(stack_X_test)

mae_test = mean_absolute_error(y_test, stacked_preds_test)
rmse_test = np.sqrt(mean_squared_error(y_test, stacked_preds_test))
r2_test = r2_score(y_test, stacked_preds_test)

print("\nBest LSTM params:", grid_lstm.best_params_)
print("Best GRU params:", grid_gru.best_params_)
print("\n--- Test Set Performance ---")
print(f"MAE={mae_test:.4f}, RMSE={rmse_test:.4f}, R²={r2_test:.4f}")
