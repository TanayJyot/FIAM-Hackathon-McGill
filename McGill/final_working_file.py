import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler

from cuml.ensemble import RandomForestRegressor
from cuml import GradientBoostingRegressor
from cuml import XGBoostRegressor

# Train and predict function for RAPIDS models
def train_rapids_model(model, X_train, Y_train, X_test):
    model.fit(X_train, Y_train)
    return model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10)
reg_pred["rf"] = train_rapids_model(rf_model, X_train.cpu().numpy(), Y_train.cpu().numpy(), X_test.cpu().numpy())

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
reg_pred["gb"] = train_rapids_model(gb_model, X_train.cpu().numpy(), Y_train.cpu().numpy(), X_test.cpu().numpy())

# Extreme Gradient Boosting (XGBoost)
xgb_model = XGBoostRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
reg_pred["xgb"] = train_rapids_model(xgb_model, X_train.cpu().numpy(), Y_train.cpu().numpy(), X_test.cpu().numpy())

class CNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(32 * (input_dim // 2), 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for CNN
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * (x.size(2)))
        return self.fc(x)

class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 50).to(x.device)  # Hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 50).to(x.device)
        c0 = torch.zeros(2, x.size(0), 50).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# class LinearRegression(nn.Module):
#     def __init__(self, input_dim):
#         super(LinearRegression, self).__init__()
#         self.linear = nn.Linear(input_dim, 1, bias=False)
#
#     def forward(self, x):
#         return self.linear(x)
#
#
# class Lasso(nn.Module):
#     def __init__(self, input_dim, alpha=1.0):
#         super(Lasso, self).__init__()
#         self.linear = nn.Linear(input_dim, 1, bias=False)
#         self.alpha = alpha
#
#     def forward(self, x):
#         return self.linear(x)
#
#     def l1_loss(self):
#         return self.alpha * torch.sum(torch.abs(self.linear.weight))
#
#
# class Ridge(nn.Module):
#     def __init__(self, input_dim, alpha=1.0):
#         super(Ridge, self).__init__()
#         self.linear = nn.Linear(input_dim, 1, bias=False)
#         self.alpha = alpha
#
#     def forward(self, x):
#         return self.linear(x)
#
#     def l2_loss(self):
#         return self.alpha * torch.sum(self.linear.weight ** 2)
#
#
# class ElasticNet(nn.Module):
#     def __init__(self, input_dim, alpha=1.0, l1_ratio=0.5):
#         super(ElasticNet, self).__init__()
#         self.linear = nn.Linear(input_dim, 1, bias=False)
#         self.alpha = alpha
#         self.l1_ratio = l1_ratio
#
#     def forward(self, x):
#         return self.linear(x)
#
#     def elastic_net_loss(self):
#         l1 = self.l1_ratio * torch.sum(torch.abs(self.linear.weight))
#         l2 = (1 - self.l1_ratio) * torch.sum(self.linear.weight ** 2)
#         return self.alpha * (l1 + l2)
#

def train_model(model, X, y, epochs=1000, lr=0.01, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    X = X.to(device)
    y = y.to(device)

    for epoch in range(epochs):
        y_pred = model(X)
        loss = criterion(y_pred, y)

        # if isinstance(model, CNNRegressor):
        #     loss += model.l1_loss()
        # elif isinstance(model, RNNRegressor):
        #     loss += model.l2_loss()
        # elif isinstance(model, LSTMRegressor):
        #     loss += model.elastic_net_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def predict(model, X, device='cpu'):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        return model(X).cpu().numpy()


if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(datetime.datetime.now())

    # Set working directory and read data
    work_dir = "/mnt/c/Users/tjsin/Documents/FIAM Hackathon"
    raw = pd.read_csv(os.path.join(work_dir, "hackathon_sample_v2.csv"), parse_dates=["date"], low_memory=False)
    stock_vars = list(pd.read_csv(os.path.join(work_dir, "factor_char_list.csv"))["variable"].values)
    ret_var = "stock_exret"
    new_set = raw[raw[ret_var].notna()].copy()

    # Transform variables
    data = pd.DataFrame()
    for _, monthly_raw in new_set.groupby("date"):
        group = monthly_raw.copy()
        for var in stock_vars:
            group[var] = group[var].fillna(group[var].median())
            group[var] = (group[var].rank(method="dense") - 1) / (group[var].rank(method="dense").max() - 1) * 2 - 1
        data = data._append(group, ignore_index=True)

    # Initialize variables
    starting = pd.to_datetime("20000101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()

    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime("20240101", format="%Y%m%d"):
        cutoff = [
            starting,
            starting + pd.DateOffset(years=8 + counter),
            starting + pd.DateOffset(years=10 + counter),
            starting + pd.DateOffset(years=11 + counter),
        ]

        train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
        validate = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]
        test = data[(data["date"] >= cutoff[2]) & (data["date"] < cutoff[3])]

        scaler = StandardScaler().fit(train[stock_vars])
        train[stock_vars] = scaler.transform(train[stock_vars])
        validate[stock_vars] = scaler.transform(validate[stock_vars])
        test[stock_vars] = scaler.transform(test[stock_vars])

        X_train = torch.FloatTensor(train[stock_vars].values)
        Y_train = torch.FloatTensor(train[ret_var].values).unsqueeze(1)
        X_val = torch.FloatTensor(validate[stock_vars].values)
        Y_val = torch.FloatTensor(validate[ret_var].values).unsqueeze(1)
        X_test = torch.FloatTensor(test[stock_vars].values)

        Y_mean = Y_train.mean().item()
        Y_train_dm = Y_train - Y_mean

        reg_pred = test[["year", "month", "date", "permno", ret_var]]

        # # Linear Regression
        # model = LinearRegression(len(stock_vars))
        # train_model(model, X_train, Y_train_dm, device=device)
        # reg_pred["ols"] = predict(model, X_test, device) + Y_mean
        #
        # # Lasso
        # best_mse = float('inf')
        # best_lambda = 0
        # for lambda_ in np.logspace(-4, 4, 81):
        #     model = Lasso(len(stock_vars), alpha=lambda_)
        #     train_model(model, X_train, Y_train_dm, device=device)
        #     val_pred = predict(model, X_val, device) + Y_mean
        #     mse = ((Y_val.numpy() - val_pred) ** 2).mean()
        #     if mse < best_mse:
        #         best_mse = mse
        #         best_lambda = lambda_
        #
        # model = Lasso(len(stock_vars), alpha=best_lambda)
        # train_model(model, X_train, Y_train_dm, device=device)
        # reg_pred["lasso"] = predict(model, X_test, device) + Y_mean
        #
        # # Ridge
        # best_mse = float('inf')
        # best_lambda = 0
        # for lambda_ in np.logspace(-1, 8, 91):
        #     model = Ridge(len(stock_vars), alpha=lambda_ * 0.5)
        #     train_model(model, X_train, Y_train_dm, device=device)
        #     val_pred = predict(model, X_val, device) + Y_mean
        #     mse = ((Y_val.numpy() - val_pred) ** 2).mean()
        #     if mse < best_mse:
        #         best_mse = mse
        #         best_lambda = lambda_
        #
        # model = Ridge(len(stock_vars), alpha=best_lambda * 0.5)
        # train_model(model, X_train, Y_train_dm, device=device)
        # reg_pred["ridge"] = predict(model, X_test, device) + Y_mean
        #
        # # Elastic Net
        # best_mse = float('inf')
        # best_lambda = 0
        # for lambda_ in np.logspace(-4, 4, 81):
        #     model = ElasticNet(len(stock_vars), alpha=lambda_)
        #     train_model(model, X_train, Y_train_dm, device=device)
        #     val_pred = predict(model, X_val, device) + Y_mean
        #     mse = ((Y_val.numpy() - val_pred) ** 2).mean()
        #     if mse < best_mse:
        #         best_mse = mse
        #         best_lambda = lambda_
        #
        # model = ElasticNet(len(stock_vars), alpha=best_lambda)
        # train_model(model, X_train, Y_train_dm, device=device)
        # reg_pred["en"] = predict(model, X_test, device) + Y_mean
        # Train CNN
        cnn_model = CNNRegressor(len(stock_vars))
        train_model(cnn_model, X_train, Y_train_dm, device=device_str)
        reg_pred["cnn"] = predict(cnn_model, X_test, device_str) + Y_mean

        # Train RNN
        rnn_model = RNNRegressor(len(stock_vars), hidden_dim=50)
        train_model(rnn_model, X_train, Y_train_dm, device=device_str)
        reg_pred["rnn"] = predict(rnn_model, X_test, device_str) + Y_mean

        # Train LSTM
        lstm_model = LSTMRegressor(len(stock_vars), hidden_dim=50)
        train_model(lstm_model, X_train, Y_train_dm, device=device_str)
        reg_pred["lstm"] = predict(lstm_model, X_test, device_str) + Y_mean

        pred_out = pred_out._append(reg_pred, ignore_index=True)
        counter += 1

    out_path = os.path.join(work_dir, "output_linux.csv")
    print(out_path)
    pred_out.to_csv(out_path, index=False)

    yreal = pred_out[ret_var].values
    for model_name in ["cnn", "rnn", "lstm"]:
        ypred = pred_out[model_name].values
        r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
        print(model_name, r2)

    print(datetime.datetime.now())
