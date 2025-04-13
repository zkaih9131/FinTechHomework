import random

import joblib
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from model import StockPriceLSTM
from settings import random_state, ticker


def pre_train():
    """2、进行模型训练预先处理数据"""
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    stock_data = pd.read_csv(f"{ticker}_trend.csv", parse_dates=["date"])
    sentiment_data = pd.read_csv("sentiment_scores_title.csv", parse_dates=["date"])
    merged_data = pd.merge(stock_data, sentiment_data, on="date", how="left")
    merged_data = merged_data.sort_values("date")
    merged_data = merged_data.dropna(subset=["closingValue", "sentiment_score"])
    merged_data["closingValue"] = merged_data["closingValue"].astype(float)

    X_days = 7
    Y_days = 1  # 这里的7天是用前7天预测下一天，可以试着改一下看看效果会不会变好
    cutoff_date = pd.to_datetime("2025-03-01")  # 这里的cutoff_date是在切分训练集和测试集，下面进入评估模式那里还有日期，这些日期不要改，我们尽可能预测不同股票的相同的期间
    train_data = merged_data[merged_data["date"] < cutoff_date].copy()
    closing_values = train_data["closingValue"].values
    sentiment_scores = train_data["sentiment_score"].values

    features, labels = [], []
    for i in range(max(X_days, Y_days), len(train_data)):
        stock_features = closing_values[i - X_days:i]
        sentiment_feature = sentiment_scores[i - Y_days:i]
        if np.any(np.isnan(stock_features)) or np.any(np.isnan(sentiment_feature)) or np.isnan(closing_values[i]):
            continue
        feature_vector = np.concatenate([stock_features, sentiment_feature])
        features.append(feature_vector)
        labels.append(closing_values[i])

    features = np.array(features)
    labels = np.array(labels)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    features_scaled = scaler_X.fit_transform(features)
    labels_scaled = scaler_y.fit_transform(labels.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels_scaled, test_size=0.2, random_state=random_state
    )

    X_train_seq = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_seq = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    return merged_data, scaler_X, scaler_y, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_days, Y_days

def train(device):
    _, scaler_X, scaler_y, x_train, y_train, x_test, y_test, _ = pre_train()
    # 使用wandb自动训练
    wandb.init()
    config = wandb.config
    model = StockPriceLSTM(1, config.hidden_size, config.num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    X_train_dev = x_train.to(device)
    y_train_dev = y_train.to(device)
    X_test_dev = x_test.to(device)
    y_test_dev = y_test.to(device)

    for epoch in range(config.num_epochs):
        model.train()
        output = model(X_train_dev)
        loss = criterion(output, y_train_dev)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_test_dev)
                val_loss = criterion(val_output, y_test_dev)
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": loss.item(),
                    "test_mse": val_loss.item(),
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "lr": config.lr
                })
                print(f"Epoch {epoch+1}: Test MSE = {val_loss.item():.4f}")

    torch.save(model.state_dict(), f"best_model_h{config.hidden_size}_l{config.num_layers}_lr{config.lr}.pth")
    joblib.dump(scaler_X, "scaler_X.save")
    joblib.dump(scaler_y, "scaler_y.save")
    wandb.finish()