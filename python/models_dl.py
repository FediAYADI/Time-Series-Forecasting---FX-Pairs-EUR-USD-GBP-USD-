"""LSTM, GRU, hybrid ARIMA+LSTM, metrics, Diebold-Mariano."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout

from config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_supervised(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])
    return np.array(X), np.array(y)


def build_rnn(kind: str, window: int, units: int = 32, dropout: float = 0.1):
    set_seed()
    layer = LSTM if kind == "lstm" else GRU
    model = Sequential(
        [
            layer(units, input_shape=(window, 1), return_sequences=False),
            Dropout(dropout),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_rnn(series: pd.Series, window: int = 30, kind: str = "lstm", epochs: int = 25, val_ratio: float = 0.1):
    """Train on scaled returns; last val_ratio for monitoring."""
    s = series.dropna().values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(s).flatten()
    X, y = make_supervised(scaled, window)
    n_val = max(int(len(X) * val_ratio), 1)
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]
    X_train = X_train.reshape(-1, window, 1)
    X_val = X_val.reshape(-1, window, 1)
    model = build_rnn(kind, window)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0,
    )
    return model, scaler, window, history


def forecast_rnn_multistep(model, scaler, window: int, history: np.ndarray, steps: int):
    """Recursive multi-step forecast in scaled space."""
    buf = list(history[-window:])
    preds = []
    for _ in range(steps):
        x = np.array(buf[-window:]).reshape(1, window, 1)
        p = model.predict(x, verbose=0)[0, 0]
        preds.append(p)
        buf.append(p)
    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()


def hybrid_arima_lstm(train: pd.Series, window: int = 30, arima_order=(1, 0, 1), epochs: int = 20):
    """ARIMA on train; LSTM on in-sample ARIMA residuals."""
    # Fit on a RangeIndex: statsmodels cannot forecast from a freq-less date index
    arima_train = train.reset_index(drop=True) if isinstance(train.index, pd.DatetimeIndex) else train
    arima = ARIMA(arima_train, order=arima_order).fit()
    resid = pd.Series(np.asarray(arima.resid), index=train.index).dropna()
    aligned_train = train.loc[resid.index]
    lstm_model, scaler, window, hist = train_rnn(resid, window=window, kind="lstm", epochs=epochs)
    return arima, lstm_model, scaler, window, resid, hist


def hybrid_forecast(arima, lstm_model, scaler, window, train, steps: int):
    arima_fc = arima.forecast(steps=steps)
    resid_hist = pd.Series(np.asarray(arima.resid), index=train.index).dropna().values
    scaled_hist = scaler.transform(resid_hist.reshape(-1, 1)).flatten()
    resid_fc = forecast_rnn_multistep(lstm_model, scaler, window, scaled_hist, steps)
    return arima_fc + resid_fc


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    e = y_true - y_pred
    mse = float(np.mean(e**2))
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(e / (y_true + 1e-12))) * 100)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1):
    """Simple DM test on squared errors (loss differential)."""
    d = e1**2 - e2**2
    d = d - d.mean()
    n = len(d)
    gamma0 = np.dot(d, d) / n
    var_d = gamma0
    for k in range(1, h):
        gamma_k = np.dot(d[k:], d[:-k]) / n
        var_d += 2 * gamma_k
    stat = d.mean() / np.sqrt(var_d / n + 1e-12)
    from scipy.stats import norm

    p = 2 * (1 - norm.cdf(abs(stat)))
    return float(stat), float(p)


def time_series_cv_rnn(series: pd.Series, window: int = 30, kind: str = "lstm", n_splits: int = 3):
    """Expanding-window CV: report mean RMSE."""
    s = series.dropna()
    fold_size = len(s) // (n_splits + 1)
    rmses = []
    for k in range(1, n_splits + 1):
        end = fold_size * (k + 1)
        train = s.iloc[: end - fold_size]
        test = s.iloc[end - fold_size : end]
        if len(test) <= window + 5:
            continue
        model, scaler, w, _ = train_rnn(train, window=window, kind=kind, epochs=15)
        hist = scaler.transform(train.values.reshape(-1, 1)).flatten()
        pred = forecast_rnn_multistep(model, scaler, w, hist, len(test))
        rmses.append(metrics(test.values, pred)["RMSE"])
    return float(np.mean(rmses)) if rmses else np.nan
