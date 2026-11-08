"""
End-to-end mini-project pipeline — Finance FX (EUR/USD, GBP/USD).
Run: python run_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    END_DATE,
    FIG_DIR,
    FORECAST_H,
    PRIMARY,
    RESULTS_DIR,
    START_DATE,
    TICKER_LABELS,
    TICKERS,
    TRAIN_RATIO,
)
from data_load import load_or_download, train_test_split_ts
from diagnostics import (
    outlier_iqr_flags,
    plot_acf_pacf,
    plot_series_panel,
    residual_diagnostics,
    stationarity_table,
    stl_decompose,
)
from models_classical import (
    fit_arima_auto,
    fit_arima_manual,
    fit_garch11,
    fit_var,
    fit_vecm,
    forecast_arima,
    forecast_var,
    granger_tests,
    johansen_test,
)
from models_dl import (
    hybrid_arima_lstm,
    hybrid_forecast,
    metrics,
    diebold_mariano,
    time_series_cv_rnn,
    train_rnn,
    forecast_rnn_multistep,
)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_download(START_DATE, END_DATE)
    eur_label = TICKER_LABELS[PRIMARY]
    ret_col = f"{eur_label}_log_return"
    gbp_label = TICKER_LABELS[TICKERS[1]]

    plot_series_panel(
        df,
        [f"{eur_label}_Close", f"{gbp_label}_Close", ret_col, f"{eur_label}_realized_vol"],
        "EUR/USD & GBP/USD — levels, returns, realized volatility",
        "01_panel.png",
    )

    stat_rows = [
        stationarity_table(df[f"{eur_label}_Close"], "EURUSD Close"),
        stationarity_table(df[ret_col], "EURUSD log return"),
        stationarity_table(df[f"{gbp_label}_Close"], "GBPUSD Close"),
    ]
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(RESULTS_DIR / "stationarity.csv", index=False)

    plot_acf_pacf(df[ret_col], "eur_return", diff=0)
    plot_acf_pacf(df[ret_col], "eur_return", diff=1)
    stl_decompose(df[ret_col], "eur_log_return", period=5)

    outliers = outlier_iqr_flags(df[ret_col])
    outlier_rate = outliers.mean()
    pd.Series({"outlier_rate_iqr": outlier_rate}).to_csv(RESULTS_DIR / "outliers.csv")

    y = df[ret_col]
    y_train, y_test = train_test_split_ts(y, TRAIN_RATIO)
    h = min(FORECAST_H, len(y_test))

    var_cols = [ret_col, f"{gbp_label}_log_return"]
    var_train = df.loc[y_train.index, var_cols]
    var_test = df.loc[y_test.index, var_cols]

    # --- ARIMA manual vs auto ---
    arima_man = fit_arima_manual(y_train, order=(1, 0, 1))
    arima_auto, auto_order = fit_arima_auto(y_train)
    diag_frames = [
        residual_diagnostics(arima_man.resid, "ARIMA(1,0,1)"),
        residual_diagnostics(arima_auto.resid, f"auto.arima {auto_order}"),
    ]

    fc_man = forecast_arima(arima_man, h)
    fc_auto = forecast_arima(arima_auto, h)
    y_true = y_test.iloc[:h].values

    # --- VAR ---
    var_res, var_lag, var_sel = fit_var(var_train)
    fc_var = forecast_var(var_res, h, var_train)
    fc_var_eur = fc_var[:, 0]
    granger = granger_tests(var_train, maxlag=var_lag)
    granger.to_csv(RESULTS_DIR / "granger.csv", index=False)

    # --- Johansen / VECM on log levels ---
    level_cols = [f"{eur_label}_Close", f"{gbp_label}_Close"]
    levels_train = df.loc[y_train.index, level_cols]
    levels_test = df.loc[y_test.index, level_cols]
    joh = johansen_test(levels_train)
    rank = sum(joh.lr1 > joh.cvt[:, 1])
    rank = max(rank, 1)
    vecm_res = fit_vecm(levels_train, rank=min(rank, 1))
    vecm_fc = vecm_res.predict(steps=h)
    vecm_eur_level = vecm_fc[:, 0]
    vecm_eur_ret_fc = np.diff(np.log(np.r_[levels_train.iloc[-1, 0], vecm_eur_level[:h]]))

    pd.DataFrame(
        {
            "trace_stat": joh.lr1,
            "crit_95": joh.cvt[:, 1],
            "eigen_stat": joh.lr2,
        }
    ).to_csv(RESULTS_DIR / "johansen.csv")

    # --- GARCH ---
    garch = fit_garch11(y_train)
    garch_std = garch.resid / garch.conditional_volatility
    diag_frames.append(residual_diagnostics(garch_std, "GARCH(1,1) std resid"))
    garch_fc = garch.forecast(horizon=h)
    fc_garch_mean = garch_fc.mean.values[-1] / 100

    cond_vol = garch.conditional_volatility / 100
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y_train.index[-500:], cond_vol.iloc[-500:], color="darkred", lw=0.8)
    ax.set_title("GARCH(1,1) conditional volatility — EUR/USD returns (last 500 obs)")
    ax.set_ylabel("Volatility")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "garch_volatility.png", dpi=150)
    plt.close(fig)

    # --- LSTM / GRU ---
    lstm_model, lstm_scaler, window, lstm_hist = train_rnn(y_train, window=30, kind="lstm", epochs=25)
    gru_model, gru_scaler, _, gru_hist = train_rnn(y_train, window=30, kind="gru", epochs=25)
    cv_lstm = time_series_cv_rnn(y_train, window=30, kind="lstm", n_splits=3)
    cv_gru = time_series_cv_rnn(y_train, window=30, kind="gru", n_splits=3)

    scaled_hist = lstm_scaler.transform(y_train.values.reshape(-1, 1)).flatten()
    fc_lstm = forecast_rnn_multistep(lstm_model, lstm_scaler, window, scaled_hist, h)
    scaled_hist_g = gru_scaler.transform(y_train.values.reshape(-1, 1)).flatten()
    fc_gru = forecast_rnn_multistep(gru_model, gru_scaler, window, scaled_hist_g, h)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lstm_hist.history["loss"], label="LSTM train")
    ax.plot(lstm_hist.history["val_loss"], label="LSTM val")
    ax.plot(gru_hist.history["loss"], label="GRU train", alpha=0.7)
    ax.plot(gru_hist.history["val_loss"], label="GRU val", alpha=0.7)
    ax.legend()
    ax.set_title("RNN training curves")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rnn_loss.png", dpi=150)
    plt.close(fig)

    # --- Hybrid ARIMA + LSTM ---
    arima_h, lstm_h, sc_h, w_h, _, _ = hybrid_arima_lstm(y_train, window=30, arima_order=(1, 0, 1), epochs=20)
    fc_hybrid = hybrid_forecast(arima_h, lstm_h, sc_h, w_h, y_train, h)

    diag_all = pd.concat(diag_frames, ignore_index=True)
    diag_all.to_csv(RESULTS_DIR / "residual_diagnostics.csv", index=False)

    results = []
    for name, pred in [
        ("ARIMA(1,0,1)", fc_man),
        (f"auto.arima{auto_order}", fc_auto),
        (f"VAR({var_lag}) EUR leg", fc_var_eur),
        ("VECM implied return", vecm_eur_ret_fc[:h]),
        ("GARCH mean", fc_garch_mean),
        ("LSTM", fc_lstm),
        ("GRU", fc_gru),
        ("Hybrid ARIMA+LSTM", fc_hybrid),
    ]:
        pred = np.asarray(pred)[:h]
        m = metrics(y_true, pred)
        m["model"] = name
        results.append(m)

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(RESULTS_DIR / "forecast_metrics.csv", index=False)

    e_arima = y_true - np.asarray(fc_auto)[:h]
    e_lstm = y_true - fc_lstm[:h]
    dm_stat, dm_p = diebold_mariano(e_arima, e_lstm, h=1)
    pd.Series({"dm_stat_auto_vs_lstm": dm_stat, "pvalue": dm_p}).to_csv(
        RESULTS_DIR / "diebold_mariano.csv"
    )

    summary = {
        "n_obs": len(df),
        "train_n": len(y_train),
        "test_n": len(y_test),
        "forecast_h": h,
        "auto_arima_order": list(auto_order),
        "var_lag_aic": int(var_lag),
        "johansen_rank_used": int(min(rank, 1)),
        "ts_cv_rmse_lstm": cv_lstm,
        "ts_cv_rmse_gru": cv_gru,
        "best_model_by_rmse": metrics_df.sort_values("RMSE").iloc[0]["model"],
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(y_test.index[:h], y_true, label="Actual", marker="o", ms=4)
    ax.plot(y_test.index[:h], fc_auto[:h], label="auto.arima", alpha=0.8)
    ax.plot(y_test.index[:h], fc_lstm[:h], label="LSTM", alpha=0.8)
    ax.plot(y_test.index[:h], fc_hybrid[:h], label="Hybrid", alpha=0.8)
    ax.legend()
    ax.set_title(f"Out-of-sample forecasts (h={h}) — EUR/USD log returns")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecast_comparison.png", dpi=150)
    plt.close(fig)

    print("Pipeline finished.")
    print(metrics_df.to_string(index=False))
    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
