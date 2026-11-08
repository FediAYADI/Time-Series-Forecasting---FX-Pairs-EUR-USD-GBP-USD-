"""ARIMA, VAR, cointegration/VECM, GARCH."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from pmdarima import auto_arima
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

warnings.filterwarnings("ignore", category=UserWarning)


def _as_array_indexed(train: pd.Series) -> pd.Series:
    # statsmodels cannot forecast from a date index without a freq;
    # a RangeIndex lets it generate a numeric prediction index.
    return train.reset_index(drop=True) if isinstance(train.index, pd.DatetimeIndex) else train


def fit_arima_manual(train: pd.Series, order=(1, 0, 1)):
    return ARIMA(_as_array_indexed(train), order=order).fit()


def fit_arima_auto(train: pd.Series):
    train = _as_array_indexed(train)
    step = auto_arima(
        train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=5,
        max_q=5,
        d=None,
        trace=False,
    )
    order = step.order
    model = ARIMA(train, order=order)
    return model.fit(), order


def fit_var(train_df: pd.DataFrame, maxlags: int = 15):
    model = VAR(train_df)
    sel = model.select_order(maxlags=maxlags)
    lag = sel.aic
    if lag is None or lag < 1:
        lag = 1
    res = model.fit(lag)
    return res, lag, sel


def granger_tests(data: pd.DataFrame, maxlag: int):
    """Pairwise Granger tests both directions."""
    cols = list(data.columns)
    rows = []
    for i, cause in enumerate(cols):
        for j, effect in enumerate(cols):
            if i == j:
                continue
            test_df = data[[effect, cause]].dropna()
            try:
                res = grangercausalitytests(test_df, maxlag=[maxlag], verbose=False)
                p = res[maxlag][0]["ssr_ftest"][1]
            except Exception:
                p = np.nan
            rows.append({"cause": cause, "effect": effect, "lag": maxlag, "pvalue": p})
    return pd.DataFrame(rows)


def johansen_test(levels: pd.DataFrame, det_order: int = 0):
    return coint_johansen(levels, det_order=det_order, k_ar_diff=1)


def fit_vecm(train_levels: pd.DataFrame, rank: int, k_ar_diff: int = 1):
    return VECM(train_levels, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic="ci").fit()


def fit_garch11(returns: pd.Series):
    scaled = returns.dropna() * 100
    am = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    return am.fit(disp="off")


def forecast_arima(model, steps: int):
    # Use get_forecast which handles the index properly
    forecast_result = model.get_forecast(steps=steps)
    return forecast_result.predicted_mean.values


def forecast_var(model, steps: int, last_obs: pd.DataFrame):
    return model.forecast(last_obs.values[-model.k_ar :], steps=steps)
