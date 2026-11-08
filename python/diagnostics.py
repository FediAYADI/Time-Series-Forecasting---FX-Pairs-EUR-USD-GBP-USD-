"""Exploratory analysis and statistical tests."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from config import FIG_DIR


def ensure_fig_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def stationarity_table(series: pd.Series, name: str) -> dict:
    """ADF and KPSS — opposite nulls; report both."""
    s = series.dropna()
    adf = adfuller(s, autolag="AIC")
    with np.errstate(all="ignore"):
        kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
    return {
        "series": name,
        "adf_stat": adf[0],
        "adf_p": adf[1],
        "kpss_stat": kpss_stat,
        "kpss_p": kpss_p,
    }


def plot_series_panel(df: pd.DataFrame, cols: list[str], title: str, fname: str):
    ensure_fig_dir()
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 2.5 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, c in zip(axes, cols):
        ax.plot(df.index, df[c], linewidth=0.8)
        ax.set_ylabel(c)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def plot_acf_pacf(series: pd.Series, name: str, diff: int = 0):
    ensure_fig_dir()
    s = series.dropna()
    if diff:
        s = s.diff(diff).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_acf(s, ax=axes[0], lags=40)
    plot_pacf(s, ax=axes[1], lags=40, method="ywm")
    tag = "diff" if diff else "raw"
    fig.suptitle(f"ACF/PACF — {name} ({tag})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"acf_pacf_{name}_{tag}.png", dpi=150)
    plt.close(fig)


def stl_decompose(series: pd.Series, name: str, period: int = 5):
    """STL on business-day FX; period=5 ≈ weekly structure."""
    ensure_fig_dir()
    s = series.dropna()
    stl = STL(s, period=period, robust=True).fit()
    fig = stl.plot()
    fig.set_size_inches(11, 8)
    fig.suptitle(f"STL decomposition — {name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"stl_{name}.png", dpi=150)
    plt.close(fig)


def outlier_iqr_flags(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling IQR rule for extreme returns."""
    s = series.dropna()
    q1 = s.rolling(window).quantile(0.25)
    q3 = s.rolling(window).quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (s < lower) | (s > upper)


def residual_diagnostics(residuals: pd.Series, model_name: str) -> pd.DataFrame:
    """Ljung-Box, normality, ARCH-LM on model residuals."""
    r = residuals.dropna()
    lb = acorr_ljungbox(r, lags=[10, 20], return_df=True)
    jb_stat, jb_p = stats.jarque_bera(r)
    arch_lm = het_arch(r, nlags=10)
    rows = [
        {"test": "Ljung-Box lag10", "stat": lb["lb_stat"].iloc[0], "pvalue": lb["lb_pvalue"].iloc[0]},
        {"test": "Ljung-Box lag20", "stat": lb["lb_stat"].iloc[1], "pvalue": lb["lb_pvalue"].iloc[1]},
        {"test": "Jarque-Bera", "stat": jb_stat, "pvalue": jb_p},
        {"test": "ARCH-LM lag10", "stat": arch_lm[0], "pvalue": arch_lm[1]},
    ]
    out = pd.DataFrame(rows)
    out["model"] = model_name
    return out
