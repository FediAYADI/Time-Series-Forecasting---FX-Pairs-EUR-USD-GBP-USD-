# Time Series Forecasting - Foreign Exchange Rates
## EUR/USD and GBP/USD: Classical and Deep Learning Approaches

---

## Project Overview

This project implements and compares classical time series models (ARIMA, VAR, VECM, GARCH) and advanced deep learning models (LSTM, GRU, Hybrid ARIMA+LSTM) for forecasting EUR/USD foreign exchange rates. The project includes implementations in both R and Python to ensure cross-language validation.

**Theme:** Finance - Foreign Exchange Rates
**Assets:** EUR/USD and GBP/USD (daily, ≥5 years from Yahoo Finance)
**Target:** Log returns of EUR/USD
**Forecast Horizon:** 7 days ahead

---

## Project Structure

```
proj/
├── data/
│   └── fx_panel.csv              # Cached FX data (auto-generated)
├── outputs/
│   ├── figures/                  # Diagnostic plots (auto-generated)
│   └── results/                  # Numerical results (auto-generated)
├── python/
│   ├── config.py                 # Configuration
│   ├── data_load.py              # Data loading
│   ├── diagnostics.py            # Statistical tests
│   ├── models_classical.py       # ARIMA, VAR, VECM, GARCH
│   ├── models_dl.py              # LSTM, GRU, hybrid
│   └── run_pipeline.py           # Python pipeline script
├── R/
│   ├── config.R                  # R configuration
│   ├── data_load.R               # R data loading
│   ├── diagnostics.R             # R statistical tests
│   ├── models_classical.R         # R classical models
│   ├── run_pipeline.R            # R pipeline script
│   ├── install_packages.R        # R package installation
│   └── README.md                 # R documentation
├── notebook.ipynb                # Jupyter notebook (main executable)
├── REPORT.md                     # Project report
├── LLM_CONVERSATION.md           # LLM interaction documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Quick Start

### Option 1: Jupyter Notebook (Recommended)

This is the main executable for the project, containing all models (classical and advanced).

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook notebook.ipynb
```

### Option 2: Python Pipeline

```bash
cd python
python run_pipeline.py
```

### Option 3: R Implementation

```r
# Install R packages
source("R/install_packages.R")

# Run R pipeline
source("R/run_pipeline.R")
```

---

## Requirements

### Python Dependencies
- numpy, pandas
- yfinance
- statsmodels
- pmdarima
- arch
- tensorflow/keras
- matplotlib, seaborn
- scikit-learn

Install with:
```bash
pip install -r requirements.txt
```

### R Dependencies
- quantmod
- forecast
- vars
- urca
- rugarch
- tseries
- ggplot2
- lmtest
- nortest
- here
- jsonlite

Install with:
```r
source("R/install_packages.R")
```

---

## Models Implemented

### Classical Models (R and Python)
- **ARIMA**: Manual (1,0,1) and auto-selected using AIC
- **VAR**: Vector autoregression with optimal lag selection
- **VECM**: Vector error correction model with Johansen cointegration test
- **GARCH(1,1)**: Volatility modeling

### Advanced Models (Python/Jupyter only)
- **LSTM**: Long Short-Term Memory network
- **GRU**: Gated Recurrent Unit
- **Hybrid ARIMA+LSTM**: ARIMA on data, LSTM on residuals

---

## Output Files

After running the pipeline, the following files are generated:

### Figures (`outputs/figures/`)
- `01_panel.png` - Time series panel plot
- `acf_pacf_*.png` - ACF/PACF plots
- `stl_*.png` - STL decomposition
- `garch_volatility.png` - GARCH conditional volatility
- `rnn_loss.png` - LSTM/GRU training curves
- `forecast_comparison.png` - Forecast comparison plot

### Results (`outputs/results/`)
- `stationarity.csv` - ADF and KPSS test results
- `outliers.csv` - Outlier detection results
- `granger.csv` - Granger causality tests
- `johansen.csv` - Johansen cointegration test
- `residual_diagnostics.csv` - Model residual diagnostics
- `forecast_metrics.csv` - Model performance metrics
- `diebold_mariano.csv` - Diebold-Mariano test results
- `summary.json` - Summary statistics

---

## Documentation

- **REPORT.md** - Comprehensive project report with:
  - Methodology and mathematical formulations
  - Model descriptions
  - Results and discussion (to be filled after running models)
  - References

- **LLM_CONVERSATION.md** - Documentation of LLM (Claude) assistance
  - Prompts and responses
  - Implementation decisions
  - Code generation statistics

- **R/README.md** - R implementation documentation

---

## Key Features

1. **Cross-Language Validation**: Classical models implemented in both R and Python
2. **Comprehensive Model Suite**: Linear, multivariate, cointegration, volatility, and deep learning models
3. **Statistical Rigor**: Multiple diagnostic tests and evaluation metrics
4. **Reproducibility**: Seed setting, clear configuration, modular code
5. **Educational Focus**: Detailed documentation and Jupyter notebook with explanations

---

## Evaluation Metrics

- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)

Statistical tests include:
- ADF and KPSS (stationarity)
- Ljung-Box (autocorrelation)
- Jarque-Bera (normality)
- ARCH-LM (heteroskedasticity)
- Granger causality
- Johansen cointegration
- Diebold-Mariano (forecast comparison)

---

## Next Steps

1. **Run the Jupyter notebook** to generate all results
2. **Fill in results tables** in REPORT.md with actual model outputs
3. **Compare R and Python results** for classical models
4. **Write discussion sections** in REPORT.md based on results
5. **Generate final figures** and update report accordingly

---

## Notes

- Data is automatically downloaded from Yahoo Finance on first run
- Cached data is used on subsequent runs for faster execution
- The project uses a 90/10 train/test split (chronological)
- Random seed is set to 42 for reproducibility
- Forecast horizon is 7 days by default (configurable)

---

## References

- Box, G. E. P., & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control.
- Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy.

---

## License

This project is for educational purposes.

---

**Generated with assistance from Claude (Anthropic) - documented in LLM_CONVERSATION.md**
