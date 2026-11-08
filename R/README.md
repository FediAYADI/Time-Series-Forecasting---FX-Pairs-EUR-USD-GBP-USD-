# R Implementation - Classical Time Series Models

This directory contains the R implementation of classical time series models for the FX forecasting project.

## Files

- `config.R` - Project configuration (tickers, dates, parameters)
- `data_load.R` - Download and prepare Yahoo Finance FX data
- `diagnostics.R` - Exploratory analysis and statistical tests
- `models_classical.R` - ARIMA, VAR, VECM, GARCH models
- `run_pipeline.R` - Main pipeline script
- `install_packages.R` - Install required R packages

## Setup

1. Install R and RStudio if not already installed
2. Run the package installation script:
   ```r
   source("install_packages.R")
   ```

## Running the Pipeline

From the project root directory:
```r
setwd("D:/STUDY/Time series/third yerar/proj/R")
source("run_pipeline.R")
```

Or from within R:
```r
source("R/run_pipeline.R")
```

## Output

The pipeline will create:
- `data/fx_panel.csv` - Cached FX data
- `outputs/figures/` - Diagnostic plots
- `outputs/results/` - CSV and JSON results

## Models Implemented

- **ARIMA**: Manual (1,0,1) and auto-selected
- **VAR**: Vector autoregression with optimal lag selection
- **VECM**: Vector error correction model with Johansen cointegration test
- **GARCH(1,1)**: Volatility modeling

## Notes

- This R implementation mirrors the Python implementation in `python/`
- Both implementations should produce comparable results for the classical models
- Advanced deep learning models (LSTM, GRU, hybrid) are in the Python Jupyter notebook
