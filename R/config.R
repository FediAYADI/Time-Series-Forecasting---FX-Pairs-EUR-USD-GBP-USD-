# Project configuration — finance theme EUR/USD + GBP/USD
library(here)

ROOT <- here()
DATA_DIR <- file.path(ROOT, "data")
FIG_DIR <- file.path(ROOT, "outputs", "figures")
RESULTS_DIR <- file.path(ROOT, "outputs", "results")

TICKERS <- c("EURUSD=X", "GBPUSD=X")
TICKER_LABELS <- c("EURUSD=X" = "EURUSD", "GBPUSD=X" = "GBPUSD")
PRIMARY <- "EURUSD=X"

START_DATE <- "2015-01-01"
END_DATE <- NULL  # today when downloaded

TRAIN_RATIO <- 0.90
FORECAST_H <- 7
RANDOM_SEED <- 42

MA_WINDOW <- 20
REALIZED_VOL_WINDOW <- 20

