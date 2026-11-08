# Download and prepare Yahoo Finance FX data
library(quantmod)
library(zoo)
library(TTR)

source("config.R")

download_fx <- function(start, end = NULL) {
  # Pull adjusted OHLCV for EUR/USD and GBP/USD; align on business days
  frames <- list()
  for (t in TICKERS) {
    symbol <- gsub("=X", "", t)
    getSymbols(symbol, src = "yahoo", from = start, to = end, auto.assign = FALSE)
    raw <- get(symbol)
    colnames(raw) <- paste0(TICKER_LABELS[t], "_", colnames(raw))
    frames[[t]] <- raw
  }
  df <- do.call(merge, frames)
  df <- na.omit(df)
  index(df) <- as.Date(index(df))
  return(df)
}

add_features <- function(df) {
  # Log returns, moving average, and realized volatility for primary pair
  label <- TICKER_LABELS[PRIMARY]
  close <- df[, paste0(label, "_Close")]
  out <- df
  out[[paste0(label, "_log_return")]] <- diff(log(close))
  out[[paste0(label, "_ma", MA_WINDOW)]] <- runMean(close, MA_WINDOW)
  realized_vol <- rollapply(
    out[[paste0(label, "_log_return")]],
    REALIZED_VOL_WINDOW,
    function(x) sd(x) * sqrt(252),
    align = "right",
    fill = NA
  )
  out[[paste0(label, "_realized_vol")]] <- realized_vol

  for (t in TICKERS) {
    if (t == PRIMARY) next
    lb <- TICKER_LABELS[t]
    out[[paste0(lb, "_log_return")]] <- diff(log(out[[paste0(lb, "_Close")]]))
  }
  return(na.omit(out))
}

train_test_split_ts <- function(series, train_ratio) {
  # Chronological split — no shuffle
  n <- length(series)
  split <- floor(n * train_ratio)
  train <- series[1:split]
  test <- series[(split + 1):n]
  return(list(train = train, test = test))
}

cache_data <- function(df, name = "fx_panel.csv") {
  if (!dir.exists(DATA_DIR)) dir.create(DATA_DIR, recursive = TRUE)
  path <- file.path(DATA_DIR, name)
  write.zoo(df, path, index.name = "Date")
  return(df)
}

load_or_download <- function(start, end = NULL) {
  path <- file.path(DATA_DIR, "fx_panel.csv")
  if (file.exists(path)) {
    df <- read.zoo(path, header = TRUE, index.column = 1, FUN = as.Date)
    if (nrow(df) > 500) return(df)
  }
  df <- download_fx(start, end)
  df <- add_features(df)
  return(cache_data(df))
}

