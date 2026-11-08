# End-to-end mini-project pipeline — Finance FX (EUR/USD, GBP/USD)
library(here)
library(jsonlite)

source("config.R")
source("data_load.R")
source("diagnostics.R")
source("models_classical.R")

# Create output directories
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR, recursive = TRUE)
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR, recursive = TRUE)

# Load data
df <- load_or_download(START_DATE, END_DATE)
eur_label <- TICKER_LABELS[PRIMARY]
ret_col <- paste0(eur_label, "_log_return")
gbp_label <- TICKER_LABELS[TICKERS[2]]

# Plot panel
plot_series_panel(
  df,
  c(paste0(eur_label, "_Close"), paste0(gbp_label, "_Close"), ret_col, paste0(eur_label, "_realized_vol")),
  "EUR/USD & GBP/USD — levels, returns, realized volatility",
  "01_panel.png"
)

# Stationarity tests
stat_rows <- rbind(
  stationarity_table(df[[paste0(eur_label, "_Close")]], "EURUSD Close"),
  stationarity_table(df[[ret_col]], "EURUSD log return"),
  stationarity_table(df[[paste0(gbp_label, "_Close")]], "GBPUSD Close")
)
write.csv(stat_rows, file.path(RESULTS_DIR, "stationarity.csv"), row.names = FALSE)

# ACF/PACF plots
plot_acf_pacf(df[[ret_col]], "eur_return", diff = 0)
plot_acf_pacf(df[[ret_col]], "eur_return", diff = 1)

# STL decomposition
stl_decompose(df[[ret_col]], "eur_log_return", period = 5)

# Outlier detection
outliers <- outlier_iqr_flags(df[[ret_col]])
outlier_rate <- mean(outliers, na.rm = TRUE)
write.csv(data.frame(outlier_rate_iqr = outlier_rate), file.path(RESULTS_DIR, "outliers.csv"), row.names = FALSE)

# Train/test split
y <- df[[ret_col]]
split_result <- train_test_split_ts(y, TRAIN_RATIO)
y_train <- split_result$train
y_test <- split_result$test
h <- min(FORECAST_H, length(y_test))

# VAR data
var_cols <- c(ret_col, paste0(gbp_label, "_log_return"))
var_train <- df[index(y_train), var_cols]
var_test <- df[index(y_test), var_cols]

# ARIMA manual vs auto
arima_man <- fit_arima_manual(y_train, order = c(1, 0, 1))
arima_auto_res <- fit_arima_auto(y_train)
arima_auto <- arima_auto_res$model
auto_order <- arima_auto_res$order

diag_frames <- rbind(
  residual_diagnostics(residuals(arima_man), "ARIMA(1,0,1)"),
  residual_diagnostics(residuals(arima_auto), paste0("auto.arima ", paste(auto_order, collapse = ",")))
)

fc_man <- forecast_arima(arima_man, h)
fc_auto <- forecast_arima(arima_auto, h)
y_true <- as.numeric(y_test[1:h])

# VAR
var_res <- fit_var(var_train)
var_model <- var_res$model
var_lag <- var_res$lag
fc_var <- forecast_var(var_model, h, tail(var_train, var_lag))
fc_var_eur <- fc_var[[1]][, 1]  # EUR leg

granger <- granger_tests(var_train, var_lag)
write.csv(granger, file.path(RESULTS_DIR, "granger.csv"), row.names = FALSE)

# Johansen / VECM on log levels
level_cols <- c(paste0(eur_label, "_Close"), paste0(gbp_label, "_Close"))
levels_train <- df[index(y_train), level_cols]
levels_test <- df[index(y_test), level_cols]

joh <- johansen_test(levels_train)
rank <- sum(joh@teststat > joh@cval[, 2])
rank <- max(rank, 1)

vecm_res <- fit_vecm(levels_train, rank = min(rank, 1))
# VECM forecast requires reconstruction
vecm_fc <- predict(vecm_res, n.ahead = h)$fcst
vecm_eur_level <- vecm_fc[[1]][, 1]
vecm_eur_ret_fc <- diff(c(as.numeric(tail(levels_train[, 1], 1)), vecm_eur_level[1:h]))

johansen_df <- data.frame(
  trace_stat = joh@teststat,
  crit_95 = joh@cval[, 2],
  eigen_stat = joh@teststat[2]
)
write.csv(johansen_df, file.path(RESULTS_DIR, "johansen.csv"), row.names = FALSE)

# GARCH
garch <- fit_garch11(y_train)
garch_std <- residuals(garch) / sigma(garch)
diag_frames <- rbind(diag_frames, residual_diagnostics(garch_std, "GARCH(1,1) std resid"))

garch_fc <- ugarchforecast(garch, n.ahead = h)
fc_garch_mean <- as.numeric(fitted(garch_fc)) / 100

# Plot GARCH conditional volatility
cond_vol <- sigma(garch) / 100
png(file.path(FIG_DIR, "garch_volatility.png"),
    width = 11, height = 4, units = "in", res = 150)
plot(tail(index(y_train), 500), tail(cond_vol, 500),
     type = "l", col = "darkred", lwd = 0.8,
     main = "GARCH(1,1) conditional volatility — EUR/USD returns (last 500 obs)",
     ylab = "Volatility", xlab = "Date")
dev.off()

# Save residual diagnostics
write.csv(diag_frames, file.path(RESULTS_DIR, "residual_diagnostics.csv"), row.names = FALSE)

# Metrics calculation
calculate_metrics <- function(y_true, y_pred) {
  e <- y_true - y_pred
  mse <- mean(e^2)
  mae <- mean(abs(e))
  rmse <- sqrt(mse)
  mape <- mean(abs(e / (y_true + 1e-12))) * 100
  return(c(MSE = mse, RMSE = rmse, MAE = mae, MAPE = mape))
}

# Collect forecasts and calculate metrics
results <- data.frame()
model_names <- c(
  "ARIMA(1,0,1)",
  paste0("auto.arima", paste(auto_order, collapse = ",")),
  paste0("VAR(", var_lag, ") EUR leg"),
  "VECM implied return",
  "GARCH mean"
)
forecasts <- list(fc_man, fc_auto, fc_var_eur, vecm_eur_ret_fc[1:h], fc_garch_mean)

for (i in seq_along(model_names)) {
  pred <- as.numeric(forecasts[[i]])[1:h]
  m <- calculate_metrics(y_true, pred)
  results <- rbind(results, data.frame(
    model = model_names[i],
    MSE = m["MSE"],
    RMSE = m["RMSE"],
    MAE = m["MAE"],
    MAPE = m["MAPE"]
  ))
}

write.csv(results, file.path(RESULTS_DIR, "forecast_metrics.csv"), row.names = FALSE)

# Summary
summary <- list(
  n_obs = nrow(df),
  train_n = length(y_train),
  test_n = length(y_test),
  forecast_h = h,
  auto_arima_order = as.list(auto_order),
  var_lag_aic = var_lag,
  johansen_rank_used = min(rank, 1),
  best_model_by_rmse = results[which.min(results$RMSE), "model"]
)
write_json(summary, file.path(RESULTS_DIR, "summary.json"), auto_unbox = TRUE, pretty = TRUE)

# Plot forecast comparison
png(file.path(FIG_DIR, "forecast_comparison.png"),
    width = 11, height = 5, units = "in", res = 150)
plot(index(y_test)[1:h], y_true, type = "b", pch = 16, col = "black",
     main = paste("Out-of-sample forecasts (h=", h, ") — EUR/USD log returns"),
     ylab = "Log Return", xlab = "Date", lwd = 1)
lines(index(y_test)[1:h], fc_auto[1:h], col = "blue", lwd = 1)
legend("topright", legend = c("Actual", "auto.arima"),
       col = c("black", "blue"), lty = c(1, 1), pch = c(16, NA))
dev.off()

print("Pipeline finished.")
print(results)
print("Summary:")
print(jsonlite::toJSON(summary, auto_unbox = TRUE, pretty = TRUE))

