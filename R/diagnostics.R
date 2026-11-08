# Exploratory analysis and statistical tests
library(ggplot2)
library(tseries)
library(forecast)
library(TTR)
library(lmtest)
library(nortest)

source("config.R")

ensure_fig_dir <- function() {
  if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR, recursive = TRUE)
}

stationarity_table <- function(series, name) {
  s <- na.omit(series)
  adf <- adf.test(s)
  kpss_test <- kpss.test(s, null = "Level")
  return(data.frame(
    series = name,
    adf_stat = adf$statistic,
    adf_p = adf$p.value,
    kpss_stat = kpss_test$statistic,
    kpss_p = kpss_test$p.value
  ))
}

plot_series_panel <- function(df, cols, title, fname) {
  ensure_fig_dir()
  df_long <- data.frame(
    Date = rep(index(df), length(cols)),
    Value = unlist(lapply(cols, function(c) df[, c])),
    Variable = rep(cols, each = nrow(df))
  )
  p <- ggplot(df_long, aes(x = Date, y = Value)) +
    geom_line(linewidth = 0.8) +
    facet_wrap(~ Variable, ncol = 1, scales = "free_y") +
    labs(title = title, x = "Date") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  ggsave(file.path(FIG_DIR, fname), p, width = 11, height = 2.5 * length(cols), dpi = 150)
}

plot_acf_pacf <- function(series, name, diff = 0) {
  ensure_fig_dir()
  s <- na.omit(series)
  if (diff > 0) s <- diff(s, differences = diff)
  tag <- ifelse(diff > 0, "diff", "raw")
  png(file.path(FIG_DIR, paste0("acf_pacf_", name, "_", tag, ".png")),
      width = 11, height = 4, units = "in", res = 150)
  par(mfrow = c(1, 2))
  acf(s, main = paste("ACF -", name, tag), lag.max = 40)
  pacf(s, main = paste("PACF -", name, tag), lag.max = 40)
  dev.off()
}

stl_decompose <- function(series, name, period = 5) {
  ensure_fig_dir()
  s <- na.omit(series)
  ts_obj <- ts(s, frequency = period)
  stl_fit <- stl(ts_obj, s.window = "periodic")
  png(file.path(FIG_DIR, paste0("stl_", name, ".png")),
      width = 11, height = 8, units = "in", res = 150)
  plot(stl_fit, main = paste("STL decomposition -", name))
  dev.off()
}

outlier_iqr_flags <- function(series, window = 60) {
  s <- na.omit(series)
  q1 <- rollapply(s, window, quantile, probs = 0.25, align = "right", fill = NA)
  q3 <- rollapply(s, window, quantile, probs = 0.75, align = "right", fill = NA)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  return((s < lower) | (s > upper))
}

residual_diagnostics <- function(residuals, model_name) {
  r <- na.omit(residuals)
  lb_test <- Box.test(r, lag = 10, type = "Ljung-Box")
  lb_test20 <- Box.test(r, lag = 20, type = "Ljung-Box")
  jb_test <- jarque.bera.test(r)
  arch_test <- ArchTest(r, lags = 10)

  results <- data.frame(
    test = c("Ljung-Box lag10", "Ljung-Box lag20", "Jarque-Bera", "ARCH-LM lag10"),
    stat = c(lb_test$statistic, lb_test20$statistic, jb_test$statistic, arch_test$statistic),
    pvalue = c(lb_test$p.value, lb_test20$p.value, jb_test$p.value, arch_test$p.value),
    model = model_name
  )
  return(results)
}

