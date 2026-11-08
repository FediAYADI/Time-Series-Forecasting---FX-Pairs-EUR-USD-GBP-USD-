# ARIMA, VAR, cointegration/VECM, GARCH
library(forecast)
library(vars)
library(urca)
library(rugarch)
library(tseries)

suppressWarnings(suppressMessages(library(dplyr)))

fit_arima_manual <- function(train, order = c(1, 0, 1)) {
  return(Arima(train, order = order))
}

fit_arima_auto <- function(train) {
  auto_fit <- auto.arima(
    train,
    seasonal = FALSE,
    stepwise = TRUE,
    approximation = FALSE,
    max.p = 5,
    max.q = 5,
    trace = FALSE
  )
  order <- arimaorder(auto_fit)
  return(list(model = auto_fit, order = order))
}

fit_var <- function(train_df, maxlags = 15) {
  train_df <- as.data.frame(train_df)
  var_select <- VARselect(train_df, lag.max = maxlags, type = "const")
  lag <- var_select$selection["AIC(n)"]
  if (is.na(lag) || lag < 1) lag <- 1
  var_fit <- VAR(train_df, p = lag, type = "const")
  return(list(model = var_fit, lag = lag, selection = var_select))
}

granger_tests <- function(data, maxlag) {
  # Pairwise Granger tests both directions
  cols <- colnames(data)
  results <- data.frame()
  for (i in seq_along(cols)) {
    for (j in seq_along(cols)) {
      if (i == j) next
      cause <- cols[i]
      effect <- cols[j]
      test_df <- data[, c(effect, cause)]
      test_df <- na.omit(test_df)
      tryCatch({
        granger <- grangertest(test_df[, 2] ~ test_df[, 1], order = maxlag)
        pvalue <- granger$`Pr(>F)`[2]
        results <- rbind(results, data.frame(
          cause = cause,
          effect = effect,
          lag = maxlag,
          pvalue = pvalue
        ))
      }, error = function(e) {
        results <<- rbind(results, data.frame(
          cause = cause,
          effect = effect,
          lag = maxlag,
          pvalue = NA
        ))
      })
    }
  }
  return(results)
}

johansen_test <- function(levels, det_order = 0) {
  joh <- ca.jo(levels, type = "trace", ecdet = "const", K = 1)
  return(joh)
}

fit_vecm <- function(train_levels, rank, k_ar_diff = 1) {
  vecm_fit <- cajorls(ca.jo(train_levels, type = "trace", ecdet = "const", K = 1), r = rank)
  return(vecm_fit)
}

fit_garch11 <- function(returns) {
  scaled <- returns * 100
  scaled <- na.omit(scaled)
  spec <- ugarchspec(
    mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
    variance.model = list(garchOrder = c(1, 1)),
    distribution.model = "norm"
  )
  garch_fit <- ugarchfit(spec = spec, data = scaled, solver = "hybrid")
  return(garch_fit)
}

forecast_arima <- function(model, steps) {
  return(forecast(model, h = steps)$mean)
}

forecast_var <- function(model, steps, last_obs) {
  return(predict(model, n.ahead = steps)$fcst)
}

