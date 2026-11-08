# Install required R packages for the time series project
required_packages <- c(
  "quantmod",      # Yahoo Finance data download
  "zoo",           # Time series manipulation
  "TTR",           # Technical trading rules (moving averages)
  "forecast",      # ARIMA and forecasting functions
  "vars",          # VAR models
  "urca",          # Unit root tests and cointegration (Johansen)
  "rugarch",       # GARCH models
  "tseries",       # Time series tests (ADF, KPSS, etc.)
  "ggplot2",       # Plotting
  "lmtest",        # Lagrange multiplier tests
  "nortest",       # Normality tests
  "here",          # File path management
  "jsonlite"       # JSON output
)

# Install packages if not already installed
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
    library(pkg, character.only = TRUE)
  } else {
    library(pkg, character.only = TRUE)
  }
}

print("All required packages installed/loaded successfully.")

