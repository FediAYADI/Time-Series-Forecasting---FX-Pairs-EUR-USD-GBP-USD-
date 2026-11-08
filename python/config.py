"""Project configuration — finance theme EUR/USD + GBP/USD."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
RESULTS_DIR = ROOT / "outputs" / "results"

TICKERS = ["EURUSD=X", "GBPUSD=X"]
TICKER_LABELS = {"EURUSD=X": "EURUSD", "GBPUSD=X": "GBPUSD"}
PRIMARY = "EURUSD=X"

START_DATE = "2015-01-01"
END_DATE = None  # today when downloaded

TRAIN_RATIO = 0.90
FORECAST_H = 7
RANDOM_SEED = 42

MA_WINDOW = 20
REALIZED_VOL_WINDOW = 20
