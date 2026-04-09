"""
Local ML predictor — trains on historical data to predict price direction.

Uses scikit-learn (lightweight, no GPU needed) to train a model that predicts
whether a stock will go UP or DOWN in the next trading session, based on
technical indicators.

The model improves over time as more data is collected from paper trading.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import config
from strategy import add_indicators
from data_fetcher import get_historical_data

MODEL_DIR = os.path.join(config.PROJECT_DIR, "models")
TRAINING_LOG = os.path.join(config.PROJECT_DIR, "training_log.json")


def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature matrix from OHLCV data."""
    df = add_indicators(df).copy()

    # Price-based features
    df["return_1d"] = df["Close"].pct_change()
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)
    df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["close_vs_high"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-8)

    # Volatility
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # Volume features
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["volume_change"] = df["Volume"].pct_change()

    # EMA features
    df["ema_diff"] = (df["ema_short"] - df["ema_long"]) / df["Close"]
    df["price_vs_ema9"] = (df["Close"] - df["ema_short"]) / df["Close"]
    df["price_vs_ema21"] = (df["Close"] - df["ema_long"]) / df["Close"]

    # RSI features
    df["rsi_change"] = df["rsi"].diff()
    
    # New features (MACD, BB, ATR)
    df["macd_norm"] = df["macd"] / df["Close"]
    df["macd_diff_norm"] = df["macd_diff"] / df["Close"]
    df["price_vs_bb_mid"] = (df["Close"] - df["bb_mid"]) / df["Close"]
    df["atr_norm"] = df["atr"] / df["Close"]

    # Target: will price go up by more than 0.3% tomorrow? (noise filtering)
    df["target"] = (df["Close"].shift(-1) > df["Close"] * 1.003).astype(int)

    return df


FEATURE_COLS = [
    "rsi", "rsi_change", "ema_diff", "price_vs_ema9", "price_vs_ema21",
    "return_1d", "return_3d", "return_5d",
    "high_low_range", "close_vs_high",
    "volatility_5d", "volatility_20d",
    "volume_ratio", "volume_change",
    "macd_norm", "macd_diff_norm", "bb_width", "price_vs_bb_mid", "atr_norm"
]


def train_model(symbols: list[str] | None = None, period: str = "1y") -> dict:
    """
    Train a prediction model on historical data for watchlist stocks.
    Returns training metrics.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report

    _ensure_model_dir()

    if symbols is None:
        symbols = config.WATCHLIST

    # Collect training data from all symbols
    all_data = []
    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        df = get_historical_data(symbol, period=period, interval="1d")
        if df.empty or len(df) < 30:
            continue
        features = prepare_features(df)
        features["symbol"] = symbol
        all_data.append(features)

    if not all_data:
        return {"error": "No data to train on"}

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.dropna(subset=FEATURE_COLS + ["target"])
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)

    if len(combined) < 50:
        return {"error": f"Only {len(combined)} samples, need at least 50"}

    X = combined[FEATURE_COLS].values
    y = combined["target"].values

    # Time-series cross-validation (no data leakage)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Hyperparameter tuning
    from sklearn.model_selection import RandomizedSearchCV
    
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [5, 10, 20]
    }
    
    base_model = GradientBoostingClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10,
        cv=tscv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    print("  Tuning hyperparameters...")
    random_search.fit(X, y)
    
    final_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Get CV scores of the best model
    best_index = random_search.best_index_
    scores = [
        random_search.cv_results_[f'split{i}_test_score'][best_index]
        for i in range(tscv.n_splits)
    ]

    # Save model
    model_path = os.path.join(MODEL_DIR, "predictor.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    # Feature importance
    importance = dict(zip(FEATURE_COLS, final_model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

    metrics = {
        "samples": len(combined),
        "symbols": len(all_data),
        "cv_accuracy": round(np.mean(scores) * 100, 1),
        "cv_scores": [round(s * 100, 1) for s in scores],
        "top_features": {k: round(v, 3) for k, v in top_features},
        "model_path": model_path,
    }

    # Save training log
    log = []
    if os.path.exists(TRAINING_LOG):
        with open(TRAINING_LOG) as f:
            log = json.load(f)
    log.append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "metrics": metrics,
    })
    with open(TRAINING_LOG, "w") as f:
        json.dump(log, f, indent=2)

    return metrics


def predict(symbol: str, df: pd.DataFrame) -> dict:
    """Predict whether a stock will go up or down tomorrow."""
    model_path = os.path.join(MODEL_DIR, "predictor.pkl")
    if not os.path.exists(model_path):
        return {"symbol": symbol, "error": "No trained model. Run: python main.py train"}

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    features = prepare_features(df)
    features = features.dropna(subset=FEATURE_COLS)

    if features.empty:
        return {"symbol": symbol, "error": "Insufficient data"}

    latest = features.iloc[-1:]
    X = latest[FEATURE_COLS].values

    prob = model.predict_proba(X)[0]
    prediction = "UP" if prob[1] > 0.5 else "DOWN"

    return {
        "symbol": symbol,
        "prediction": prediction,
        "confidence": round(float(max(prob)), 3),
        "prob_up": round(float(prob[1]), 3),
        "prob_down": round(float(prob[0]), 3),
        "price": round(float(latest.iloc[0]["Close"]), 2),
    }


def predict_watchlist(symbols: list[str] | None = None) -> list[dict]:
    """Get predictions for all watchlist stocks."""
    if symbols is None:
        symbols = config.WATCHLIST

    results = []
    for symbol in symbols:
        df = get_historical_data(symbol, period="60d", interval="1d")
        if df.empty:
            continue
        result = predict(symbol, df)
        results.append(result)

    return results


def print_predictions(predictions: list[dict]):
    """Print formatted predictions."""
    print(f"\n{'='*60}")
    print(f"  ML PREDICTIONS (Next Trading Day)")
    print(f"{'='*60}")

    for p in predictions:
        if "error" in p:
            print(f"  {p['symbol']:20s}  ERROR: {p['error']}")
            continue
        arrow = "^" if p["prediction"] == "UP" else "v"
        conf_bar = "#" * int(p["confidence"] * 10) + "." * (10 - int(p["confidence"] * 10))
        print(f"  {arrow} {p['symbol']:20s}  Rs.{p['price']:>8.2f}  {p['prediction']:4s}  "
              f"[{conf_bar}] {p['confidence']:.0%}  (up:{p['prob_up']:.0%} down:{p['prob_down']:.0%})")

    ups = [p for p in predictions if p.get("prediction") == "UP"]
    downs = [p for p in predictions if p.get("prediction") == "DOWN"]
    print(f"\n  {len(ups)} UP, {len(downs)} DOWN")
    print(f"{'='*60}")
