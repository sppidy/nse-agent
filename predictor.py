"""
Local ML predictor — trains on historical data to predict price direction.

Uses scikit-learn (lightweight, no GPU needed) to train a model that predicts
whether a stock will go UP or DOWN in the next trading session, based on
technical indicators.

The model improves over time as more data is collected from paper trading.
"""

import os
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import config
from logger import logger
from persistence import read_json, write_json_atomic
from strategy import add_indicators
from data_fetcher import get_historical_data

MODEL_DIR = os.path.join(config.PROJECT_DIR, "models")
TRAINING_LOG = os.path.join(config.PROJECT_DIR, "training_log.json")
MODEL_HASH_FILE = os.path.join(MODEL_DIR, "predictor.pkl.sha256")


def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_training_log() -> list[dict]:
    return read_json(TRAINING_LOG, default=[])


def get_latest_training_metrics() -> dict | None:
    log = _get_training_log()
    if not log:
        return None
    return log[-1].get("metrics", {})


def should_retrain(min_hours: int = 18) -> tuple[bool, str]:
    """Return retrain decision and reason based on training recency."""
    log = _get_training_log()
    if not log:
        return True, "No previous training log"
    last_ts = log[-1].get("timestamp")
    if not last_ts:
        return True, "Missing last training timestamp"
    try:
        last_dt = datetime.fromisoformat(last_ts)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return True, "Invalid last training timestamp"
    now = datetime.now(timezone.utc)
    elapsed_hours = (now - last_dt).total_seconds() / 3600
    if elapsed_hours >= min_hours:
        return True, f"{elapsed_hours:.1f}h elapsed since last training"
    return False, f"Only {elapsed_hours:.1f}h since last training"


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
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    _ensure_model_dir()

    if symbols is None:
        symbols = config.WATCHLIST

    # Collect training data from all symbols
    all_data = []
    for symbol in symbols:
        logger.info(f"  Fetching {symbol}...")
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
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("  Tuning hyperparameters...")
    random_search.fit(X, y)
    
    final_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Get CV scores of the best model
    best_index = random_search.best_index_
    scores = [
        random_search.cv_results_[f'split{i}_test_score'][best_index]
        for i in range(tscv.n_splits)
    ]

    # Walk-forward out-of-fold metrics with best tuned parameters
    oof_true = []
    oof_pred = []
    for train_idx, test_idx in tscv.split(X):
        fold_model = clone(final_model)
        fold_model.fit(X[train_idx], y[train_idx])
        preds = fold_model.predict(X[test_idx])
        oof_true.extend(y[test_idx].tolist())
        oof_pred.extend(preds.tolist())

    oof_accuracy = accuracy_score(oof_true, oof_pred) if oof_true else 0.0
    oof_precision = precision_score(oof_true, oof_pred, zero_division=0) if oof_true else 0.0
    oof_recall = recall_score(oof_true, oof_pred, zero_division=0) if oof_true else 0.0
    oof_f1 = f1_score(oof_true, oof_pred, zero_division=0) if oof_true else 0.0
    class_balance_up = float(np.mean(y)) if len(y) > 0 else 0.0
    naive_baseline = max(class_balance_up, 1 - class_balance_up)

    # Final chronological holdout evaluation (live-like split).
    split_idx = int(len(X) * 0.85)
    holdout_accuracy = holdout_precision = holdout_recall = holdout_f1 = 0.0
    if split_idx > 30 and split_idx < len(X) - 5:
        final_model.fit(X[:split_idx], y[:split_idx])
        holdout_pred = final_model.predict(X[split_idx:])
        y_holdout = y[split_idx:]
        holdout_accuracy = accuracy_score(y_holdout, holdout_pred)
        holdout_precision = precision_score(y_holdout, holdout_pred, zero_division=0)
        holdout_recall = recall_score(y_holdout, holdout_pred, zero_division=0)
        holdout_f1 = f1_score(y_holdout, holdout_pred, zero_division=0)
    else:
        # Fallback to full-fit when split is not reliable for tiny datasets.
        final_model.fit(X, y)

    # Refit on full data for deployment after validation metrics are computed.
    final_model.fit(X, y)

    # Feature importance
    importance = dict(zip(FEATURE_COLS, final_model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

    model_path = os.path.join(MODEL_DIR, "predictor.pkl")
    prev_metrics = get_latest_training_metrics() or {}
    prev_f1 = float(prev_metrics.get("walk_forward_f1", 0))
    prev_holdout_f1 = float(prev_metrics.get("holdout_f1", 0))

    promote_model = True
    promote_reason = "No previous model baseline"
    if prev_metrics:
        if (oof_f1 * 100) + 0.5 < prev_f1 and (holdout_f1 * 100) + 0.5 < prev_holdout_f1:
            promote_model = False
            promote_reason = "Candidate underperforms previous model on both walk-forward and holdout F1"
        else:
            promote_reason = "Candidate is stable/improved vs previous model"

    model_hash = ""
    if promote_model or not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        model_hash = _sha256_file(model_path)
        with open(MODEL_HASH_FILE, "w", encoding="utf-8") as f:
            f.write(model_hash)
        promote_model = True
    else:
        model_hash = prev_metrics.get("model_sha256", "")

    metrics = {
        "samples": len(combined),
        "symbols": len(all_data),
        "cv_accuracy": round(np.mean(scores) * 100, 1),
        "cv_scores": [round(s * 100, 1) for s in scores],
        "walk_forward_accuracy": round(oof_accuracy * 100, 1),
        "walk_forward_precision": round(oof_precision * 100, 1),
        "walk_forward_recall": round(oof_recall * 100, 1),
        "walk_forward_f1": round(oof_f1 * 100, 1),
        "holdout_accuracy": round(holdout_accuracy * 100, 1),
        "holdout_precision": round(holdout_precision * 100, 1),
        "holdout_recall": round(holdout_recall * 100, 1),
        "holdout_f1": round(holdout_f1 * 100, 1),
        "baseline_accuracy": round(naive_baseline * 100, 1),
        "up_class_ratio": round(class_balance_up * 100, 1),
        "top_features": {k: round(v, 3) for k, v in top_features},
        "model_path": model_path,
        "model_sha256": model_hash,
        "model_promoted": promote_model,
        "promotion_reason": promote_reason,
        "best_params": best_params,
    }

    # Save training log
    log = _get_training_log()
    log.append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "metrics": metrics,
    })
    write_json_atomic(TRAINING_LOG, log)

    return metrics


def predict(symbol: str, df: pd.DataFrame) -> dict:
    """Predict whether a stock will go up or down tomorrow."""
    model_path = os.path.join(MODEL_DIR, "predictor.pkl")
    if not os.path.exists(model_path):
        return {"symbol": symbol, "error": "No trained model. Run: python main.py train"}
    if not os.path.exists(MODEL_HASH_FILE):
        return {"symbol": symbol, "error": "Model integrity file missing. Re-train model."}

    expected_hash = Path(MODEL_HASH_FILE).read_text(encoding="utf-8").strip()
    actual_hash = _sha256_file(model_path)
    if expected_hash != actual_hash:
        return {"symbol": symbol, "error": "Model integrity check failed. Re-train model."}

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
