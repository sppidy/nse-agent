"""
CatBoost ML predictor — trained on 5 years of daily data for 210+ NSE stocks.

Predicts whether a stock will move up ≥2% within the next 5 trading days,
based on 45+ technical features (RSI, EMA, MACD, Bollinger, ATR, Stochastic,
ADX, multi-period returns, volatility, volume, price action, support/resistance).

Model is trained in Google Colab (GPU) and deployed here for inference.
"""

import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import ta
import config
from logger import logger
from persistence import read_json, write_json_atomic
from data_fetcher import get_historical_data

MODEL_DIR = os.path.join(config.PROJECT_DIR, "models")
TRAINING_LOG = os.path.join(config.PROJECT_DIR, "training_log.json")

# Model files (Colab-trained CatBoost)
MODEL_PKL = os.path.join(MODEL_DIR, "predictor_catboost.pkl")
MODEL_CBM = os.path.join(MODEL_DIR, "predictor_catboost.cbm")
MODEL_HASH_FILE = os.path.join(MODEL_DIR, "predictor_catboost.pkl.sha256")
FEATURE_META = os.path.join(MODEL_DIR, "feature_cols.json")

# Legacy model path (for backward compat)
LEGACY_MODEL = os.path.join(MODEL_DIR, "predictor.pkl")
LEGACY_HASH = os.path.join(MODEL_DIR, "predictor.pkl.sha256")


FEATURE_COLS = [
    # RSI
    'rsi_14', 'rsi_7', 'rsi_21', 'rsi_change', 'rsi_change_3',
    # EMA
    'ema_diff_9_21', 'ema_diff_21_50', 'ema_diff_50_200',
    'price_vs_ema9', 'price_vs_ema21', 'price_vs_ema50', 'price_vs_ema200',
    # MACD
    'macd', 'macd_signal', 'macd_diff', 'macd_diff_change',
    # Bollinger
    'bb_width', 'bb_position',
    # ATR
    'atr', 'atr_7',
    # Stochastic
    'stoch_k', 'stoch_d', 'stoch_diff',
    # ADX
    'adx', 'adx_diff',
    # Returns
    'return_1', 'return_3', 'return_5', 'return_10', 'return_20',
    # Volatility
    'volatility_5', 'volatility_10', 'volatility_20', 'vol_ratio_5_20',
    # Volume
    'volume_ratio', 'volume_change', 'volume_trend',
    # Price action
    'high_low_range', 'close_position', 'upper_wick', 'lower_wick', 'body_size',
    # Support/Resistance
    'dist_to_high_20', 'dist_to_low_20', 'dist_to_high_50', 'dist_to_low_50',
    # Divergence
    'rsi_divergence',
    # Time
    'dow_sin', 'dow_cos',
]


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
    """CatBoost model is trained in Colab — local retraining not needed."""
    if not os.path.exists(MODEL_PKL) and not os.path.exists(MODEL_CBM):
        return True, "No CatBoost model found — train in Colab and upload"
    return False, "CatBoost model loaded (Colab-trained)"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 45+ features from daily OHLCV data — matches Colab training exactly."""
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # RSI (multiple periods)
    df['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df['rsi_7'] = ta.momentum.RSIIndicator(close, window=7).rsi()
    df['rsi_21'] = ta.momentum.RSIIndicator(close, window=21).rsi()
    df['rsi_change'] = df['rsi_14'].diff()
    df['rsi_change_3'] = df['rsi_14'].diff(3)

    # EMAs
    df['ema_9'] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df['ema_100'] = ta.trend.EMAIndicator(close, window=100).ema_indicator()
    df['ema_200'] = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    df['ema_diff_9_21'] = (df['ema_9'] - df['ema_21']) / close
    df['ema_diff_21_50'] = (df['ema_21'] - df['ema_50']) / close
    df['ema_diff_50_200'] = (df['ema_50'] - df['ema_200']) / close
    df['price_vs_ema9'] = (close - df['ema_9']) / close
    df['price_vs_ema21'] = (close - df['ema_21']) / close
    df['price_vs_ema50'] = (close - df['ema_50']) / close
    df['price_vs_ema200'] = (close - df['ema_200']) / close

    # MACD
    macd = ta.trend.MACD(close)
    df['macd'] = macd.macd() / close
    df['macd_signal'] = macd.macd_signal() / close
    df['macd_diff'] = macd.macd_diff() / close
    df['macd_diff_change'] = df['macd_diff'].diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df['bb_position'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)

    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range() / close
    df['atr_7'] = ta.volatility.AverageTrueRange(high, low, close, window=7).average_true_range() / close

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']

    # ADX
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df['adx'] = adx.adx()
    df['adx_diff'] = adx.adx_pos() - adx.adx_neg()

    # Returns
    df['return_1'] = close.pct_change(1)
    df['return_3'] = close.pct_change(3)
    df['return_5'] = close.pct_change(5)
    df['return_10'] = close.pct_change(10)
    df['return_20'] = close.pct_change(20)

    # Volatility
    df['volatility_5'] = df['return_1'].rolling(5).std()
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()
    df['vol_ratio_5_20'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)

    # Volume
    vol_sma_20 = volume.rolling(20).mean()
    df['volume_ratio'] = volume / (vol_sma_20 + 1)
    df['volume_change'] = volume.pct_change()
    df['volume_trend'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1)

    # Price action
    df['high_low_range'] = (high - low) / close
    df['close_position'] = (close - low) / (high - low + 1e-8)
    df['upper_wick'] = (high - np.maximum(close, df['Open'])) / (high - low + 1e-8)
    df['lower_wick'] = (np.minimum(close, df['Open']) - low) / (high - low + 1e-8)
    df['body_size'] = abs(close - df['Open']) / (high - low + 1e-8)

    # Support / Resistance
    df['dist_to_high_20'] = (high.rolling(20).max() - close) / close
    df['dist_to_low_20'] = (close - low.rolling(20).min()) / close
    df['dist_to_high_50'] = (high.rolling(50).max() - close) / close
    df['dist_to_low_50'] = (close - low.rolling(50).min()) / close

    # RSI divergence
    df['rsi_divergence'] = np.sign(close.diff(5)) * np.sign(-df['rsi_14'].diff(5))

    # Day of week (cyclical)
    if hasattr(df.index, 'dayofweek'):
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)
    else:
        df['dow_sin'] = 0
        df['dow_cos'] = 0

    return df


def _load_model():
    """Load the CatBoost model — try .cbm (native) first, then .pkl."""
    # Try native CatBoost format first (faster, smaller)
    if os.path.exists(MODEL_CBM):
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(MODEL_CBM)
            return model
        except Exception as e:
            logger.warning(f"Failed to load .cbm model: {e}")

    # Fall back to pickle
    if os.path.exists(MODEL_PKL):
        if os.path.exists(MODEL_HASH_FILE):
            expected = Path(MODEL_HASH_FILE).read_text(encoding="utf-8").strip()
            actual = _sha256_file(MODEL_PKL)
            if expected != actual:
                logger.error("CatBoost model integrity check failed")
                return None
        with open(MODEL_PKL, "rb") as f:
            return pickle.load(f)

    # Legacy GradientBoosting model
    if os.path.exists(LEGACY_MODEL):
        logger.info("Loading legacy sklearn model (not CatBoost)")
        if os.path.exists(LEGACY_HASH):
            expected = Path(LEGACY_HASH).read_text(encoding="utf-8").strip()
            actual = _sha256_file(LEGACY_MODEL)
            if expected != actual:
                return None
        with open(LEGACY_MODEL, "rb") as f:
            return pickle.load(f)

    return None


# Cache the model in memory after first load
_cached_model = None


def predict(symbol: str, df: pd.DataFrame) -> dict:
    """Predict whether a stock will go up ≥2% in the next 5 trading days."""
    global _cached_model

    if _cached_model is None:
        _cached_model = _load_model()

    if _cached_model is None:
        return {"symbol": symbol, "error": "No model found. Train in Colab and upload to models/"}

    features = prepare_features(df)
    features = features.dropna(subset=FEATURE_COLS)
    features = features.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)

    if features.empty:
        return {"symbol": symbol, "error": "Insufficient data for features"}

    latest = features.iloc[-1:]
    X = latest[FEATURE_COLS].values

    try:
        prob = _cached_model.predict_proba(X)[0]
        prediction = "UP" if prob[1] > 0.5 else "DOWN"

        return {
            "symbol": symbol,
            "prediction": prediction,
            "confidence": round(float(max(prob)), 3),
            "prob_up": round(float(prob[1]), 3),
            "prob_down": round(float(prob[0]), 3),
            "price": round(float(latest.iloc[0]["Close"]), 2),
        }
    except Exception as e:
        return {"symbol": symbol, "error": f"Prediction failed: {e}"}


def predict_watchlist(symbols: list[str] | None = None) -> list[dict]:
    """Get predictions for all watchlist stocks."""
    if symbols is None:
        symbols = config.WATCHLIST

    results = []
    for symbol in symbols:
        df = get_historical_data(symbol, period="1y", interval="1d")
        if df.empty:
            continue
        result = predict(symbol, df)
        results.append(result)

    return results


def train_model(symbols: list[str] | None = None, period: str = "5y") -> dict:
    """
    Finetune existing CatBoost model with fresh data.
    Loads the current model and continues training (incremental learning),
    preserving what it already learned while adapting to new market data.
    Falls back to training from scratch if no existing model is found.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if symbols is None:
        symbols = config.WATCHLIST

    # Collect training data
    all_data = []
    for symbol in symbols:
        logger.info(f"  Fetching {symbol}...")
        df = get_historical_data(symbol, period=period, interval="1d")
        if df.empty or len(df) < 50:
            continue
        features = prepare_features(df)

        # Target: 2% up in next 5 days
        future_max = features['High'].rolling(5).max().shift(-5)
        features['target'] = (future_max >= features['Close'] * 1.02).astype(int)

        features["symbol"] = symbol
        all_data.append(features)

    if not all_data:
        return {"error": "No data to train on"}

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.dropna(subset=FEATURE_COLS + ["target"])
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)

    if len(combined) < 100:
        return {"error": f"Only {len(combined)} samples, need at least 100"}

    X = combined[FEATURE_COLS].values
    y = combined["target"].values

    # Chronological split
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Load existing model for finetuning
    existing_model = _load_model()
    is_finetune = existing_model is not None and hasattr(existing_model, 'get_params')

    try:
        from catboost import CatBoostClassifier

        if is_finetune:
            # Finetune: continue from existing model with lower learning rate
            logger.info(f"  Finetuning existing CatBoost model on {len(X_train):,} samples...")
            model = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.01,       # Lower LR for finetuning — gentle updates
                l2_leaf_reg=5,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                min_data_in_leaf=20,
                loss_function='Logloss',
                eval_metric='Logloss',
                random_seed=42,
                early_stopping_rounds=50,
                verbose=100,
            )
            model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                init_model=existing_model,  # Continue from existing model
            )
        else:
            # No existing model — train from scratch
            logger.info(f"  No existing model found. Training CatBoost from scratch on {len(X_train):,} samples...")
            model = CatBoostClassifier(
                iterations=1000,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                min_data_in_leaf=20,
                loss_function='Logloss',
                eval_metric='Logloss',
                random_seed=42,
                early_stopping_rounds=100,
                verbose=100,
            )
            model.fit(X_train, y_train, eval_set=(X_test, y_test))

        model_type = "catboost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        logger.info(f"  CatBoost not available, using sklearn GradientBoosting...")
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, random_state=42,
        )
        model.fit(X_train, y_train)
        model_type = "sklearn"
        is_finetune = False

    # Evaluate
    from sklearn.metrics import accuracy_score, f1_score, precision_score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)

    # Only save if finetuned model is not worse than existing
    if is_finetune:
        old_pred = existing_model.predict(X_test)
        old_f1 = f1_score(y_test, old_pred, zero_division=0)
        if f1 < old_f1 - 0.02:  # Allow 2% margin
            logger.warning(f"  Finetuned model worse (F1: {f1:.3f} vs old {old_f1:.3f}). Keeping old model.")
            return {
                "samples": len(combined), "symbols": len(all_data),
                "model_type": model_type, "action": "kept_old",
                "old_f1": round(old_f1 * 100, 1),
                "new_f1": round(f1 * 100, 1),
                "reason": "Finetuned model regressed — old model retained",
            }
        logger.info(f"  Finetuned model accepted (F1: {f1:.3f} vs old {old_f1:.3f})")

    # Save
    model_path = MODEL_PKL if model_type == "catboost" else LEGACY_MODEL
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    model_hash = _sha256_file(model_path)
    hash_path = MODEL_HASH_FILE if model_type == "catboost" else LEGACY_HASH
    with open(hash_path, "w", encoding="utf-8") as f:
        f.write(model_hash)

    if model_type == "catboost":
        model.save_model(MODEL_CBM)

    # Clear cached model so next predict() loads the new one
    global _cached_model
    _cached_model = None

    metrics = {
        "samples": len(combined),
        "symbols": len(all_data),
        "model_type": model_type,
        "action": "finetuned" if is_finetune else "trained_from_scratch",
        "holdout_accuracy": round(accuracy * 100, 1),
        "holdout_f1": round(f1 * 100, 1),
        "holdout_precision": round(precision * 100, 1),
        "model_path": model_path,
        "model_sha256": model_hash,
    }

    log = _get_training_log()
    log.append({"timestamp": pd.Timestamp.now().isoformat(), "metrics": metrics})
    write_json_atomic(TRAINING_LOG, log)

    return metrics


def print_predictions(predictions: list[dict]):
    """Print formatted predictions."""
    print(f"\n{'='*60}")
    print(f"  ML PREDICTIONS — CatBoost (≥2% in 5 days)")
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
