"""
NSE CatBoost Model Training — Kaggle Kernel (no internet dependencies)
All indicators computed manually. Data fetched via pandas_datareader or pre-loaded.
Outputs: predictor_catboost.pkl, predictor_catboost.cbm, feature_cols.json, predictor_catboost.pkl.sha256
"""

import os
import time
import json
import pickle
import hashlib
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import catboost as cb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print(f"CatBoost version: {cb.__version__}")

# ══════════════════════════════════════════════════════════════
# Manual Technical Indicator Functions (no 'ta' library needed)
# ══════════════════════════════════════════════════════════════

def _ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def _sma(series, window):
    return series.rolling(window).mean()

def _rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def _macd(close, fast=12, slow=26, signal=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger(close, window=20, std_dev=2):
    mid = _sma(close, window)
    std = close.rolling(window).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def _atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def _stochastic(high, low, close, k_window=14, d_window=3):
    lowest = low.rolling(k_window).min()
    highest = high.rolling(k_window).max()
    k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    d = k.rolling(d_window).mean()
    return k, d

def _adx(high, low, close, window=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = _atr(high, low, close, window)
    plus_di = 100 * _ema(plus_dm, window) / (atr + 1e-10)
    minus_di = 100 * _ema(minus_dm, window) / (atr + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = _ema(dx, window)
    return adx, plus_di, minus_di


# ══════════════════════════════════════════════════════════════
# Stock List
# ══════════════════════════════════════════════════════════════

NIFTY_250 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "AXISBANK.NS", "KOTAKBANK.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS",
    "WIPRO.NS", "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS", "M&M.NS",
    "NTPC.NS", "TATASTEEL.NS", "COALINDIA.NS", "HINDALCO.NS",
    "JSWSTEEL.NS", "ADANIPORTS.NS", "GRASIM.NS", "ADANIENT.NS", "TECHM.NS",
    "BRITANNIA.NS", "INDUSINDBK.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "CIPLA.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS", "HDFCLIFE.NS",
    "SBILIFE.NS", "BPCL.NS", "SHREECEM.NS", "LTIM.NS", "TRENT.NS",
    "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS", "AUROPHARMA.NS", "BANKBARODA.NS",
    "BEL.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS", "CANBK.NS",
    "CHOLAFIN.NS", "COLPAL.NS", "CONCOR.NS", "CUMMINSIND.NS", "DABUR.NS",
    "DLF.NS", "GAIL.NS", "GODREJCP.NS", "GODREJPROP.NS", "HAL.NS",
    "HAVELLS.NS", "ICICIPRULI.NS", "IDEA.NS", "IDFCFIRSTB.NS", "INDHOTEL.NS",
    "INDUSTOWER.NS", "IOC.NS", "IRCTC.NS", "IRFC.NS", "JIOFIN.NS",
    "JSWENERGY.NS", "JUBLFOOD.NS", "LICI.NS", "LUPIN.NS", "MARICO.NS",
    "MPHASIS.NS", "NAUKRI.NS", "NHPC.NS", "NMDC.NS",
    "OBEROIRLTY.NS", "OFSS.NS", "PAYTM.NS", "PIDILITIND.NS",
    "PNB.NS", "POLYCAB.NS", "RECLTD.NS", "SBICARD.NS", "SIEMENS.NS",
    "ABB.NS", "ABCAPITAL.NS", "ACC.NS", "ALKEM.NS",
    "ASHOKLEY.NS", "ASTRAL.NS", "ATUL.NS", "AUBANK.NS", "BALKRISIND.NS",
    "BATAINDIA.NS", "BHEL.NS", "CANFINHOME.NS", "CHAMBLFERT.NS", "COFORGE.NS",
    "COROMANDEL.NS", "CROMPTON.NS", "CUB.NS", "CYIENT.NS",
    "DEEPAKNTR.NS", "DEVYANI.NS", "DIXON.NS", "ESCORTS.NS",
    "EXIDEIND.NS", "FEDERALBNK.NS", "FORTIS.NS", "GLENMARK.NS",
    "GNFC.NS", "GSPL.NS", "HFCL.NS", "HINDPETRO.NS", "HUDCO.NS",
    "IBREALEST.NS", "IEX.NS", "INDIANB.NS", "INDIAMART.NS",
    "IREDA.NS", "JKCEMENT.NS", "JSL.NS", "KAJARIACER.NS", "KEI.NS",
    "LALPATHLAB.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LTTS.NS",
    "MANAPPURAM.NS", "MFSL.NS", "MOTHERSON.NS", "MUTHOOTFIN.NS",
    "NATIONALUM.NS", "NAVINFLUOR.NS", "NBCC.NS", "NCC.NS",
    "NOCIL.NS", "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PHOENIXLTD.NS",
    "PIIND.NS", "PRESTIGE.NS", "PVRINOX.NS", "RAMCOCEM.NS", "RBLBANK.NS",
    "RVNL.NS", "SAIL.NS", "SJVN.NS",
    "SRF.NS", "SUNDARMFIN.NS", "SUNDRMFAST.NS", "SUNTV.NS",
    "SUPREMEIND.NS", "SUZLON.NS", "SYNGENE.NS", "TATACHEM.NS", "TATACOMM.NS",
    "TATACONSUM.NS", "TATAELXSI.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TORNTPOWER.NS",
    "TVSMOTOR.NS", "UBL.NS", "UNIONBANK.NS", "UPL.NS",
    "VEDL.NS", "VOLTAS.NS", "YESBANK.NS", "ZEEL.NS",
    "DELHIVERY.NS", "NYKAA.NS", "POLICYBZR.NS",
    "HAPPSTMNDS.NS", "KPITTECH.NS", "SONACOMS.NS",
    "TIINDIA.NS", "APLAPOLLO.NS", "GUJGASLTD.NS",
    "PAGEIND.NS", "MCX.NS", "IPCALAB.NS", "SUMICHEM.NS",
]
NIFTY_250 = list(dict.fromkeys(NIFTY_250))
print(f"Symbols: {len(NIFTY_250)}")

# ══════════════════════════════════════════════════════════════
# Fetch Data — try yfinance, fall back to pre-uploaded CSV
# ══════════════════════════════════════════════════════════════

DATA_CSV = "/kaggle/input/nse-stock-data/nse_daily_5y.csv"
all_stock_data = {}
failed = []

try:
    import yfinance as yf
    print("yfinance available — fetching live data...")
    for i, symbol in enumerate(NIFTY_250):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"Fetching [{i+1}/{len(NIFTY_250)}]: {symbol}")
        try:
            df = yf.download(symbol, period="5y", interval="1d", progress=False)
            if not df.empty and len(df) > 200:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                all_stock_data[symbol] = df
            else:
                failed.append(symbol)
        except Exception:
            failed.append(symbol)
        if (i + 1) % 10 == 0:
            time.sleep(0.3)
except ImportError:
    print("yfinance not available — checking for pre-uploaded CSV...")
    if os.path.exists(DATA_CSV):
        big_df = pd.read_csv(DATA_CSV, parse_dates=['Date'], index_col='Date')
        for symbol in big_df['symbol'].unique():
            sdf = big_df[big_df['symbol'] == symbol].drop(columns=['symbol']).sort_index()
            if len(sdf) > 200:
                all_stock_data[symbol] = sdf
        print(f"Loaded {len(all_stock_data)} stocks from CSV")
    else:
        # Last resort: generate the CSV download script
        print("ERROR: No data source available!")
        print("Please either:")
        print("  1. Upload nse_daily_5y.csv as a Kaggle dataset")
        print("  2. Run this kernel with internet access enabled")
        raise SystemExit(1)

print(f"Fetched: {len(all_stock_data)} stocks, Failed: {len(failed)}")

# ══════════════════════════════════════════════════════════════
# Feature Engineering (45+ features, manual calculations)
# ══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    'rsi_14', 'rsi_7', 'rsi_21', 'rsi_change', 'rsi_change_3',
    'ema_diff_9_21', 'ema_diff_21_50', 'ema_diff_50_200',
    'price_vs_ema9', 'price_vs_ema21', 'price_vs_ema50', 'price_vs_ema200',
    'macd', 'macd_signal', 'macd_diff', 'macd_diff_change',
    'bb_width', 'bb_position',
    'atr', 'atr_7',
    'stoch_k', 'stoch_d', 'stoch_diff',
    'adx', 'adx_diff',
    'return_1', 'return_3', 'return_5', 'return_10', 'return_20',
    'volatility_5', 'volatility_10', 'volatility_20', 'vol_ratio_5_20',
    'volume_ratio', 'volume_change', 'volume_trend',
    'high_low_range', 'close_position', 'upper_wick', 'lower_wick', 'body_size',
    'dist_to_high_20', 'dist_to_low_20', 'dist_to_high_50', 'dist_to_low_50',
    'rsi_divergence',
    'dow_sin', 'dow_cos',
]


def engineer_features(df):
    df = df.copy()
    close, high, low, volume, opn = df['Close'], df['High'], df['Low'], df['Volume'], df['Open']

    # RSI
    df['rsi_14'] = _rsi(close, 14)
    df['rsi_7'] = _rsi(close, 7)
    df['rsi_21'] = _rsi(close, 21)
    df['rsi_change'] = df['rsi_14'].diff()
    df['rsi_change_3'] = df['rsi_14'].diff(3)

    # EMAs
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    ema100 = _ema(close, 100)
    ema200 = _ema(close, 200)
    df['ema_diff_9_21'] = (ema9 - ema21) / close
    df['ema_diff_21_50'] = (ema21 - ema50) / close
    df['ema_diff_50_200'] = (ema50 - ema200) / close
    df['price_vs_ema9'] = (close - ema9) / close
    df['price_vs_ema21'] = (close - ema21) / close
    df['price_vs_ema50'] = (close - ema50) / close
    df['price_vs_ema200'] = (close - ema200) / close

    # MACD
    macd_line, macd_sig, macd_hist = _macd(close)
    df['macd'] = macd_line / close
    df['macd_signal'] = macd_sig / close
    df['macd_diff'] = macd_hist / close
    df['macd_diff_change'] = df['macd_diff'].diff()

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = _bollinger(close)
    df['bb_width'] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)

    # ATR
    df['atr'] = _atr(high, low, close, 14) / close
    df['atr_7'] = _atr(high, low, close, 7) / close

    # Stochastic
    df['stoch_k'], df['stoch_d'] = _stochastic(high, low, close)
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']

    # ADX
    adx_val, plus_di, minus_di = _adx(high, low, close, 14)
    df['adx'] = adx_val
    df['adx_diff'] = plus_di - minus_di

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
    df['upper_wick'] = (high - np.maximum(close, opn)) / (high - low + 1e-8)
    df['lower_wick'] = (np.minimum(close, opn) - low) / (high - low + 1e-8)
    df['body_size'] = abs(close - opn) / (high - low + 1e-8)

    # Support / Resistance
    df['dist_to_high_20'] = (high.rolling(20).max() - close) / close
    df['dist_to_low_20'] = (close - low.rolling(20).min()) / close
    df['dist_to_high_50'] = (high.rolling(50).max() - close) / close
    df['dist_to_low_50'] = (close - low.rolling(50).min()) / close

    # RSI divergence
    df['rsi_divergence'] = np.sign(close.diff(5)) * np.sign(-df['rsi_14'].diff(5))

    # Day of week
    if hasattr(df.index, 'dayofweek'):
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)
    else:
        df['dow_sin'] = 0
        df['dow_cos'] = 0

    return df


# ══════════════════════════════════════════════════════════════
# Build Dataset
# ══════════════════════════════════════════════════════════════

TARGET_HORIZON = 5
TARGET_THRESHOLD = 0.02

all_processed = []
for i, (symbol, df) in enumerate(all_stock_data.items()):
    if (i + 1) % 50 == 0:
        print(f"Processing [{i+1}/{len(all_stock_data)}]")
    try:
        featured = engineer_features(df)
        future_max = featured['High'].rolling(TARGET_HORIZON).max().shift(-TARGET_HORIZON)
        featured['target'] = (future_max >= featured['Close'] * (1 + TARGET_THRESHOLD)).astype(int)
        valid = featured.dropna(subset=FEATURE_COLS + ['target'])
        valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
        if len(valid) > 50:
            all_processed.append(valid)
    except Exception:
        pass

combined = pd.concat(all_processed).sort_index()
print(f"Total samples: {len(combined):,}, Stocks: {len(all_processed)}, Positive rate: {combined['target'].mean():.1%}")

# ══════════════════════════════════════════════════════════════
# Split
# ══════════════════════════════════════════════════════════════

X = combined[FEATURE_COLS].values
y = combined['target'].values
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_holdout, y_holdout = X[val_end:], y[val_end:]

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Holdout: {len(X_holdout):,}")

# ══════════════════════════════════════════════════════════════
# Train
# ══════════════════════════════════════════════════════════════

task_type = 'GPU'
try:
    test = CatBoostClassifier(iterations=5, task_type='GPU', verbose=0)
    test.fit(X_train[:100], y_train[:100])
    print("Using GPU")
except Exception:
    task_type = 'CPU'
    print("Using CPU")

params = dict(
    iterations=5000,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=5,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    min_data_in_leaf=30,
    loss_function='Logloss',
    eval_metric='Logloss',
    task_type=task_type,
    random_seed=42,
    early_stopping_rounds=300,
    verbose=200,
)
if task_type == 'CPU':
    params['colsample_bylevel'] = 0.8

model = CatBoostClassifier(**params)
train_pool = Pool(X_train, y_train, feature_names=FEATURE_COLS)
val_pool = Pool(X_val, y_val, feature_names=FEATURE_COLS)

print(f"Training on {len(X_train):,} samples...")
model.fit(train_pool, eval_set=val_pool, use_best_model=True)
print(f"Best iteration: {model.get_best_iteration()}")

# ══════════════════════════════════════════════════════════════
# Evaluate
# ══════════════════════════════════════════════════════════════

y_pred = model.predict(X_holdout)
y_prob = model.predict_proba(X_holdout)[:, 1]

print("\n" + "=" * 60)
print("  HOLDOUT RESULTS")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_holdout, y_pred):.1%}")
print(f"Precision: {precision_score(y_holdout, y_pred, zero_division=0):.1%}")
print(f"Recall:    {recall_score(y_holdout, y_pred, zero_division=0):.1%}")
print(f"F1:        {f1_score(y_holdout, y_pred, zero_division=0):.1%}")
print(classification_report(y_holdout, y_pred, target_names=['DOWN/FLAT', 'UP >=2%']))

for thr in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    preds = (y_prob >= thr).astype(int)
    ns = preds.sum()
    if ns > 0:
        p = precision_score(y_holdout, preds, zero_division=0)
        print(f"  @{thr:.0%}: {ns:>5,} signals, precision {p:.1%}")

# ══════════════════════════════════════════════════════════════
# Retrain on train+val for deployment
# ══════════════════════════════════════════════════════════════

X_deploy = np.concatenate([X_train, X_val])
y_deploy = np.concatenate([y_train, y_val])
best_iter = max(model.get_best_iteration(), 300)

deploy_params = dict(params)
deploy_params['iterations'] = best_iter + 200
deploy_params.pop('early_stopping_rounds', None)

final_model = CatBoostClassifier(**deploy_params)
final_model.fit(Pool(X_deploy, y_deploy, feature_names=FEATURE_COLS),
                eval_set=Pool(X_holdout, y_holdout, feature_names=FEATURE_COLS))

y_final = final_model.predict(X_holdout)
print(f"\nFinal holdout — Accuracy: {accuracy_score(y_holdout, y_final):.1%}, F1: {f1_score(y_holdout, y_final, zero_division=0):.1%}")

# ══════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════

OUT = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
os.makedirs(OUT, exist_ok=True)
final_model.save_model(f"{OUT}/predictor_catboost.cbm")
with open(f"{OUT}/predictor_catboost.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open(f"{OUT}/feature_cols.json", "w") as f:
    json.dump({
        'features': FEATURE_COLS,
        'target_horizon': TARGET_HORIZON,
        'target_threshold': TARGET_THRESHOLD,
        'model_type': 'catboost',
        'training_samples': len(X_deploy),
        'stocks_count': len(all_processed),
        'interval': '1d',
        'holdout_f1': round(f1_score(y_holdout, y_final, zero_division=0) * 100, 1),
        'holdout_precision': round(precision_score(y_holdout, y_final, zero_division=0) * 100, 1),
        'holdout_accuracy': round(accuracy_score(y_holdout, y_final) * 100, 1),
    }, f, indent=2)

sha = hashlib.sha256()
with open(f"{OUT}/predictor_catboost.pkl", "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
        sha.update(chunk)
with open(f"{OUT}/predictor_catboost.pkl.sha256", "w") as f:
    f.write(sha.hexdigest())

print(f"\nSaved to {OUT}/:")
for fn in ["predictor_catboost.cbm", "predictor_catboost.pkl", "feature_cols.json", "predictor_catboost.pkl.sha256"]:
    sz = os.path.getsize(f"{OUT}/{fn}") / 1024
    print(f"  {fn} ({sz:.1f} KB)")

print("\nDone!")
