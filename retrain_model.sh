#!/bin/bash
# Retrain CatBoost model via Kaggle API
# Usage: ./retrain_model.sh
# Can be triggered manually or via cron (e.g., weekly Sunday night)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KAGGLE_DIR="$SCRIPT_DIR/kaggle_training"
MODEL_DIR="$SCRIPT_DIR/models"
KERNEL_SLUG="spidys/nse-catboost-trainer"
LOG_FILE="$SCRIPT_DIR/logs/retrain.log"

mkdir -p "$SCRIPT_DIR/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting model retrain ==="

# 1. Push the kernel to Kaggle
log "Pushing kernel to Kaggle..."
cd "$KAGGLE_DIR"
kaggle kernels push -p . 2>&1 | tee -a "$LOG_FILE"

# 2. Wait for completion (poll every 60s, max 30 min)
log "Waiting for training to complete..."
MAX_WAIT=1800
WAITED=0
POLL=60

while [ $WAITED -lt $MAX_WAIT ]; do
    sleep $POLL
    WAITED=$((WAITED + POLL))

    STATUS=$(kaggle kernels status "$KERNEL_SLUG" 2>/dev/null | tail -1)
    log "  Status: $STATUS (${WAITED}s elapsed)"

    if echo "$STATUS" | grep -qi "complete"; then
        log "Training completed!"
        break
    fi

    if echo "$STATUS" | grep -qi "error\|cancel"; then
        log "ERROR: Training failed — $STATUS"
        exit 1
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    log "ERROR: Timed out after ${MAX_WAIT}s"
    exit 1
fi

# 3. Download output files
log "Downloading model files..."
TMPDIR=$(mktemp -d)
kaggle kernels output "$KERNEL_SLUG" -p "$TMPDIR" 2>&1 | tee -a "$LOG_FILE"

# 4. Verify files exist
REQUIRED="predictor_catboost.pkl predictor_catboost.cbm feature_cols.json predictor_catboost.pkl.sha256"
for f in $REQUIRED; do
    if [ ! -f "$TMPDIR/$f" ]; then
        log "ERROR: Missing output file: $f"
        rm -rf "$TMPDIR"
        exit 1
    fi
done

# 5. Verify integrity
EXPECTED_HASH=$(cat "$TMPDIR/predictor_catboost.pkl.sha256")
ACTUAL_HASH=$(sha256sum "$TMPDIR/predictor_catboost.pkl" | cut -d' ' -f1)
if [ "$EXPECTED_HASH" != "$ACTUAL_HASH" ]; then
    log "ERROR: SHA256 mismatch! Expected=$EXPECTED_HASH Actual=$ACTUAL_HASH"
    rm -rf "$TMPDIR"
    exit 1
fi
log "Integrity check passed"

# 6. Deploy — copy to models/
log "Deploying model files..."
cp "$TMPDIR/predictor_catboost.pkl" "$MODEL_DIR/"
cp "$TMPDIR/predictor_catboost.cbm" "$MODEL_DIR/"
cp "$TMPDIR/feature_cols.json" "$MODEL_DIR/"
cp "$TMPDIR/predictor_catboost.pkl.sha256" "$MODEL_DIR/"
rm -rf "$TMPDIR"

# 7. Restart the trading agent to pick up new model
log "Restarting trading agent..."
sudo systemctl restart ai-trading-agent 2>/dev/null || true
sudo systemctl restart ai-trader-api 2>/dev/null || true

# 8. Log metrics from feature_cols.json
METRICS=$(python3 -c "
import json
with open('$MODEL_DIR/feature_cols.json') as f:
    m = json.load(f)
print(f\"Accuracy: {m.get('holdout_accuracy','?')}% | F1: {m.get('holdout_f1','?')}% | Precision: {m.get('holdout_precision','?')}% | Samples: {m.get('training_samples','?')}\")
" 2>/dev/null || echo "Could not parse metrics")
log "Model metrics: $METRICS"

log "=== Retrain complete ==="
