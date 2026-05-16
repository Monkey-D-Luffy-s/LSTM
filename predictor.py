try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import os
    import tensorflow as tf
    import logging
    import warnings
    # Suppress Python-level warnings
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')
    
    # Limit memory growth and force CPU if needed to avoid overhead
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Set threading limits to avoid excessive core utilization/memory overhead
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    import os
    import sys
    import joblib
    import gc
except ImportError as e:
    print(f"[CRITICAL ERROR] Missing Python dependency: {e}")
    print("Please run: pip install pandas numpy scikit-learn tensorflow joblib")
    sys.exit(1)

# To run this script, you will need:
# pip install pandas numpy scikit-learn tensorflow joblib

FEATURES = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'ema', 'ema9', 'sma20', 'vwap', 'volatility', 'bb_upper', 'bb_lower', 'atr']
MODEL_PATH = 'model.keras'
SCALER_PATH = 'scaler.gz'
THRESHOLD_PATH = 'threshold.json'


def calculate_indicators(df):
    """
    Calculates technical indicators: RSI, MACD, EMA, VWAP, and Volatility.
    """
    try:
        # EMA (20 periods)
        df['ema'] = df['close'].ewm(span=20, adjust=False).mean()

        # EMA (9 periods) - specifically requested
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        
        # SMA (20 periods)
        df['sma20'] = df['close'].rolling(window=20).mean()
        
        # Bollinger Bands (20 periods, 2 std dev)
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (std_20 * 2)
        df['bb_lower'] = df['sma20'] - (std_20 * 2)
        
        # ATR (14 periods)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()

        # RSI (14 periods)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # VWAP
        v = df['volume']
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * v).cumsum() / (v.cumsum() + 1e-10)

        # Volatility Regime
        df['returns'] = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
        df['volatility'] = df['returns'].rolling(window=20).std()

        # Handle NaNs from indicators (like RSI/rolling) by dropping them to prevent data leakage
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        raise

def prepare_train_data(df, window_size=60, horizon=2):
    """Prepare (X, y) where X is a window ending at time t and y is whether close(t+horizon) > close(t)."""
    data = df[FEATURES].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    # i is the window end index (t). Need i + horizon to exist.
    for i in range(window_size, len(scaled_data) - horizon):
        X.append(scaled_data[i - window_size:i])
        y.append(1 if df['close'].iloc[i + horizon] > df['close'].iloc[i] else 0)

    return np.array(X), np.array(y), scaler


def focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        return -tf.keras.backend.mean(alpha_t * tf.keras.backend.pow(1. - pt, gamma) * tf.keras.backend.log(pt + tf.keras.backend.epsilon()))
    return focal_loss_fixed

def build_model(input_shape):
    model = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        Bidirectional(LSTM(units=64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),
        Bidirectional(LSTM(units=32, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=focal_loss(), metrics=['accuracy'])
    return model

def load_and_preprocess(csv_path, limit=15000):
    """
    Robustly loads CSV/TSV data and handles the provided headers.
    """
    try:
        # Try reading with auto-detection first
        try:
            df = pd.read_csv(csv_path, sep=None, engine='python')
        except:
            # Fallback for specifically messy whitespace
            df = pd.read_csv(csv_path, sep='\s+', engine='python')
        
        # If auto-detection results in only 1 column, it might be tab-separated but didn't pick up well
        if len(df.columns) <= 1:
             df = pd.read_csv(csv_path, sep='\s+', engine='python')
             if len(df.columns) <= 1:
                  df = pd.read_csv(csv_path, sep='\t', engine='python')
             
        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # Mapping common variations to our required names
        rename_map = {
            'rawtimestamp': 'timestamp',
            'timestamp': 'timestamp',
            'time': 'timestamp',
            'date': 'timestamp',
            'datetime': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'vol': 'volume'
        }
        
        # Keep only what we need and rename
        existing_cols = {c: rename_map[c] for c in df.columns if c in rename_map}
        df = df.rename(columns=existing_cols)
        
        # DROP DUPLICATES: Essential for health of LSTM training
        if 'timestamp' in df.columns:
            # Sort by timestamp and drop duplicates
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        else:
            # Fallback to dropping exact duplicates if no timestamp column
            df = df.drop_duplicates()
            
        # LIMIT DATA: Focus on most recent rows to prevent memory limits / optimize inference
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
            
        print(f"Data cleaned. Proceeding with {len(df)} rows.")

        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            missing = [col for col in required if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train(csv_path):
    from sklearn.model_selection import train_test_split

    df = load_and_preprocess(csv_path, limit=15000)
    if df is None:
        return

    df = calculate_indicators(df)

    window_size = 50
    horizon = 2

    X, y, scaler = prepare_train_data(df, window_size, horizon=horizon)

    print(f"Sliding window applied: Created {len(X)} training sequences.")

    # Train/val split for threshold tuning (time-ordered split to avoid leakage)
    # X is already built from sequential windows; random shuffling breaks temporal validity.
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = build_model((X.shape[1], X.shape[2]))

    # Ensure we don't compute class weights on empty splits
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}


    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=50,
        verbose=1,
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        validation_data=(X_val, y_val),
        shuffle=True,
    )

    # Find best threshold on validation set (optimize F1 instead of accuracy)
    # Accuracy can be misleading when classes are imbalanced.
    val_probs = model.predict(X_val, verbose=0).reshape(-1)
    y_val = y_val.reshape(-1).astype(int)

    def f1_score_from_preds(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        return 2 * precision * recall / (precision + recall + 1e-12)

    # Threshold sweep on validation set (optimize F1), but avoid degenerate thresholds
    # that make the model predict only one class (common when training/val are tiny).
    best_thr = 0.5
    best_f1 = -1.0

    # Guardrail: require that the model predicts UP for a non-trivial fraction.
    # If not, the threshold is likely an artifact of tiny/noisy validation.
    min_positive_ratio = 0.2
    max_positive_ratio = 0.8

    for thr in np.linspace(0.05, 0.95, 181):
        y_pred = (val_probs >= thr).astype(int)

        positive_ratio = float(np.mean(y_pred))  # fraction of UP predictions
        if positive_ratio < min_positive_ratio or positive_ratio > max_positive_ratio:
            continue

        f1 = f1_score_from_preds(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    # If guardrails rejected everything, fall back to the best F1 without guardrails.
    if best_f1 < 0:
        for thr in np.linspace(0.05, 0.95, 181):
            y_pred = (val_probs >= thr).astype(int)
            f1 = f1_score_from_preds(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)



    # Persist threshold and horizon for consistent inference
    import json

    with open(THRESHOLD_PATH, 'w', encoding='utf-8') as f:
        json.dump({'threshold': best_thr, 'horizon': horizon, 'window_size': window_size}, f)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Training complete. Best threshold={best_thr:.2f}, val_best_f1={best_f1*100:.2f}%")


    tf.keras.backend.clear_session()
    gc.collect()


def predict(csv_path):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or scaler not found. Please train the model first.")
        return

    if not os.path.exists(THRESHOLD_PATH):
        print("Error: threshold.json not found. Train again so threshold is saved.")
        return


    # For prediction, we only need a max of 200 candles to warm up the indicators
    df = load_and_preprocess(csv_path, limit=200)
    if df is None:
        return

    df = calculate_indicators(df)
    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})

    import json
    with open(THRESHOLD_PATH, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    threshold = float(cfg.get('threshold', 0.5))
    horizon = int(cfg.get('horizon', 2))
    window_size = int(cfg.get('window_size', 50))

    if len(df) < window_size + horizon:
        print(f"Error: Need at least {window_size + horizon} candles for horizon={horizon} prediction.")
        return

    # Rolling prediction: for each window ending at t, predict whether close(t+horizon) > close(t)
    X_all = []
    y_close_t = []
    for i in range(window_size, len(df) - horizon):
        window = df[FEATURES].iloc[i - window_size:i].values
        scaled_window = scaler.transform(window)
        X_all.append(scaled_window)
        y_close_t.append(float(df['close'].iloc[i]))

    X_all = np.array(X_all)
    probs = model.predict(X_all, verbose=0).reshape(-1)

    directions = np.where(probs >= threshold, "UP", "DOWN")
    confidences = np.where(probs >= threshold, probs, 1 - probs)

    # Output the last prediction aligned with the most recent available time t
    direction = directions[-1]
    confidence = float(confidences[-1])

    print(f"RESULT_DIRECTION:{direction}")
    print(f"RESULT_CONFIDENCE:{confidence * 100:.2f}")
    print(f"THRESHOLD_USED:{threshold:.4f}")
    print(f"HORIZON_USED:{horizon}")

    tf.keras.backend.clear_session()
    gc.collect()


def main():
    if len(sys.argv) < 3:
        print("Usage: python predictor.py <train|predict> <csv_file>")
        return
    
    mode = sys.argv[1]
    csv_file = sys.argv[2]
    
    if mode == "train":
        train(csv_file)
    elif mode == "predict":
        predict(csv_file)
    else:
        print("Unknown mode. Use 'train' or 'predict'.")

if __name__ == "__main__":
    main()






