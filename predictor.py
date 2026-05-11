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

def prepare_train_data(df, window_size=60):
    data = df[FEATURES].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X = []
    y = []
    
    for i in range(window_size, len(scaled_data) - 2):
        X.append(scaled_data[i-window_size:i])
        # Target: Predict UP moves (price is higher in 2 candles)
        y.append(1 if df['close'].iloc[i+2] > df['close'].iloc[i] else 0)
        
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
    df = load_and_preprocess(csv_path, limit=15000)
    if df is None: return

    df = calculate_indicators(df)
    window_size = 50 
    X, y, scaler = prepare_train_data(df, window_size)
    
    print(f"Sliding window applied: Created {len(X)} training sequences.")
    model = build_model((X.shape[1], X.shape[2]))
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
    
    # Use EarlyStopping to train until convergence without overfitting
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    model.fit(X, y, batch_size=64, epochs=50, verbose=1, callbacks=[early_stop], class_weight=class_weight_dict)
    
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Training complete.")
    
    # Cleanup memory
    tf.keras.backend.clear_session()
    gc.collect()

def predict(csv_path):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or scaler not found. Please train the model first.")
        return

    # For prediction, we only need a max of 200 candles to warm up the indicators
    df = load_and_preprocess(csv_path, limit=200)
    if df is None: return

    df = calculate_indicators(df)
    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss()})
    
    window_size = 50
    if len(df) < window_size:
        print(f"Error: Need at least {window_size} candles for prediction.")
        return

    last_window = df[FEATURES].tail(window_size).values
    scaled_window = scaler.transform(last_window)
    X_input = np.expand_dims(scaled_window, axis=0)
    
    pred = model.predict(X_input, verbose=0)[0][0]
    direction = "UP" if pred >= 0.5 else "DOWN"
    confidence = pred if pred >= 0.5 else (1 - pred)
    
    print(f"RESULT_DIRECTION:{direction}")
    print(f"RESULT_CONFIDENCE:{confidence * 100:.2f}")

    # Cleanup
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
