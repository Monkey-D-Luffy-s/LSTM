try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import os
    import sys
    import joblib
except ImportError as e:
    print(f"[CRITICAL ERROR] Missing Python dependency: {e}")
    print("Please run: pip install pandas numpy scikit-learn tensorflow joblib")
    sys.exit(1)

# To run this script, you will need:
# pip install pandas numpy scikit-learn tensorflow joblib

FEATURES = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'ema', 'vwap', 'volatility']
MODEL_PATH = 'model.keras'
SCALER_PATH = 'scaler.gz'

def calculate_indicators(df):
    """
    Calculates technical indicators: RSI, MACD, EMA, VWAP, and Volatility.
    """
    try:
        # EMA (20 periods)
        df['ema'] = df['close'].ewm(span=20, adjust=False).mean()

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

        # Handle NaNs from indicators (like RSI/rolling)
        df.bfill(inplace=True)
        df.ffill(inplace=True)
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
    
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        # Target: Is current close > previous close?
        y.append(1 if df['close'].iloc[i] > df['close'].iloc[i-1] else 0)
        
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_and_preprocess(csv_path):
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
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        # Keep only what we need and rename
        existing_cols = {c: rename_map[c] for c in df.columns if c in rename_map}
        print(f"Detected columns: {list(df.columns)}")
        df = df.rename(columns=existing_cols)
        print(f"Mapped columns: {list(df.columns)}")
        
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            missing = [col for col in required if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train(csv_path):
    df = load_and_preprocess(csv_path)
    if df is None: return

    df = calculate_indicators(df)
    window_size = 100 
    X, y, scaler = prepare_train_data(df, window_size)
    
    print(f"Sliding window applied: Created {len(X)} training sequences from {len(df)} candles.")
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, batch_size=64, epochs=5, verbose=1)
    
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Training complete.")

def predict(csv_path):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or scaler not found. Please train the model first.")
        return

    df = load_and_preprocess(csv_path)
    if df is None: return

    df = calculate_indicators(df)
    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH)
    
    window_size = 100
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
