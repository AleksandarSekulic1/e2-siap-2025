# ==========================================================================
# Feature Engineering - Kreiranje tehnickih indikatora
# ==========================================================================

import numpy as np
import pandas as pd


def engineer_features(df):
    """
    Kreiranje tehnickih indikatora na osnovu sirovih cena.
    
    Kreira 12 tehnickih indikatora:
    - MA20, EMA20: Moving averages
    - RSI14: Relative Strength Index
    - MACD: Moving Average Convergence Divergence
    - ATR14: Average True Range
    - BB_UP, BB_LO: Bollinger Bands
    - OBV: On-Balance Volume
    - Price_Change, Volatility, Volume_Change
    
    Args:
        df (pd.DataFrame): DataFrame sa OHLCV podacima
        
    Returns:
        pd.DataFrame: DataFrame sa dodatim tehnickim indikatorima
    """
    df = df.copy()
    
    # Moving Averages
    df['MA20']  = df['Close'].rolling(20).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    up   = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    df['RSI14'] = 100 - 100 / (1 + up.ewm(com=13).mean() / down.ewm(com=13).mean())

    # MACD
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()

    # ATR (Average True Range)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()

    # Bollinger Bands
    mid = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_UP'] = mid + 2 * std
    df['BB_LO'] = mid - 2 * std

    # Volume-based indicators
    df['OBV'] = (np.sign(delta) * df['Volume']).cumsum()
    
    # Price metrics
    df['Price_Change']  = (df['Close'] - df['Open']) / df['Open']
    df['Volatility']    = df['High'] - df['Low']
    df['Volume_Change'] = df['Volume'].diff()

    # Uklanjanje NaN vrednosti koje nastaju zbog rolling kalkulacija
    df.dropna(inplace=True)
    
    return df
