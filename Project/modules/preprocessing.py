# ==========================================================================
# Preprocessing - Labeliranje, sekvenciranje i balansiranje
# ==========================================================================

import numpy as np


def get_trend_classes(close, threshold=0.005):
    """
    Klasifikacija trenda cene zlata na osnovu procentualne promene.
    
    Klase:
    - 0 (Pad): promena < -threshold (-0.5%)
    - 1 (Stabilno): -threshold <= promena <= +threshold
    - 2 (Rast): promena > +threshold (+0.5%)
    
    Args:
        close (pd.Series): Serija Close cena
        threshold (float): Prag za klasifikaciju (default: 0.005 = 0.5%)
        
    Returns:
        np.array: Niz klasa (0, 1, 2) za svaki dan
    """
    future_returns = close.pct_change().shift(-1)
    y = np.select(
        [future_returns < -threshold, future_returns > threshold],
        [0, 2], 
        default=1
    )
    return y[:-1]  # Uklanjanje poslednjeg dana (nema labele)


def create_sequences(data, labels, lookback=30):
    """
    Kreiranje sekvenci za model pomocu sliding window metode.
    
    Za svaki dan kreira sekvencu od 'lookback' prethodnih dana.
    
    Args:
        data (np.array): Normalizovani podaci (n_samples, n_features)
        labels (np.array): Labele za svaki dan
        lookback (int): Broj dana u sekvenci (default: 30)
        
    Returns:
        tuple: (X, y) gde je X shape (n_samples, lookback, n_features)
               i y shape (n_samples,)
    """
    X, y = [], []
    
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(labels[i])
    
    return np.array(X), np.array(y)


def oversample_minority_classes(X, y):
    """
    Balansiranje dataset-a random oversampling-om manjinskih klasa.
    
    Umnozava manjinske klase tako da sve klase imaju jednak broj instanci
    kao najzastupljenija klasa.
    
    Args:
        X (np.array): Ulazni podaci
        y (np.array): Labele
        
    Returns:
        tuple: (X_balanced, y_balanced) sa balansiranim klasama
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    
    X_res, y_res = [X], [y]
    
    for cls in unique_classes:
        mask = (y == cls).flatten()
        n_extra = max_count - mask.sum()
        
        if n_extra > 0:
            # Random sampling sa ponavljanjem
            idx = np.random.choice(np.where(mask)[0], size=n_extra, replace=True)
            X_res.append(X[idx])
            y_res.append(y[idx])
    
    # Spajanje i mesanje
    X_out = np.concatenate(X_res)
    y_out = np.concatenate(y_res)
    shuf = np.random.permutation(len(X_out))
    
    return X_out[shuf], y_out[shuf]
