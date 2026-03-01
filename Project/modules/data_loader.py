# ==========================================================================
# Data Loader - Prikupljanje podataka sa Yahoo Finance
# ==========================================================================

import yfinance as yf


def get_data():
    """
    Preuzimanje podataka sa Yahoo Finance za zlato i korelisane instrumente.
    
    Returns:
        pd.DataFrame: DataFrame sa kolonama za zlato (Open, High, Low, Close, Volume)
                      i makroekonomskim faktorima (DXY, Oil, 10Y_Treasury, TIP, SP500, VIX)
    """
    print("\n  Preuzimanje podataka...")
    
    # Osnovni podaci za zlato (GC=F)
    df_gold = yf.download('GC=F', start='2010-01-01', end='2025-05-01', 
                          progress=False)[['Open','High','Low','Close','Volume']]
    
    # Makroekonomski i finansijski faktori
    tickers = {
        'DX-Y.NYB': 'DXY',           # USD Index
        'CL=F': 'Oil',                # Crude Oil
        '^TNX': '10Y_Treasury',       # 10-Year Treasury Yield
        'TIP': 'TIP',                 # TIPS ETF
        '^GSPC': 'SP500',             # S&P 500
        '^VIX': 'VIX'                 # Volatility Index
    }
    
    extras = []
    for ticker, name in tickers.items():
        tmp = yf.download(ticker, start='2010-01-01', end='2025-05-01', 
                         progress=False)[['Close']]
        tmp = tmp.rename(columns={'Close': name})
        extras.append(tmp)
    
    # Spajanje svih podataka
    df = df_gold.join(extras, how='inner')
    df = df.ffill()  # Forward fill za popunjavanje missing vrednosti
    
    return df
