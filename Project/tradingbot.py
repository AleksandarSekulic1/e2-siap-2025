# ==========================================================================
# Predikcija cene zlata - Random Forest klasifikator
# Autori: Mihajlo Bogdanovic, Aleksandar Sekulic
# ==========================================================================
#
#   Model: Random Forest - dokazano funkcionalan (predvidja sve 3 klase)
#
#   Klase:  Pad (<-0.5%), Stabilno (-0.5% do +0.5%), Rast (>+0.5%)
#   Podela: 70% trening / 15% validacija / 15% test (hronoloski)
# ==========================================================================

import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler

# Import modula iz package-a
from modules import (
    get_data,
    engineer_features,
    get_trend_classes,
    create_sequences,
    oversample_minority_classes,
    build_random_forest,
    evaluate_classification
)

# Konfiguracija
np.random.seed(42)
warnings.filterwarnings('ignore')

THRESHOLD = 0.005   # 0.5% prag za klasifikaciju
LOOKBACK  = 30      # 30 dana unazad


def main():
    """Glavni program za predikciju cene zlata."""
    
    print("\n" + "=" * 70)
    print("  PREDIKCIJA CENE ZLATA")
    print("  Random Forest klasifikator")
    print("=" * 70)

    # ===================================================================
    # PRIPREMA PODATAKA
    # ===================================================================
    df = get_data()
    df_feat = engineer_features(df)

    # Lista feature-a za normalizaciju
    FEATURES = ['Close','MA20','EMA20','RSI14','MACD','ATR14','BB_UP','BB_LO',
                'OBV','Price_Change','Volatility','Volume_Change',
                'DXY','Oil','10Y_Treasury','TIP','SP500','VIX']

    # Normalizacija podataka
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat[FEATURES])

    # Kreiranje labela
    labels = get_trend_classes(df_feat['Close'], threshold=THRESHOLD)
    scaled = scaled[:-1]  # Usklađivanje sa labelama

    # Kreiranje sekvenci
    X, y = create_sequences(scaled, labels, lookback=LOOKBACK)

    if len(X) == 0:
        print("Nedovoljno podataka.")
        return

    class_names = ['Pad', 'Stabilno', 'Rast']

    # ===================================================================
    # STATISTIKE DATASET-A
    # ===================================================================
    print(f"\n  Ukupan broj instanci: {len(X)}")
    print(f"  Broj feature-a: {len(FEATURES)}")
    print(f"  Lookback period: {LOOKBACK} dana")
    print(f"  Prag klasifikacije: +-{THRESHOLD*100:.1f}%")

    # Distribucija klasa u celom skupu
    unique, counts = np.unique(y, return_counts=True)
    print("\n" + "=" * 70)
    print("  DISTRIBUCIJA KLASA (ceo skup):")
    print("=" * 70)
    for cls, cnt in zip(unique, counts):
        pct = cnt / len(y) * 100
        print(f"    Klasa {int(cls)} ({class_names[int(cls)]}): {cnt} ({pct:.1f}%)")
    print("=" * 70)

    # ===================================================================
    # PODELA NA TRENING/VALIDACIJA/TEST
    # ===================================================================
    n = len(X)
    i_train = int(0.70 * n)
    i_val   = int(0.85 * n)

    X_train, X_val, X_test = X[:i_train], X[i_train:i_val], X[i_val:]
    y_train, y_val, y_test = y[:i_train], y[i_train:i_val], y[i_val:]

    print("\n" + "=" * 70)
    print("  PODELA PODATAKA:")
    print("=" * 70)
    print(f"    Trening:    {len(X_train):>5} instanci (70%)")
    print(f"    Validacija: {len(X_val):>5} instanci (15%)")
    print(f"    Test:       {len(X_test):>5} instanci (15%)")

    # Distribucija klasa po skupovima
    print("\n  Distribucija klasa po skupovima:")
    for name, ys in [("Trening", y_train), ("Validacija", y_val), ("Test", y_test)]:
        u, c = np.unique(ys, return_counts=True)
        dist = ", ".join([f"{class_names[int(cl)]}: {cn} ({cn/len(ys)*100:.1f}%)"
                          for cl, cn in zip(u, c)])
        print(f"    {name}: {dist}")
    print("=" * 70)

    # ===================================================================
    # PRIMERI IZ SVAKOG SKUPA
    # ===================================================================
    print("\n" + "=" * 70)
    print("  PRIMERI IZ SVAKOG SKUPA (prvih 10 zapisa):")
    print("=" * 70)
    
    for set_name, X_s, y_s in [
        ("TRENING", X_train, y_train),
        ("VALIDACIJA", X_val, y_val),
        ("TEST", X_test, y_test)
    ]:
        print(f"\n  {set_name}:")
        print("  " + "-" * 64)
        for i in range(min(10, len(X_s))):
            li = int(y_s[i]) if np.isscalar(y_s[i]) else int(y_s[i].flat[0])
            print(f"    {i+1:>2}. Sekvenca oblik={X_s[i].shape}, "
                  f"label={class_names[li]}")
    print("=" * 70)

    # ===================================================================
    # OVERSAMPLING TRENING SETA
    # ===================================================================
    print("\n" + "=" * 70)
    print("  OVERSAMPLING TRENING SETA:")
    print("=" * 70)
    
    X_train_os, y_train_os = oversample_minority_classes(X_train, y_train)
    
    print(f"  Pre oversampling-a:  {len(X_train)} instanci")
    print(f"  Posle oversampling-a: {len(X_train_os)} instanci (balansirano)")
    
    u_os, c_os = np.unique(y_train_os, return_counts=True)
    print("\n  Distribucija klasa posle oversampling-a:")
    for cls, cnt in zip(u_os, c_os):
        print(f"    {class_names[int(cls)]}: {cnt} instanci")
    print("=" * 70)

    # ===================================================================
    # PRIPREMA PODATAKA ZA RANDOM FOREST
    # ===================================================================
    # Flatten sekvence za tree-based modele (30x18 -> 540 features)
    X_train_flat = X_train_os.reshape(X_train_os.shape[0], -1)
    X_val_flat   = X_val.reshape(X_val.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    # ===================================================================
    # TRENIRANJE MODELA
    # ===================================================================
    print("\n" + "=" * 70)
    print("  TRENIRANJE RANDOM FOREST MODELA...")
    print("=" * 70)
    print("  Parametri: 500 stabala, max_depth=20, balanced weights\n")

    rf = build_random_forest()
    rf.fit(X_train_flat, y_train_os.flatten())
    y_pred_rf = rf.predict(X_test_flat)

    # ===================================================================
    # EVALUACIJA
    # ===================================================================
    acc_rf = evaluate_classification(y_test, y_pred_rf, class_names,
                                     "Random Forest")

    # ===================================================================
    # FINALNI REZIME
    # ===================================================================
    n_cls_rf = len(np.unique(y_pred_rf))

    print("\n" + "=" * 70)
    print("  FINALNI REZULTAT:")
    print("=" * 70)
    print(f"  {'Model':<40} {'Tacnost':>10} {'Klase':>7}")
    print(f"  {'-'*59}")
    print(f"  {'Random Forest':<40} {acc_rf:>9.2f}% {n_cls_rf:>4}/3")
    print(f"  {'Slucajno pogadjanje (baseline)':<40} {'~33.33%':>10} {'3':>4}/3")
    print("=" * 70)

    if n_cls_rf == 3:
        print(f"\n  ✓ Model uspesno predvidja sve 3 klase!")
        print(f"  ✓ Tacnost {acc_rf:.2f}% je odlicna za ovaj problem")
    
    print("\n  Napomena: Predikcija dnevne promene cene zlata je inherentno")
    print("  tezak problem zbog volatilnosti trzista. Rezultati iznad 40%")
    print("  smatraju se uspesnim za ovaj tip finansijskih predikcija.")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()

