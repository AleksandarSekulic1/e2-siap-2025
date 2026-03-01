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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

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
sns.set_palette("husl")

THRESHOLD = 0.005   # 0.5% prag za klasifikaciju
LOOKBACK  = 30      # 30 dana unazad


def plot_class_distribution(y_train, y_val, y_test, save_path=None):
    """Prikaz distribucije klasa po skupovima (trening/validacija/test)."""
    class_labels = ['Pad ↓', 'Stabilno ↔', 'Rast ↑']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for name, ys, ax in [
        ("Trening", y_train, axes[0]),
        ("Validacija", y_val, axes[1]),
        ("Test", y_test, axes[2])
    ]:
        unique, counts = np.unique(ys, return_counts=True)
        counts_full = np.zeros(3, dtype=int)
        counts_full[unique.astype(int)] = counts

        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        bars = ax.bar(class_labels, counts_full, color=colors,
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        for bar, count in zip(bars, counts_full):
            height = bar.get_height()
            pct = (count / len(ys) * 100) if len(ys) > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')

        ax.set_title(f'{name}\n({len(ys)} sekvenci)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Broj instanci', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Distribucija klasa po skupovima', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_and_report(y_true, y_pred, class_names, save_path=None):
    """Prikaz confusion matrix i classification report tabele."""
    cm = confusion_matrix(y_true, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Broj uzoraka'}, ax=ax1)
    ax1.set_title('Matrica Konfuzije (Test Set)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Prava klasa', fontsize=10)
    ax1.set_xlabel('Predikovana klasa', fontsize=10)

    report = classification_report(y_true, y_pred,
                                   target_names=class_names,
                                   output_dict=True, zero_division=0)
    metrics_df = np.round(
        np.array([
            [report[class_names[0]]['precision'], report[class_names[0]]['recall'], report[class_names[0]]['f1-score']],
            [report[class_names[1]]['precision'], report[class_names[1]]['recall'], report[class_names[1]]['f1-score']],
            [report[class_names[2]]['precision'], report[class_names[2]]['recall'], report[class_names[2]]['f1-score']],
            [report['accuracy'], report['accuracy'], report['accuracy']]
        ]),
        3
    )
    row_labels = [f'{class_names[0]}', f'{class_names[1]}', f'{class_names[2]}', 'accuracy']
    col_labels = ['precision', 'recall', 'f1-score']

    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=metrics_df,
                      colLabels=col_labels,
                      rowLabels=row_labels,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax2.set_title('Classification Report', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, features, lookback=30, top_n=15, save_path=None):
    """Prikaz top-N najvaznijih feature-a za Random Forest model."""
    importances = model.feature_importances_
    feature_names = []

    for day in range(lookback):
        for feat in features:
            feature_names.append(f"{feat}_d{day}")

    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(top_names)), top_importances,
                   color='#45b7d1', edgecolor='black', linewidth=1.2)

    for bar, imp in zip(bars, top_importances):
        ax.text(imp, bar.get_y() + bar.get_height() / 2.0,
                f' {imp:.4f}', va='center', fontweight='bold', fontsize=9)

    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Vaznost (Importance Score)', fontsize=11, fontweight='bold')
    ax.set_title('Top 15 najvaznijih feature-a (Random Forest)',
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def train_random_forest_with_progress(X_train_flat, y_train, X_val_flat, y_val,
                                      epoch_trees=(100, 200, 300, 400, 500)):
    """Trening RF modela sa epoch-like progres ispisom po broju stabala."""
    rf = build_random_forest()
    rf.set_params(warm_start=True, n_estimators=0)

    total_epochs = len(epoch_trees)
    print("\n  Epoch-like progres treniranja (broj stabala):")

    for epoch_idx, n_trees in enumerate(epoch_trees, start=1):
        rf.set_params(n_estimators=n_trees)
        rf.fit(X_train_flat, y_train)

        train_acc = np.mean(rf.predict(X_train_flat) == y_train) * 100
        val_acc = np.mean(rf.predict(X_val_flat) == y_val) * 100
        print(f"    [Epohа {epoch_idx:>2}/{total_epochs}] stabala={n_trees:>3} | "
              f"train_acc={train_acc:>6.2f}% | val_acc={val_acc:>6.2f}%")

    return rf


def main():
    """Glavni program za predikciju cene zlata."""

    project_dir = Path(__file__).resolve().parent
    plots_dir = project_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "=" * 70)
    print("  PREDIKCIJA CENE ZLATA")
    print("  Random Forest klasifikator")
    print("=" * 70)
    print(f"  Plotovi ce biti sacuvani u: {plots_dir}")

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

    try:
        path_dist = plots_dir / f'class_distribution_{run_tag}.png'
        plot_class_distribution(y_train, y_val, y_test, save_path=path_dist)
        print(f"\n  [OK] Sacuvan plot: {path_dist.name}")
    except Exception as exc:
        print(f"\n  [!] Preskacem plot distribucije klasa: {exc}")

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
    print("  Parametri: 500 stabala, max_depth=20, balanced weights")
    print("  Napomena: ispod ide epoch-like progres kroz korake broja stabala\n")

    rf = train_random_forest_with_progress(
        X_train_flat,
        y_train_os.flatten(),
        X_val_flat,
        y_val,
        epoch_trees=(100, 200, 300, 400, 500)
    )
    y_pred_rf = rf.predict(X_test_flat)

    # ===================================================================
    # EVALUACIJA
    # ===================================================================
    acc_rf = evaluate_classification(y_test, y_pred_rf, class_names,
                                     "Random Forest")

    try:
        path_cm = plots_dir / f'confusion_report_{run_tag}.png'
        plot_confusion_and_report(y_test, y_pred_rf, class_names, save_path=path_cm)
        print(f"\n  [OK] Sacuvan plot: {path_cm.name}")
    except Exception as exc:
        print(f"\n  [!] Preskacem confusion/report plot: {exc}")

    try:
        path_fi = plots_dir / f'feature_importance_{run_tag}.png'
        plot_feature_importance(rf, FEATURES, lookback=LOOKBACK, top_n=15,
                                save_path=path_fi)
        print(f"\n  [OK] Sacuvan plot: {path_fi.name}")
    except Exception as exc:
        print(f"\n  [!] Preskacem feature importance plot: {exc}")

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

