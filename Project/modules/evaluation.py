# ==========================================================================
# Evaluation - Metrike i evaluacija modela
# ==========================================================================

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classification(y_true, y_pred, class_names, model_name):
    """
    Kompletna evaluacija klasifikacionih rezultata.
    
    Prikazuje:
    - Ukupnu tačnost (Accuracy)
    - Classification report (precision, recall, F1-score po klasama)
    - Confusion matrix
    - Distribuciju predikcija
    - Upozorenje ako model ne koristi sve klase
    
    Args:
        y_true (np.array): Prave labele
        y_pred (np.array): Predikovane labele
        class_names (list): Nazivi klasa ['Pad', 'Stabilno', 'Rast']
        model_name (str): Naziv modela za ispis
        
    Returns:
        float: Tačnost (accuracy) u procentima
    """
    acc = np.mean(y_pred == y_true) * 100
    n_pred_classes = len(np.unique(y_pred))

    print("\n" + "=" * 70)
    print(f"  REZULTATI: {model_name}")
    print("=" * 70)
    print(f"\n  Tacnost (Accuracy): {acc:.2f}%")
    
    # Classification Report
    print(f"\n  Detaljna klasifikacija:\n")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("  Matrica konfuzije:")
    print(f"  {'':>14} Pred:Pad  Pred:Stab  Pred:Rast")
    for i, row in enumerate(cm):
        print(f"    Real:{class_names[i]:>8}  {row[0]:>7}  {row[1]:>9}  {row[2]:>9}")

    # Distribucija predikcija
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    print(f"\n  Distribucija predikcija:")
    for cls, cnt in zip(unique_pred, counts_pred):
        pct = cnt / len(y_pred) * 100
        print(f"    {class_names[int(cls)]}: {cnt} ({pct:.1f}%)")

    # Provera da li model koristi sve 3 klase
    if n_pred_classes < 3:
        print(f"\n  [!] UPOZORENJE: Model predvidja samo {n_pred_classes}/3 klase!")
    else:
        print(f"\n  [OK] Model predvidja sve 3 klase.")
    
    print("=" * 70)
    return acc
