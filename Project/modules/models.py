# ==========================================================================
# Models - Definicije masinski-learning modela
# ==========================================================================

from sklearn.ensemble import RandomForestClassifier


def build_random_forest():
    """
    Kreira Random Forest klasifikator sa optimizovanim parametrima.
    
    Parametri su podeseni za 3-klasni problem klasifikacije sa:
    - 500 stabala za robusnost
    - max_depth=20 za sprečavanje overfitting-a
    - class_weight='balanced' za rukovanje neujednačenim klasama
    - n_jobs=-1 za paralelizaciju na svim jezgrima
    
    Returns:
        RandomForestClassifier: Konfigurisani model
    """
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
