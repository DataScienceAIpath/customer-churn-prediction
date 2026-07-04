from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from src.data.generator import generate_churn_data
from src.features.engineering import build_features
from src.models.trainer import train_all


def _get_split():
    df = generate_churn_data(n=600)
    X, y = build_features(df)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def test_all_models_train():
    X_train, X_test, y_train, y_test = _get_split()
    results = train_all(X_train, y_train)
    assert len(results) == 4


def test_predictions_binary():
    X_train, X_test, y_train, y_test = _get_split()
    results = train_all(X_train, y_train)
    for r in results:
        preds = r["model"].predict(X_test)
        assert set(preds).issubset({0, 1})


def test_roc_auc_above_random():
    X_train, X_test, y_train, y_test = _get_split()
    results = train_all(X_train, y_train)
    for r in results:
        proba = r["model"].predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        assert auc > 0.55, f"{r['name']} ROC-AUC={auc:.3f} — barely above random"
