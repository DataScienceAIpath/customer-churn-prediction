"""Train 4 classification models and return fitted estimators."""

import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ModelResult = dict  # {name, model, train_time_s}


def _models(scale_pos_weight: float = 1.0) -> list[tuple[str, object]]:
    return [
        ("Logistic Regression", Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, C=1.0, random_state=42)),
        ])),
        ("Decision Tree", DecisionTreeClassifier(max_depth=6, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=8,
                                                   n_jobs=-1, random_state=42)),
        ("XGBoost", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    scale_pos_weight=scale_pos_weight,
                                    eval_metric="logloss", use_label_encoder=False,
                                    random_state=42, verbosity=0)),
    ]


def train_all(X_train: pd.DataFrame, y_train: pd.Series) -> list[ModelResult]:
    # Compute class imbalance ratio for XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    results = []
    for name, model in _models(scale_pos_weight=spw):
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        results.append({"name": name, "model": model, "train_time_s": round(elapsed, 3)})
    return results
