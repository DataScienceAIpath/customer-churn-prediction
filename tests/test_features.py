from src.data.generator import generate_churn_data
from src.features.engineering import build_features


def test_build_features_shape():
    df = generate_churn_data(n=200)
    X, y = build_features(df)
    assert len(X) == 200
    assert len(y) == 200


def test_no_text_columns():
    df = generate_churn_data(n=100)
    X, _ = build_features(df)
    text_cols = X.select_dtypes(include="object").columns.tolist()
    assert text_cols == [], f"Unexpected text columns: {text_cols}"


def test_target_not_in_X():
    df = generate_churn_data(n=100)
    X, y = build_features(df)
    assert "churned" not in X.columns


def test_engineered_features_present():
    df = generate_churn_data(n=100)
    X, _ = build_features(df)
    assert "avg_monthly_spend" in X.columns
    assert "engagement_score" in X.columns
    assert "high_value" in X.columns
    assert "contract_encoded" in X.columns
