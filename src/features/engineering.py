"""Feature engineering for churn prediction.

Steps:
  1. avg_monthly_spend = total_charges / tenure_months  (value density)
  2. high_value flag = monthly_charges > 75th percentile
  3. engagement_score = has_streaming + has_tech_support + num_products
  4. Ordinal encode contract_type (M2M=0 → Two Year=2)
  5. One-hot encode internet_service and payment_method
"""

import pandas as pd
import numpy as np

TARGET = "churned"

CONTRACT_ORDER = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for sklearn."""
    df = df.copy()

    # 1. Avg monthly spend (total vs tenure)
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure_months"].clip(lower=1)

    # 2. High-value customer flag
    threshold = df["monthly_charges"].quantile(0.75)
    df["high_value"] = (df["monthly_charges"] > threshold).astype(int)

    # 3. Product engagement score
    df["engagement_score"] = (
        df["has_streaming"] + df["has_tech_support"] + df["num_products"]
    )

    # 4. Ordinal encode contract type
    df["contract_encoded"] = df["contract_type"].map(CONTRACT_ORDER)

    # 5. One-hot encode internet_service, payment_method
    ohe_cols = ["internet_service", "payment_method"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Drop source categorical columns
    df = df.drop(columns=["contract_type"])

    y = df.pop(TARGET)
    return df, y
