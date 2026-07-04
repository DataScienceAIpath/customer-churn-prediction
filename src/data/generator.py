"""Synthetic telecom customer churn dataset.

Mirrors real patterns from IBM Telco Churn benchmark:
~26% churn rate, month-to-month contracts churn most, long tenure churns least.
7 000 records.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42

CONTRACT_TYPES = ["Month-to-Month", "One Year", "Two Year"]
INTERNET_SERVICES = ["DSL", "Fiber Optic", "No Internet"]
PAYMENT_METHODS = ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"]


def generate_churn_data(n: int = 7_000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 73, size=n).astype(float)
    monthly_charges = rng.uniform(18, 118, size=n).round(2)
    contract = rng.choice(CONTRACT_TYPES, size=n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET_SERVICES, size=n, p=[0.40, 0.44, 0.16])
    payment = rng.choice(PAYMENT_METHODS, size=n)
    support_calls_6m = rng.integers(0, 11, size=n).astype(float)
    has_streaming = rng.choice([0, 1], size=n, p=[0.48, 0.52]).astype(float)
    has_tech_support = rng.choice([0, 1], size=n, p=[0.50, 0.50]).astype(float)
    num_products = rng.integers(1, 6, size=n).astype(float)
    total_charges = (tenure_months * monthly_charges + rng.normal(0, 50, n)).clip(0).round(2)

    # Churn probability model (logistic-style)
    log_odds = (
        -2.5                                          # base intercept (~26% churn)
        + 1.5 * (contract == "Month-to-Month").astype(float)
        - 1.0 * (contract == "Two Year").astype(float)
        + 0.8 * (internet == "Fiber Optic").astype(float)
        - 0.5 * (internet == "No Internet").astype(float)
        - 0.03 * tenure_months                        # longer tenure → less churn
        + 0.02 * monthly_charges                      # higher charges → more churn
        + 0.25 * support_calls_6m                     # more complaints → more churn
        - 0.3 * has_tech_support
        - 0.2 * num_products
        + 0.6 * (payment == "Electronic Check").astype(float)
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churned = rng.binomial(1, churn_prob).astype(int)

    return pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "support_calls_6m": support_calls_6m,
        "has_streaming": has_streaming,
        "has_tech_support": has_tech_support,
        "num_products": num_products,
        "churned": churned,
    })


def load_or_generate(data_dir: Path = Path("data"), n: int = 7_000) -> pd.DataFrame:
    path = data_dir / "churn_data.csv"
    if path.exists():
        return pd.read_csv(path)
    df = generate_churn_data(n)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
