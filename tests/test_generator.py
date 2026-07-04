from src.data.generator import generate_churn_data


def test_shape():
    df = generate_churn_data(n=300)
    assert df.shape == (300, 11)


def test_churn_binary():
    df = generate_churn_data(n=500)
    assert set(df["churned"].unique()).issubset({0, 1})


def test_churn_rate_realistic():
    df = generate_churn_data(n=2000)
    rate = df["churned"].mean()
    assert 0.15 <= rate <= 0.45, f"Unexpected churn rate {rate:.2%}"


def test_no_nulls():
    df = generate_churn_data(n=200)
    assert df.isnull().sum().sum() == 0


def test_reproducibility():
    df1 = generate_churn_data(n=100, seed=7)
    df2 = generate_churn_data(n=100, seed=7)
    assert (df1["churned"].values == df2["churned"].values).all()
