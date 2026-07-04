"""CLI for Customer Churn Prediction pipeline."""

from pathlib import Path

import typer
from rich.console import Console

from .data.generator import load_or_generate
from .eda.analysis import run_eda
from .features.engineering import build_features
from .models.trainer import train_all
from .models.evaluator import evaluate_all

app = typer.Typer(help="Customer Churn Prediction — Classification Pipeline")
console = Console()

DATA_DIR = Path("data")
DOCS_DIR = Path("docs")


@app.command()
def generate(n: int = typer.Option(7000, help="Number of customers to generate")):
    """Generate synthetic churn dataset and save to data/churn_data.csv."""
    DATA_DIR.mkdir(exist_ok=True)
    df = load_or_generate(DATA_DIR, n)
    churn_rate = df["churned"].mean()
    console.print(f"[green]Dataset ready:[/green] {len(df):,} rows | "
                  f"churn rate {churn_rate:.1%} → {DATA_DIR}/churn_data.csv")


@app.command()
def eda():
    """Run minimal EDA: class balance, tenure distributions, churn by segment."""
    df = load_or_generate(DATA_DIR)
    run_eda(df, DOCS_DIR)


@app.command()
def train():
    """Feature engineering + train 4 models + print comparison table."""
    from sklearn.model_selection import StratifiedShuffleSplit

    df = load_or_generate(DATA_DIR)
    X, y = build_features(df)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    console.print(f"[cyan]Train:[/cyan] {len(X_train):,}  [cyan]Test:[/cyan] {len(X_test):,}  "
                  f"[cyan]Features:[/cyan] {X_train.shape[1]}  "
                  f"[cyan]Churn rate (test):[/cyan] {y_test.mean():.1%}")

    console.print("\n[bold]Training models...[/bold]")
    results = train_all(X_train, y_train)
    evaluate_all(results, X_test, y_test, DOCS_DIR)


@app.command()
def run_all():
    """End-to-end: generate → EDA → train → evaluate."""
    console.print("[bold cyan]Step 1/3 — Generating dataset[/bold cyan]")
    generate(n=7000)
    console.print("\n[bold cyan]Step 2/3 — Running EDA[/bold cyan]")
    eda()
    console.print("\n[bold cyan]Step 3/3 — Training & evaluating models[/bold cyan]")
    train()
    console.print("\n[bold green]Pipeline complete.[/bold green]")


if __name__ == "__main__":
    app()
