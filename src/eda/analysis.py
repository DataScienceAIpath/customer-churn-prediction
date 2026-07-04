"""Minimal EDA for churn dataset — class balance, distributions, churn rates by segment."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def run_eda(df: pd.DataFrame, output_dir: Path = Path("docs")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Basic shape ──────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Dataset Shape:[/bold cyan] {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── 2. Class balance ────────────────────────────────────────────────────
    churn_rate = df["churned"].mean()
    churned_n = df["churned"].sum()
    console.print(f"[bold]Class balance:[/bold] {churned_n:,} churned ({churn_rate:.1%}) / "
                  f"{len(df) - churned_n:,} retained ({1-churn_rate:.1%})")

    # ── 3. Missing values ───────────────────────────────────────────────────
    missing = df.isnull().sum()
    if missing.any():
        console.print(f"[yellow]Missing:[/yellow]\n{missing[missing > 0]}")
    else:
        console.print("[green]No missing values.[/green]")

    # ── 4. Numeric summary ──────────────────────────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    tbl = Table(title="Numeric Feature Summary", show_lines=True)
    tbl.add_column("Feature", style="cyan")
    for col in ["mean", "std", "min", "50%", "max"]:
        tbl.add_column(col, justify="right")
    desc = df[num_cols].describe().T
    for feat in num_cols:
        r = desc.loc[feat]
        tbl.add_row(feat, f"{r['mean']:.2f}", f"{r['std']:.2f}",
                    f"{r['min']:.2f}", f"{r['50%']:.2f}", f"{r['max']:.2f}")
    console.print(tbl)

    # ── 5. Churn rate by key segments ───────────────────────────────────────
    for col in ["contract_type", "internet_service", "payment_method"]:
        rates = df.groupby(col)["churned"].mean().sort_values(ascending=False)
        console.print(f"\n[bold]Churn rate by {col}:[/bold]")
        for val, rate in rates.items():
            bar = "█" * int(rate * 40)
            console.print(f"  {val:<25} {rate:.1%}  {bar}")

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Customer Churn — Key Patterns", fontsize=13)

    # Class balance pie
    axes[0].pie([churned_n, len(df) - churned_n], labels=["Churned", "Retained"],
                autopct="%1.1f%%", colors=["#DD8452", "#4C72B0"], startangle=90)
    axes[0].set_title("Class Balance")

    # Tenure distribution by churn
    for label, color in [(0, "#4C72B0"), (1, "#DD8452")]:
        subset = df[df["churned"] == label]["tenure_months"]
        axes[1].hist(subset, bins=24, alpha=0.6, color=color,
                     label="Retained" if label == 0 else "Churned")
    axes[1].set_xlabel("Tenure (months)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Tenure Distribution by Churn")
    axes[1].legend()

    # Churn rate by contract
    rates = df.groupby("contract_type")["churned"].mean()
    axes[2].bar(rates.index, rates.values, color=["#DD8452", "#4C72B0", "#55A868"])
    axes[2].set_ylabel("Churn Rate")
    axes[2].set_title("Churn Rate by Contract Type")
    axes[2].set_ylim(0, 1)
    for i, v in enumerate(rates.values):
        axes[2].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "eda_churn.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    console.print(f"\n[dim]EDA chart saved → {out_path}[/dim]")
