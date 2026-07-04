"""Evaluate classification models and produce comparison table + feature importance chart."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

console = Console()


def evaluate_all(
    model_results: list[dict],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = Path("docs"),
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mr in model_results:
        preds = mr["model"].predict(X_test)
        proba = mr["model"].predict_proba(X_test)[:, 1]
        rows.append({
            "Model": mr["name"],
            "Accuracy": f"{accuracy_score(y_test, preds):.4f}",
            "Precision": f"{precision_score(y_test, preds):.4f}",
            "Recall": f"{recall_score(y_test, preds):.4f}",
            "F1": f"{f1_score(y_test, preds):.4f}",
            "ROC-AUC": f"{roc_auc_score(y_test, proba):.4f}",
            "Train (s)": f"{mr['train_time_s']:.3f}",
            "_auc": roc_auc_score(y_test, proba),
        })

    tbl = Table(title="Classification Model Comparison", show_lines=True)
    for col in ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train (s)"]:
        tbl.add_column(col, justify="right" if col != "Model" else "left",
                       style="bold cyan" if col == "Model" else None)
    best_auc = max(r["_auc"] for r in rows)
    for r in rows:
        style = "bold green" if r["_auc"] == best_auc else None
        tbl.add_row(r["Model"], r["Accuracy"], r["Precision"], r["Recall"],
                    r["F1"], r["ROC-AUC"], r["Train (s)"], style=style)
    console.print(tbl)
    console.print(f"[dim]Best model (ROC-AUC): {max(rows, key=lambda x: x['_auc'])['Model']}[/dim]")

    # Feature importance for best tree-based model
    best = max(model_results, key=lambda m: roc_auc_score(
        y_test, m["model"].predict_proba(X_test)[:, 1]))
    model_obj = best["model"]
    if hasattr(model_obj, "named_steps"):
        model_obj = model_obj.named_steps.get("model", model_obj)
    if hasattr(model_obj, "feature_importances_"):
        importances = pd.Series(model_obj.feature_importances_,
                                index=X_test.columns).sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        importances.plot(kind="barh", ax=ax, color="#DD8452")
        ax.set_title(f"Top 15 Feature Importances — {best['name']}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        out = output_dir / "feature_importance_classification.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        console.print(f"[dim]Feature importance chart → {out}[/dim]")

    return pd.DataFrame([{k: v for k, v in r.items() if k != "_auc"} for r in rows])
