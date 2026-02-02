import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy.stats import ttest_ind

matplotlib.use("Agg")

# Set global matplotlib parameters for publication-quality figures
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


PRIMARY_METRIC = "accuracy"
SAMPLE_METRIC_KEY = "eval/accuracy_pi_future_sample"
SAMPLE_SUMMARY_KEY = "eval/accuracy_pi_future_samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate WandB runs and generate figures."
    )
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument(
        "--run_ids",
        type=str,
        required=True,
        help='JSON string list of run IDs, e.g. ["run-1", "run-2"]',
    )
    return parser.parse_args()


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def plot_learning_curve(history: pd.DataFrame, metric: str, out_path: Path) -> bool:
    if metric not in history.columns:
        return False
    series = history[metric].dropna()
    if series.empty:
        return False
    x_vals = history["_step"] if "_step" in history.columns else history.index
    x_vals = x_vals[series.index]  # Align x_vals with non-null series

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=x_vals, y=series, linewidth=2)

    # Clean up metric name for title
    metric_display = metric.replace("_", " ").replace("/", " / ").title()
    plt.title(f"{metric_display} Over Steps", fontsize=16, fontweight="bold")
    plt.xlabel("Step", fontsize=14)
    plt.ylabel(metric_display, fontsize=14)

    # Add final value annotation
    if len(series) > 0:
        final_val = series.iloc[-1]
        final_x = (
            x_vals.iloc[-1] if hasattr(x_vals, "iloc") else x_vals[series.index[-1]]
        )
        plt.annotate(
            f"Final: {final_val:.4f}",
            xy=(final_x, final_val),
            xytext=(10, 10),
            textcoords="offset points",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_prompt_accuracy(summary: Dict, out_path: Path) -> bool:
    prompt_keys = {
        k: v
        for k, v in summary.items()
        if k.startswith("prompt/")
        and k.endswith("/accuracy_pi_future")
        and isinstance(v, (int, float))
    }
    if not prompt_keys:
        return False
    prompt_names = [k.split("/")[1] for k in prompt_keys.keys()]
    values = list(prompt_keys.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        prompt_names, values, color="steelblue", edgecolor="black", linewidth=1.5
    )

    plt.title(
        "Prompt Accuracy under $\\pi_{future}$", fontsize=16, fontweight="bold", pad=20
    )
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Prompt Type", fontsize=14)

    # Add value labels on top of bars
    for idx, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        offset = (
            max(0.01 * (max(values) - min(values)), 0.01)
            if max(values) != min(values)
            else 0.01
        )
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_unknown_mass_curve(summary: Dict, out_path: Path) -> bool:
    curve = summary.get("eval/unknown_mass_audit_curve")
    if not isinstance(curve, list) or not curve:
        return False
    xs = [point["alpha0_unknown_mass"] for point in curve]
    ys = [point["accuracy"] for point in curve]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=2, markersize=8, color="steelblue")

    plt.title("Accuracy vs UNKNOWN Mass (BRaCHS Audit)", fontsize=16, fontweight="bold")
    plt.xlabel("UNKNOWN Mass (α₀)", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)

    # Add value labels
    y_range = max(ys) - min(ys) if len(ys) > 1 and max(ys) != min(ys) else 0.02
    offset = max(0.02 * y_range, 0.005)
    for x, y in zip(xs, ys):
        plt.text(x, y + offset, f"{y:.3f}", ha="center", va="bottom", fontsize=11)

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_confusion_matrix(summary: Dict, out_path: Path) -> bool:
    matrix = summary.get("eval/confusion_matrix")
    labels = summary.get("eval/confusion_matrix_labels")
    if matrix is None or labels is None:
        return False
    cm = np.asarray(matrix)
    if cm.size == 0:
        return False

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        linecolor="gray",
    )
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_distribution(values: List[float], out_path: Path, title: str) -> bool:
    if not values:
        return False

    plt.figure(figsize=(8, 5))
    sns.histplot(
        values, kde=True, bins=10, color="steelblue", edgecolor="black", linewidth=1.5
    )
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_comparison_bar(
    metrics: Dict[str, Dict[str, float]], metric_key: str, out_path: Path
) -> bool:
    if metric_key not in metrics:
        return False
    run_ids = list(metrics[metric_key].keys())
    values = list(metrics[metric_key].values())

    # Shorten run IDs for better display
    short_labels = []
    for rid in run_ids:
        if "comparative" in rid:
            parts = rid.replace("comparative-1-", "C1-").split("-commonsenseqa")[0]
            short_labels.append(parts)
        elif "proposed" in rid:
            parts = rid.replace("proposed-", "P-").split("-commonsenseqa")[0]
            short_labels.append(parts)
        else:
            short_labels.append(rid[:20])

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        range(len(values)), values, color="steelblue", edgecolor="black", linewidth=1.5
    )

    # Color bars differently for comparative vs proposed
    for i, (bar, rid) in enumerate(zip(bars, run_ids)):
        if "proposed" in rid:
            bar.set_color("darkgreen")
        else:
            bar.set_color("coral")

    plt.title(
        f"{metric_key.replace('_', ' ').title()} Comparison",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.ylabel(metric_key.replace("_", " ").title(), fontsize=14)
    plt.xlabel("Run ID", fontsize=14)

    # Set x-axis labels
    plt.xticks(
        range(len(short_labels)), short_labels, rotation=45, ha="right", fontsize=11
    )

    # Add value labels on top of bars
    y_range = (
        max(values) - min(values)
        if max(values) != min(values)
        else max(abs(max(values)), 0.1)
    )
    offset = max(0.02 * y_range, 0.005)
    for idx, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_comparison_box(values_by_run: Dict[str, List[float]], out_path: Path) -> bool:
    if not values_by_run:
        return False
    rows = []
    for run_id, values in values_by_run.items():
        for v in values:
            rows.append({"run_id": run_id, "value": v})
    if not rows:
        return False
    df = pd.DataFrame(rows)

    # Shorten run IDs for better display
    short_label_map = {}
    for rid in df["run_id"].unique():
        if "comparative" in rid:
            short_label_map[rid] = rid.replace("comparative-1-", "C1-").split(
                "-commonsenseqa"
            )[0]
        elif "proposed" in rid:
            short_label_map[rid] = rid.replace("proposed-", "P-").split(
                "-commonsenseqa"
            )[0]
        else:
            short_label_map[rid] = rid[:20]
    df["short_id"] = df["run_id"].apply(lambda x: short_label_map[x])

    plt.figure(figsize=(12, 6))

    # Create box plot with custom colors
    unique_ids = df["run_id"].unique().tolist()
    box_data = [df[df["run_id"] == rid]["value"].tolist() for rid in unique_ids]
    box_labels = [short_label_map[rid] for rid in unique_ids]

    box_parts = plt.boxplot(
        box_data, tick_labels=box_labels, patch_artist=True, widths=0.6
    )

    # Color boxes differently for comparative vs proposed
    for patch, rid in zip(box_parts["boxes"], unique_ids):
        if "proposed" in rid:
            patch.set_facecolor("lightgreen")
            patch.set_edgecolor("darkgreen")
        else:
            patch.set_facecolor("lightcoral")
            patch.set_edgecolor("darkred")
        patch.set_linewidth(2)

    # Style other elements
    for element in ["whiskers", "fliers", "means", "medians", "caps"]:
        if element in box_parts:
            plt.setp(box_parts[element], color="black", linewidth=1.5)

    plt.title("Primary Metric Distribution", fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Run ID", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_metric_table(metrics: Dict[str, Dict[str, float]], out_path: Path) -> bool:
    if not metrics:
        return False
    df = pd.DataFrame(metrics).T

    # Shorten column names for better display
    short_cols = []
    for col in df.columns:
        if "comparative" in col:
            short_cols.append(
                col.replace("comparative-1-", "C1-").split("-commonsenseqa")[0]
            )
        elif "proposed" in col:
            short_cols.append(col.replace("proposed-", "P-").split("-commonsenseqa")[0])
        else:
            short_cols.append(col[:15])

    plt.figure(figsize=(14, 0.5 * len(df) + 2))
    plt.axis("off")

    # Create table with better formatting
    table = plt.table(
        cellText=np.round(df.values, 4),
        rowLabels=df.index,
        colLabels=short_cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header row
    for i in range(len(short_cols)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style row labels
    for i in range(len(df)):
        table[(i + 1, -1)].set_facecolor("#E0E0E0")
        table[(i + 1, -1)].set_text_props(weight="bold")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_significance_heatmap(matrix: pd.DataFrame, out_path: Path) -> bool:
    if matrix.empty:
        return False

    # Shorten labels for better display
    short_labels = []
    for label in matrix.index:
        if "comparative" in label:
            short_labels.append(
                label.replace("comparative-1-", "C1-").split("-commonsenseqa")[0]
            )
        elif "proposed" in label:
            short_labels.append(
                label.replace("proposed-", "P-").split("-commonsenseqa")[0]
            )
        else:
            short_labels.append(label[:15])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        cmap="RdYlGn_r",
        fmt=".3f",
        vmin=0,
        vmax=1,
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "p-value"},
    )
    plt.title("Pairwise t-test p-values", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Run ID", fontsize=14)
    plt.ylabel("Run ID", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def is_trial_run(config: Dict) -> bool:
    if config.get("mode") == "trial":
        return True
    wandb_cfg = config.get("wandb")
    if isinstance(wandb_cfg, dict) and wandb_cfg.get("mode") == "disabled":
        return True
    if config.get("wandb.mode") == "disabled":
        return True
    return False


def extract_distribution(
    history: pd.DataFrame, summary: Dict, primary_key: str
) -> List[float]:
    if SAMPLE_METRIC_KEY in history.columns:
        series = history[SAMPLE_METRIC_KEY].dropna()
        if len(series) >= 2:
            return [float(v) for v in series.tolist()]
    if primary_key in history.columns:
        series = history[primary_key].dropna()
        if len(series) >= 2:
            return [float(v) for v in series.tolist()]
    summary_samples = summary.get(SAMPLE_SUMMARY_KEY)
    if isinstance(summary_samples, list) and len(summary_samples) >= 2:
        return [float(v) for v in summary_samples]
    scalar = summary.get(primary_key)
    if isinstance(scalar, (int, float)) and not math.isnan(float(scalar)):
        return [float(scalar)]
    return []


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    entity = cfg.wandb.entity
    project = cfg.wandb.project

    api = wandb.Api()

    generated_files: List[str] = []
    summaries: Dict[str, Dict] = {}
    histories: Dict[str, pd.DataFrame] = {}
    configs: Dict[str, Dict] = {}
    distributions: Dict[str, List[float]] = {}

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()
        summary = run.summary._json_dict
        config = dict(run.config)

        if is_trial_run(config):
            raise RuntimeError(
                f"Run {run_id} was executed in trial mode; evaluation is disabled."
            )

        summaries[run_id] = summary
        histories[run_id] = history
        configs[run_id] = config

        out_dir = results_dir / run_id
        metrics_path = out_dir / "metrics.json"
        payload = {
            "run_id": run_id,
            "summary": summary,
            "config": config,
            "history": history.to_dict(orient="records"),
        }
        save_json(metrics_path, payload)
        generated_files.append(str(metrics_path))

        for metric in [
            "train/loss",
            "eval/accuracy_pi_future",
            "eval/validity_pi_future",
            "eval/token_p95_test",
            "eval/drift_generalization_gap",
        ]:
            lc_path = (
                out_dir / f"{run_id}_learning_curve_{metric.replace('/', '_')}.pdf"
            )
            if plot_learning_curve(history, metric, lc_path):
                generated_files.append(str(lc_path))

        prompt_path = out_dir / f"{run_id}_prompt_accuracy_bar.pdf"
        if plot_prompt_accuracy(summary, prompt_path):
            generated_files.append(str(prompt_path))

        audit_path = out_dir / f"{run_id}_unknown_mass_audit_curve.pdf"
        if plot_unknown_mass_curve(summary, audit_path):
            generated_files.append(str(audit_path))

        cm_path = out_dir / f"{run_id}_confusion_matrix.pdf"
        if plot_confusion_matrix(summary, cm_path):
            generated_files.append(str(cm_path))

        distribution = extract_distribution(history, summary, "eval/accuracy_pi_future")
        distributions[run_id] = distribution
        dist_path = out_dir / f"{run_id}_accuracy_distribution.pdf"
        if plot_distribution(distribution, dist_path, "Accuracy sample distribution"):
            generated_files.append(str(dist_path))

    metrics: Dict[str, Dict[str, float]] = {}
    for run_id, summary in summaries.items():
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not (
                isinstance(value, float) and math.isnan(value)
            ):
                metrics.setdefault(key, {})[run_id] = float(value)

    primary_key = PRIMARY_METRIC
    primary_values = metrics.get(primary_key)
    if not primary_values:
        primary_key = "eval/accuracy_pi_future"
        primary_values = metrics.get(primary_key, {})

    best_proposed = {"run_id": None, "value": None}
    best_baseline = {"run_id": None, "value": None}

    for run_id, val in primary_values.items():
        if "proposed" in run_id:
            if best_proposed["value"] is None or val > best_proposed["value"]:
                best_proposed = {"run_id": run_id, "value": val}
        if "comparative" in run_id or "baseline" in run_id:
            if best_baseline["value"] is None or val > best_baseline["value"]:
                best_baseline = {"run_id": run_id, "value": val}

    gap = None
    if (
        best_proposed["run_id"]
        and best_baseline["run_id"]
        and best_baseline["value"] not in (0, None)
    ):
        gap = (
            (best_proposed["value"] - best_baseline["value"])
            / best_baseline["value"]
            * 100
        )

    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    agg_path = comparison_dir / "aggregated_metrics.json"
    aggregated_payload = {
        "primary_metric": PRIMARY_METRIC,
        "metrics": metrics,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
    }
    save_json(agg_path, aggregated_payload)
    generated_files.append(str(agg_path))

    comp_bar_path = comparison_dir / "comparison_accuracy_bar_chart.pdf"
    if plot_comparison_bar(metrics, primary_key, comp_bar_path):
        generated_files.append(str(comp_bar_path))

    values_by_run = {
        run_id: values for run_id, values in distributions.items() if values
    }
    if not values_by_run:
        for run_id, summary in summaries.items():
            if primary_key in summary:
                values_by_run[run_id] = [float(summary[primary_key])]

    box_path = comparison_dir / "comparison_accuracy_box_plot.pdf"
    if plot_comparison_box(values_by_run, box_path):
        generated_files.append(str(box_path))

    table_keys = [
        "accuracy",
        "eval/accuracy_pi_future",
        "eval/validity_pi_future",
        "eval/token_p95_test",
        "eval/drift_generalization_gap",
        "eval/selection_stability_bootstrap",
    ]
    table_metrics = {k: metrics[k] for k in table_keys if k in metrics}
    table_path = comparison_dir / "comparison_metric_table.pdf"
    if plot_metric_table(table_metrics, table_path):
        generated_files.append(str(table_path))

    run_list = list(distributions.keys())
    p_matrix = pd.DataFrame(index=run_list, columns=run_list, dtype=float)
    for i, run_a in enumerate(run_list):
        for j, run_b in enumerate(run_list):
            if i >= j:
                continue
            vals_a = distributions.get(run_a, [])
            vals_b = distributions.get(run_b, [])
            if len(vals_a) > 1 and len(vals_b) > 1:
                _, p_val = ttest_ind(vals_a, vals_b, equal_var=False)
                p_matrix.loc[run_a, run_b] = float(p_val)
                p_matrix.loc[run_b, run_a] = float(p_val)

    heatmap_path = comparison_dir / "comparison_significance_heatmap.pdf"
    if plot_significance_heatmap(p_matrix, heatmap_path):
        generated_files.append(str(heatmap_path))

    for path in generated_files:
        print(path)


if __name__ == "__main__":
    main()
