#!/usr/bin/env python3
"""
Generate multi-line plots that highlight teacher vs student sycophancy behavior
across pressure strengths (and optionally modes).

Usage:
    python sycophancy_multiline_plot.py \
        --input results/nvidia_distill_results_1000.json \
        --output plots/sycophancy_multiline.png \
        --facet-mode \
        --include-progressive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

from metrics import wilson_ci


METRIC_COLORS = {
    "overall": "#1f77b4",      # blue
    "regressive": "#d62728",   # red
    "progressive": "#2ca02c",  # green
}

MODEL_STYLES = {
    "teacher": "-",
    "student": "--",
}

MARKERS = {
    "teacher": "o",
    "student": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot teacher vs student sycophancy rates over pressure strengths."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to distillation JSON containing the 'records' field.",
    )
    parser.add_argument(
        "--output",
        default=Path("sycophancy_multiline.png"),
        type=Path,
        help="Destination image path (extension controls format).",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="Filter to one or more chain modes (repeat flag to select multiple).",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        help="Filter to one or more capability buckets.",
    )
    parser.add_argument(
        "--facet-mode",
        action="store_true",
        help="Create a subplot per mode instead of aggregating all modes.",
    )
    parser.add_argument(
        "--include-progressive",
        action="store_true",
        help="Add progressive sycophancy lines alongside overall/regressive.",
    )
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Disable Wilson confidence intervals shading.",
    )
    parser.add_argument(
        "--facet-metric",
        action="store_true",
        help="Create a separate subplot for each sycophancy metric to reduce overlap.",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=10.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=5.0,
        help="Figure height in inches (per subplot row).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving.",
    )
    return parser.parse_args()


def load_records(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    records = raw.get("records")
    if records is None:
        raise ValueError("JSON file missing 'records' field.")

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No records available to plot.")

    required_cols = {
        "model",
        "mode",
        "strength",
        "bucket",
        "first_label",
        "after_label",
        "sycophancy",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in records: {missing}")

    return df


def infer_strength_order(values: Iterable) -> List:
    clean = [v for v in values if pd.notna(v)]
    if not clean:
        return []

    def _to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    numeric = [_to_float(v) for v in clean]
    if all(v is not None for v in numeric):
        pairs = sorted(set(zip(clean, numeric)), key=lambda x: x[1])
        return [p[0] for p in pairs]

    canonical = ["low", "medium", "high", "extreme"]
    ordered = []
    for name in canonical:
        if name in clean and name not in ordered:
            ordered.append(name)

    for val in clean:
        if val not in ordered:
            ordered.append(val)
    return ordered


def add_behavior_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_sycophantic"] = df["sycophancy"] != "none"
    df["is_regressive"] = df["sycophancy"] == "regressive"
    df["is_progressive"] = df["sycophancy"] == "progressive"
    return df


def aggregate_rates(
    df: pd.DataFrame,
    strength_levels: List,
    facet_mode: bool,
    include_progressive: bool,
    include_ci: bool,
) -> pd.DataFrame:
    metric_map = [
        ("overall", "is_sycophantic"),
        ("regressive", "is_regressive"),
    ]
    if include_progressive:
        metric_map.append(("progressive", "is_progressive"))

    group_cols = ["model", "strength"]
    if facet_mode:
        group_cols.append("mode")

    rows = []
    for keys, group in df.groupby(group_cols):
        key_dict = dict(zip(group_cols, keys))
        total = len(group)
        for metric_label, column in metric_map:
            successes = int(group[column].sum())
            rate = successes / total if total else float("nan")
            ci_low, ci_high = (None, None)
            if include_ci and total:
                ci_low, ci_high = wilson_ci(successes, total)

            rows.append(
                {
                    "model": key_dict["model"],
                    "mode": key_dict.get("mode", "all"),
                    "strength": key_dict["strength"],
                    "metric": metric_label,
                    "rate": rate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "count": total,
                }
            )

    agg = pd.DataFrame(rows)
    if agg.empty:
        raise ValueError("No aggregated rows available. Check filters.")

    agg["strength"] = pd.Categorical(agg["strength"], categories=strength_levels, ordered=True)
    return agg.sort_values(["mode", "model", "metric", "strength"])


def resolve_strength_positions(strength_levels: List) -> dict:
    if not strength_levels:
        return {}
    return {level: idx for idx, level in enumerate(strength_levels)}


def plot_lines(
    data: pd.DataFrame,
    strength_levels: List,
    args: argparse.Namespace,
) -> None:
    mode_levels = list(data["mode"].unique()) if args.facet_mode else ["all"]
    metric_levels = list(data["metric"].unique()) if args.facet_metric else ["all"]
    mode_levels = mode_levels or ["all"]
    metric_levels = metric_levels or ["all"]

    n_subplots = len(mode_levels) * len(metric_levels)
    figsize = (args.width, args.height * n_subplots)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        n_subplots,
        1,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    if n_subplots == 1:
        axes = [axes]
    else:
        axes = axes if isinstance(axes, (list, tuple)) else list(axes)

    x_positions = resolve_strength_positions(strength_levels)
    xticks = list(range(len(strength_levels)))

    axis_idx = 0
    legend_handles = []
    legend_labels = []

    for metric_key in metric_levels:
        for mode_key in mode_levels:
            ax = axes[axis_idx]
            axis_idx += 1

            subset = data.copy()
            if mode_key != "all":
                subset = subset[subset["mode"] == mode_key]
            if args.facet_metric:
                subset = subset[subset["metric"] == metric_key]

            if subset.empty:
                ax.set_visible(False)
                continue

            for (model, metric), line_df in subset.groupby(["model", "metric"]):
                color = METRIC_COLORS.get(metric, "#555555")
                linestyle = MODEL_STYLES.get(model, "-")
                marker = MARKERS.get(model, "o")
                label = model if args.facet_metric else f"{model} - {metric}"
                x_vals = [x_positions.get(strength) for strength in line_df["strength"]]
                y_vals = line_df["rate"].tolist()
                line, = ax.plot(
                    x_vals,
                    y_vals,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2,
                )

                if not args.no_ci:
                    for x, (_, row) in zip(x_vals, line_df.iterrows()):
                        if row["ci_low"] is None or row["ci_high"] is None:
                            continue
                        ax.vlines(
                            x,
                            row["ci_low"],
                            row["ci_high"],
                            color=color,
                            alpha=0.4,
                            linewidth=2,
                        )

                if not legend_labels or label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)

            ax.set_xticks(xticks)
            ax.set_xticklabels(str(strength) for strength in strength_levels)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Rate")

            title_parts = []
            if args.facet_metric:
                title_parts.append(metric_key.capitalize())
            if args.facet_mode:
                title_parts.append(mode_key.capitalize())
            if not title_parts:
                title_parts.append("All conditions")
            ax.set_title(" • ".join(title_parts))
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Pressure strength")
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved plot to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_records(args.input)

    if args.mode:
        df = df[df["mode"].isin(args.mode)]
    if args.bucket:
        df = df[df["bucket"].isin(args.bucket)]

    if df.empty:
        raise ValueError("No data left after applying filters.")

    df = add_behavior_flags(df)

    strength_levels = infer_strength_order(df["strength"])
    if not strength_levels:
        raise ValueError("Could not infer strength ordering from data.")

    df["strength"] = pd.Categorical(df["strength"], categories=strength_levels, ordered=True)

    agg = aggregate_rates(
        df,
        strength_levels=strength_levels,
        facet_mode=args.facet_mode,
        include_progressive=args.include_progressive,
        include_ci=not args.no_ci,
    )

    plot_lines(
        data=agg,
        strength_levels=strength_levels,
        args=args,
    )


if __name__ == "__main__":
    main()

