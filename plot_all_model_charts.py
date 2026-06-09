#!/usr/bin/env python
"""Generate a large chart gallery for every model run.

This script uses the existing CSV outputs from run_crop_yield_study.py, so it
does not retrain any model. It creates aggregate model comparisons, heatmaps,
per-scenario metric charts, predicted-vs-actual grids, residual distributions,
and feature-importance summaries.
"""

from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET = "yield_t_ha"
METRICS = ["MAE", "RMSE", "R2"]
LOWER_IS_BETTER = {"MAE", "RMSE"}
SPLIT_SHORT = {
    "random_80_20": "random",
    "time_train_le_2015_test_gt_2015": "time",
    "leave_country_out_train_Australia_test_United States": "aus_to_us",
    "leave_country_out_train_United States_test_Australia": "us_to_aus",
}


def slugify(value: object, max_len: int = 80) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text[:max_len] or "value"


def short_split(split: str) -> str:
    return SPLIT_SHORT.get(split, slugify(split, 36))


def pretty_label(value: object) -> str:
    return str(value).replace("_", " ")


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    dirs = {
        "base": base_dir,
        "global": base_dir / "global",
        "heatmaps": base_dir / "heatmaps",
        "scenario_metrics": base_dir / "scenario_metrics",
        "scenario_predictions": base_dir / "scenario_predictions",
        "feature_importance": base_dir / "feature_importance",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig: plt.Figure, path: Path, manifest: list[dict[str, str]], kind: str, title: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    manifest.append({"kind": kind, "title": title, "path": str(path)})


def metric_sort(metric: str) -> bool:
    return metric in LOWER_IS_BETTER


def add_value_labels(ax: plt.Axes, values: pd.Series, metric: str) -> None:
    span = float(values.max() - values.min()) if len(values) else 0.0
    pad = span * 0.015 if span > 0 else 0.01
    for i, value in enumerate(values):
        label = f"{value:.3f}"
        if metric == "R2":
            ax.text(value + pad, i, label, va="center", fontsize=6)
        else:
            ax.text(value + pad, i, label, va="center", fontsize=6)


def plot_metric_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    path: Path,
    manifest: list[dict[str, str]],
    kind: str,
) -> None:
    data = df.sort_values(metric, ascending=metric_sort(metric)).reset_index(drop=True)
    height = max(6, 0.28 * len(data) + 1.8)
    fig, ax = plt.subplots(figsize=(10, height))
    color = "#2f6f73" if metric != "R2" else "#5d6db4"
    ax.barh(data["model"], data[metric], color=color, alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_ylabel("Model")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    add_value_labels(ax, data[metric], metric)
    save_figure(fig, path, manifest, kind, title)


def plot_heatmap(
    table: pd.DataFrame,
    title: str,
    path: Path,
    manifest: list[dict[str, str]],
    kind: str,
    cmap: str,
) -> None:
    if table.empty:
        return
    values = table.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    width = max(11, 0.42 * table.shape[1] + 3.5)
    height = max(4.5, 0.42 * table.shape[0] + 2.4)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([pretty_label(c) for c in table.columns], rotation=55, ha="right", fontsize=7)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([pretty_label(i) for i in table.index], fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    if table.size <= 240:
        for row in range(table.shape[0]):
            for col in range(table.shape[1]):
                value = values[row, col]
                if np.isfinite(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="white")
    save_figure(fig, path, manifest, kind, title)


def scenario_id(idx: int, experiment: str, split: str, feature_set: str) -> str:
    return f"s{idx:02d}_{slugify(experiment, 34)}_{short_split(split)}_{slugify(feature_set, 18)}"


def add_ranks(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics_df.copy()
    group_cols = ["experiment", "dataset", "feature_set", "split"]
    ranked["RMSE_rank"] = ranked.groupby(group_cols)["RMSE"].rank(method="min", ascending=True)
    ranked["MAE_rank"] = ranked.groupby(group_cols)["MAE"].rank(method="min", ascending=True)
    ranked["R2_rank"] = ranked.groupby(group_cols)["R2"].rank(method="min", ascending=False)
    ranked["mean_rank"] = ranked[["RMSE_rank", "MAE_rank", "R2_rank"]].mean(axis=1)
    return ranked


def make_global_charts(metrics_df: pd.DataFrame, dirs: dict[str, Path], manifest: list[dict[str, str]]) -> None:
    ranked = add_ranks(metrics_df)
    order = (
        metrics_df.groupby("model")["RMSE"].mean().sort_values().index.tolist()
    )

    mean_metrics = metrics_df.groupby("model", as_index=False)[METRICS].mean()
    for metric in METRICS:
        plot_metric_bar(
            mean_metrics,
            metric,
            f"Mean {metric} Across All Runs",
            dirs["global"] / f"mean_{metric.lower()}_by_model.png",
            manifest,
            "global",
        )

    median_metrics = metrics_df.groupby("model", as_index=False)[METRICS].median()
    for metric in METRICS:
        plot_metric_bar(
            median_metrics,
            metric,
            f"Median {metric} Across All Runs",
            dirs["global"] / f"median_{metric.lower()}_by_model.png",
            manifest,
            "global",
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, max(8, len(order) * 0.32)))
    for ax, metric in zip(axes, METRICS):
        values = [metrics_df.loc[metrics_df["model"] == model, metric].to_numpy() for model in order]
        ax.boxplot(values, tick_labels=[pretty_label(m) for m in order], vert=False, showfliers=False)
        ax.set_title(f"{metric} Distribution by Model")
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.25)
    save_figure(fig, dirs["global"] / "metric_boxplots_by_model.png", manifest, "global", "Metric Boxplots by Model")

    best_rmse = (
        ranked.sort_values(["experiment", "dataset", "feature_set", "split", "RMSE"])
        .groupby(["experiment", "dataset", "feature_set", "split"], as_index=False)
        .head(1)
    )
    wins = best_rmse["model"].value_counts().reindex(order).fillna(0).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(wins) + 1.5)))
    ax.barh(wins.index, wins.values, color="#7b6d43")
    ax.set_xlabel("Best-RMSE Wins")
    ax.set_title("How Often Each Model Wins a Scenario")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, dirs["global"] / "best_rmse_win_counts_by_model.png", manifest, "global", "Best RMSE Win Counts by Model")

    mean_rank = ranked.groupby("model", as_index=False)[["RMSE_rank", "MAE_rank", "R2_rank", "mean_rank"]].mean()
    plot_metric_bar(
        mean_rank.rename(columns={"mean_rank": "MeanRank"}),
        "MeanRank",
        "Mean Rank Across MAE, RMSE, and R2",
        dirs["global"] / "mean_rank_by_model.png",
        manifest,
        "global",
    )

    table = ranked.pivot_table(index="experiment", columns="model", values="mean_rank", aggfunc="mean")
    table = table.reindex(columns=order)
    plot_heatmap(table, "Mean Rank by Experiment and Model", dirs["global"] / "mean_rank_heatmap_experiment_model.png", manifest, "global", "viridis_r")

    for metric in METRICS:
        table = metrics_df.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
        table = table.reindex(columns=order)
        cmap = "magma_r" if metric in LOWER_IS_BETTER else "YlGn"
        plot_heatmap(table, f"Mean {metric} by Dataset and Model", dirs["global"] / f"mean_{metric.lower()}_heatmap_dataset_model.png", manifest, "global", cmap)

    fig, ax = plt.subplots(figsize=(10, 7))
    for split, grp in metrics_df.groupby("split"):
        ax.scatter(grp["RMSE"], grp["R2"], s=22, alpha=0.55, label=pretty_label(short_split(split)))
    ax.set_xlabel("RMSE")
    ax.set_ylabel("R2")
    ax.set_title("All Runs: RMSE vs R2")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    save_figure(fig, dirs["global"] / "rmse_vs_r2_all_runs.png", manifest, "global", "All Runs RMSE vs R2")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(metrics_df["train_rows"], metrics_df["RMSE"], s=24, alpha=0.55, c=metrics_df["R2"], cmap="viridis")
    ax.set_xlabel("Train Rows")
    ax.set_ylabel("RMSE")
    ax.set_title("Training Size vs RMSE")
    ax.grid(alpha=0.25)
    save_figure(fig, dirs["global"] / "train_rows_vs_rmse.png", manifest, "global", "Training Size vs RMSE")

    paired_feature = metrics_df.pivot_table(
        index=["experiment", "split", "model"],
        columns="feature_set",
        values=["MAE", "RMSE", "R2"],
        aggfunc="mean",
    )
    if {"weather_only", "weather_soil"}.issubset(set(metrics_df["feature_set"].unique())):
        deltas = []
        for metric in METRICS:
            metric_table = paired_feature[metric].dropna()
            if {"weather_only", "weather_soil"}.issubset(metric_table.columns):
                delta = metric_table["weather_soil"] - metric_table["weather_only"]
                deltas.append(delta.rename(metric))
        if deltas:
            delta_df = pd.concat(deltas, axis=1).reset_index()
            agg = delta_df.groupby("model", as_index=False)[METRICS].median()
            for metric in METRICS:
                plot_metric_bar(
                    agg.rename(columns={metric: f"{metric}_delta"}),
                    f"{metric}_delta",
                    f"Median {metric} Delta: Weather+Soil minus Weather Only",
                    dirs["global"] / f"weather_soil_minus_weather_only_{metric.lower()}_by_model.png",
                    manifest,
                    "global",
                )

    split_table = metrics_df.pivot_table(
        index=["experiment", "feature_set", "model"],
        columns="split",
        values="RMSE",
        aggfunc="mean",
    )
    if {"random_80_20", "time_train_le_2015_test_gt_2015"}.issubset(split_table.columns):
        delta = (split_table["time_train_le_2015_test_gt_2015"] - split_table["random_80_20"]).dropna()
        agg = delta.groupby("model").median().reset_index(name="RMSE_delta")
        plot_metric_bar(
            agg,
            "RMSE_delta",
            "Median RMSE Delta: Time Split minus Random Split",
            dirs["global"] / "time_minus_random_rmse_delta_by_model.png",
            manifest,
            "global",
        )


def make_heatmap_charts(metrics_df: pd.DataFrame, dirs: dict[str, Path], manifest: list[dict[str, str]]) -> None:
    order = metrics_df.groupby("model")["RMSE"].mean().sort_values().index.tolist()
    for metric in METRICS:
        for split in sorted(metrics_df["split"].unique(), key=short_split):
            for feature_set in sorted(metrics_df["feature_set"].unique()):
                mask = (metrics_df["split"] == split) & (metrics_df["feature_set"] == feature_set)
                source = metrics_df.loc[mask]
                if source.empty:
                    continue
                table = source.pivot_table(index="experiment", columns="model", values=metric, aggfunc="mean")
                table = table.reindex(columns=order)
                cmap = "magma_r" if metric in LOWER_IS_BETTER else "YlGn"
                title = f"{metric} Heatmap: {pretty_label(short_split(split))}, {pretty_label(feature_set)}"
                filename = f"heatmap_{metric.lower()}_{short_split(split)}_{slugify(feature_set, 24)}.png"
                plot_heatmap(table, title, dirs["heatmaps"] / filename, manifest, "heatmap", cmap)


def make_scenario_metric_charts(metrics_df: pd.DataFrame, dirs: dict[str, Path], manifest: list[dict[str, str]]) -> pd.DataFrame:
    scenario_cols = ["experiment", "dataset", "split", "feature_set"]
    scenarios = (
        metrics_df[scenario_cols]
        .drop_duplicates()
        .sort_values(scenario_cols)
        .reset_index(drop=True)
    )
    scenario_rows = []
    for idx, scenario in scenarios.iterrows():
        exp = scenario["experiment"]
        dataset = scenario["dataset"]
        split = scenario["split"]
        feature_set = scenario["feature_set"]
        sid = scenario_id(idx + 1, exp, split, feature_set)
        mask = (
            (metrics_df["experiment"] == exp)
            & (metrics_df["dataset"] == dataset)
            & (metrics_df["split"] == split)
            & (metrics_df["feature_set"] == feature_set)
        )
        data = metrics_df.loc[mask].copy()
        title_base = f"{exp} | {pretty_label(short_split(split))} | {pretty_label(feature_set)}"
        for metric in METRICS:
            plot_metric_bar(
                data,
                metric,
                f"{metric} by Model: {title_base}",
                dirs["scenario_metrics"] / f"{sid}_{metric.lower()}_by_model.png",
                manifest,
                "scenario_metric",
            )
        scenario_rows.append(
            {
                "scenario_id": sid,
                "experiment": exp,
                "dataset": dataset,
                "split": split,
                "feature_set": feature_set,
            }
        )
    return pd.DataFrame(scenario_rows)


def make_prediction_charts(
    errors_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    dirs: dict[str, Path],
    manifest: list[dict[str, str]],
    max_points_per_model: int,
) -> None:
    if errors_df.empty:
        return
    metric_lookup = metrics_df.set_index(["experiment", "dataset", "feature_set", "split", "model"])
    for _, scenario in scenarios_df.iterrows():
        sid = scenario["scenario_id"]
        exp = scenario["experiment"]
        dataset = scenario["dataset"]
        split = scenario["split"]
        feature_set = scenario["feature_set"]
        mask = (
            (errors_df["experiment"] == exp)
            & (errors_df["dataset"] == dataset)
            & (errors_df["split"] == split)
            & (errors_df["feature_set"] == feature_set)
        )
        data = errors_df.loc[mask].copy()
        if data.empty:
            continue
        scenario_metrics = metrics_df[
            (metrics_df["experiment"] == exp)
            & (metrics_df["dataset"] == dataset)
            & (metrics_df["split"] == split)
            & (metrics_df["feature_set"] == feature_set)
        ].sort_values("RMSE")
        model_order = scenario_metrics["model"].tolist()

        ncols = 4
        nrows = int(math.ceil(len(model_order) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(9, nrows * 2.6)), sharex=True, sharey=True)
        axes_arr = np.asarray(axes).reshape(-1)
        low = min(data[TARGET].min(), data["prediction"].min())
        high = max(data[TARGET].max(), data["prediction"].max())
        for ax, model in zip(axes_arr, model_order):
            grp = data[data["model"] == model]
            if len(grp) > max_points_per_model:
                grp = grp.sample(max_points_per_model, random_state=42)
            ax.scatter(grp[TARGET], grp["prediction"], s=8, alpha=0.35, color="#2f6f73")
            ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.8)
            rmse = metric_lookup.loc[(exp, dataset, feature_set, split, model), "RMSE"]
            ax.set_title(f"{pretty_label(model)}\nRMSE={rmse:.3f}", fontsize=8)
            ax.grid(alpha=0.2)
        for ax in axes_arr[len(model_order) :]:
            ax.axis("off")
        fig.supxlabel("Actual yield t/ha")
        fig.supylabel("Predicted yield t/ha")
        fig.suptitle(f"Predicted vs Actual: {exp} | {pretty_label(short_split(split))} | {pretty_label(feature_set)}", fontsize=12)
        save_figure(fig, dirs["scenario_predictions"] / f"{sid}_predicted_vs_actual_all_models.png", manifest, "scenario_prediction", "Predicted vs Actual All Models")

        residual_values = [data.loc[data["model"] == model, "error"].dropna().to_numpy() for model in model_order]
        fig, ax = plt.subplots(figsize=(10, max(7, 0.32 * len(model_order) + 2)))
        ax.boxplot(residual_values, tick_labels=[pretty_label(m) for m in model_order], vert=False, showfliers=False)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Model")
        ax.set_title(f"Residual Distribution: {exp} | {pretty_label(short_split(split))} | {pretty_label(feature_set)}")
        ax.grid(axis="x", alpha=0.25)
        save_figure(fig, dirs["scenario_predictions"] / f"{sid}_residual_boxplot_all_models.png", manifest, "scenario_prediction", "Residual Boxplot All Models")

        abs_error_summary = (
            data.groupby(["model", "year_start"], as_index=False)["abs_error"]
            .mean()
            .merge(scenario_metrics[["model", "RMSE"]], on="model", how="left")
        )
        top_models = scenario_metrics.head(8)["model"].tolist()
        fig, ax = plt.subplots(figsize=(11, 6))
        for model in top_models:
            grp = abs_error_summary[abs_error_summary["model"] == model].sort_values("year_start")
            ax.plot(grp["year_start"], grp["abs_error"], linewidth=1.2, alpha=0.8, label=pretty_label(model))
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title(f"Yearly Error for Top 8 Models: {exp} | {pretty_label(short_split(split))} | {pretty_label(feature_set)}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        save_figure(fig, dirs["scenario_predictions"] / f"{sid}_top8_yearly_abs_error.png", manifest, "scenario_prediction", "Top 8 Yearly Absolute Error")


def make_feature_importance_charts(
    importance_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    dirs: dict[str, Path],
    manifest: list[dict[str, str]],
) -> None:
    if importance_df.empty:
        return

    top_features = (
        importance_df.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(30)
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_features["feature"][::-1], top_features["importance"][::-1], color="#2f6f73")
    ax.set_xlabel("Mean Importance")
    ax.set_title("Top 30 Features Across All Models with Importances")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, dirs["feature_importance"] / "top30_features_overall.png", manifest, "feature_importance", "Top 30 Features Overall")

    freq = (
        importance_df[importance_df["rank"] <= 10]
        .groupby("feature")
        .size()
        .sort_values(ascending=True)
        .tail(30)
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(freq.index, freq.values, color="#7b6d43")
    ax.set_xlabel("Count in Top 10 Lists")
    ax.set_title("Most Frequent Top-10 Features")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, dirs["feature_importance"] / "most_frequent_top10_features.png", manifest, "feature_importance", "Most Frequent Top-10 Features")

    model_order = metrics_df.groupby("model")["RMSE"].mean().sort_values().index.tolist()
    selected_features = top_features["feature"].head(25).tolist()
    heat = importance_df[importance_df["feature"].isin(selected_features)].pivot_table(
        index="feature",
        columns="model",
        values="importance",
        aggfunc="mean",
    )
    heat = heat.reindex(index=selected_features, columns=[m for m in model_order if m in heat.columns])
    plot_heatmap(
        heat,
        "Mean Importance of Top Features by Model",
        dirs["feature_importance"] / "top_features_by_model_heatmap.png",
        manifest,
        "feature_importance",
        "viridis",
    )

    for model in sorted(importance_df["model"].unique()):
        model_source = importance_df[importance_df["model"] == model]
        if model_source.empty:
            continue
        top = (
            model_source.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
            .head(20)
        )
        if top.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color="#5d6db4")
        ax.set_xlabel("Mean Importance")
        ax.set_title(f"Top 20 Features for {pretty_label(model)}")
        ax.grid(axis="x", alpha=0.25)
        save_figure(
            fig,
            dirs["feature_importance"] / f"top20_features_{slugify(model, 40)}.png",
            manifest,
            "feature_importance",
            f"Top 20 Features for {model}",
        )


def write_summary(
    out_dir: Path,
    manifest: list[dict[str, str]],
    metrics_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
) -> None:
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(out_dir / "chart_manifest.csv", index=False)
    counts = manifest_df["kind"].value_counts().sort_index()
    best = (
        metrics_df.sort_values("RMSE")
        .head(15)[["experiment", "dataset", "feature_set", "split", "model", "MAE", "RMSE", "R2"]]
        .copy()
    )

    lines = [
        "# All Model Chart Gallery",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Charts generated: {len(manifest_df)}",
        f"Models covered: {metrics_df['model'].nunique()}",
        f"Scenarios covered: {len(scenarios_df)}",
        "",
        "## Chart Counts",
        "",
    ]
    for kind, count in counts.items():
        lines.append(f"- {kind}: {count}")
    lines.extend(["", "## Best 15 Runs by RMSE", ""])
    lines.append(best.to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Files", ""])
    lines.append("- Manifest CSV: `chart_manifest.csv`")
    lines.append("- Global charts: `global/`")
    lines.append("- Heatmaps: `heatmaps/`")
    lines.append("- Scenario metric charts: `scenario_metrics/`")
    lines.append("- Prediction and residual charts: `scenario_predictions/`")
    lines.append("- Feature-importance charts: `feature_importance/`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate charts for all model outputs.")
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/paper_yield_study"))
    parser.add_argument("--charts-dir-name", default="all_model_charts")
    parser.add_argument("--max-points-per-model", type=int, default=500)
    parser.add_argument("--skip-prediction-charts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir
    metrics_file = study_dir / "03_model_metrics.csv"
    errors_file = study_dir / "04_error_analysis.csv"
    importance_file = study_dir / "05_feature_importance.csv"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_file}")
    metrics_df = pd.read_csv(metrics_file)
    errors_df = pd.DataFrame()
    if errors_file.exists() and not args.skip_prediction_charts:
        errors_df = pd.read_csv(errors_file)
    importance_df = pd.DataFrame()
    if importance_file.exists():
        importance_df = pd.read_csv(importance_file)

    charts_dir = study_dir / "06_paper_figures" / args.charts_dir_name
    dirs = ensure_dirs(charts_dir)
    manifest: list[dict[str, str]] = []

    make_global_charts(metrics_df, dirs, manifest)
    make_heatmap_charts(metrics_df, dirs, manifest)
    scenarios_df = make_scenario_metric_charts(metrics_df, dirs, manifest)
    if not args.skip_prediction_charts:
        make_prediction_charts(errors_df, metrics_df, scenarios_df, dirs, manifest, args.max_points_per_model)
    make_feature_importance_charts(importance_df, metrics_df, dirs, manifest)
    write_summary(charts_dir, manifest, metrics_df, scenarios_df)

    print(f"Generated {len(manifest)} charts in {charts_dir}")
    print(f"Manifest: {charts_dir / 'chart_manifest.csv'}")
    print(f"Summary: {charts_dir / 'README.md'}")


if __name__ == "__main__":
    main()
