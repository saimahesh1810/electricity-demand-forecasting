from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from electricity_demand.config import (
    FIGURE_DIR,
    METRICS_DIR,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


PIPELINE_SCRIPTS = [
    "scripts/download_data.py",
    "scripts/make_features.py",
    "scripts/evaluate_models.py",
    "scripts/run_sarima.py",
    "scripts/run_sarimax.py",
    "scripts/run_feature_model.py",
    "scripts/run_neural.py",
]


METRIC_FILES = [
    "benchmark_metrics.csv",
    "sarima_metrics.csv",
    "sarimax_metrics.csv",
    "feature_model_metrics.csv",
    "lstm_metrics.csv",
]


CORE_METRICS = [
    "MAE",
    "RMSE",
    "MASE",
    "Bias",
    "sMAPE",
]


def run_project_script(
    relative_script_path: str,
) -> None:
    """
    Run one project script using the active Python interpreter.
    """

    script_path = (
        PROJECT_ROOT
        / relative_script_path
    )

    if not script_path.exists():
        raise FileNotFoundError(
            f"Pipeline script was not found: "
            f"{script_path}"
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"Running: {relative_script_path}"
    )

    print(
        "=" * 72
    )

    subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def load_metric_file(
    metric_path: Path,
) -> pd.DataFrame:
    """
    Load and validate one model-metrics CSV.
    """

    metrics = pd.read_csv(
        metric_path
    )

    if "model" not in metrics.columns:
        raise ValueError(
            f"'model' column is missing from "
            f"{metric_path.name}."
        )

    missing_metrics = [
        metric
        for metric in CORE_METRICS
        if metric not in metrics.columns
    ]

    if missing_metrics:
        raise ValueError(
            f"{metric_path.name} is missing "
            f"metrics: {missing_metrics}"
        )

    return metrics


def build_model_comparison() -> pd.DataFrame:
    """
    Combine saved model metrics into one comparison table.

    Models are ranked using test-period MAE.
    """

    metric_frames: list[pd.DataFrame] = []

    missing_files: list[str] = []

    for filename in METRIC_FILES:
        metric_path = (
            METRICS_DIR
            / filename
        )

        if not metric_path.exists():
            missing_files.append(
                filename
            )

            continue

        metric_frame = load_metric_file(
            metric_path
        )

        metric_frame[
            "source_file"
        ] = filename

        metric_frames.append(
            metric_frame
        )

    if missing_files:
        raise FileNotFoundError(
            "The following metric files are "
            f"missing: {missing_files}"
        )

    comparison = pd.concat(
        metric_frames,
        ignore_index=True,
        sort=False,
    )

    comparison = (
        comparison
        .sort_values(
            by=[
                "MAE",
                "RMSE",
            ],
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    comparison.insert(
        0,
        "rank",
        range(
            1,
            len(comparison) + 1,
        ),
    )

    comparison_output_path = (
        METRICS_DIR
        / "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_output_path,
        index=False,
    )

    return comparison


def plot_model_comparison(
    comparison: pd.DataFrame,
):
    """
    Plot MAE and RMSE for all evaluated models.
    """

    required_columns = [
        "model",
        "MAE",
        "RMSE",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in comparison.columns
    ]

    if missing_columns:
        raise ValueError(
            "Model comparison data is missing "
            f"columns: {missing_columns}"
        )

    plot_data = (
        comparison[
            required_columns
        ]
        .copy()
        .sort_values(
            "MAE",
            ascending=True,
        )
    )

    positions = list(
        range(
            len(plot_data)
        )
    )

    bar_width = 0.38

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.bar(
        [
            position
            - bar_width / 2
            for position in positions
        ],
        plot_data["MAE"],
        width=bar_width,
        label="MAE",
    )

    ax.bar(
        [
            position
            + bar_width / 2
            for position in positions
        ],
        plot_data["RMSE"],
        width=bar_width,
        label="RMSE",
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        plot_data["model"],
        rotation=35,
        ha="right",
    )

    ax.set_title(
        "Forecast Accuracy by Model"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Forecast error (GW)"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    return fig


def save_model_comparison_outputs(
    comparison: pd.DataFrame,
) -> None:
    """
    Save model-comparison figure and print final ranking.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_figure = (
        plot_model_comparison(
            comparison
        )
    )

    figure_output_path = (
        FIGURE_DIR
        / "model_comparison.png"
    )

    comparison_figure.savefig(
        figure_output_path,
        dpi=300,
        bbox_inches="tight",
    )

    display_columns = [
        "rank",
        "model",
        "MAE",
        "RMSE",
        "MASE",
        "Bias",
        "sMAPE",
    ]

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL MODEL COMPARISON"
    )

    print(
        "=" * 72
    )

    print(
        comparison[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nComparison table saved to:"
    )

    print(
        METRICS_DIR
        / "model_comparison.csv"
    )

    print(
        "\nComparison figure saved to:"
    )

    print(
        figure_output_path
    )


def run_full_pipeline(
    include_neural: bool = True,
) -> pd.DataFrame:
    """
    Execute the complete forecasting workflow.
    """

    scripts_to_run = list(
        PIPELINE_SCRIPTS
    )

    if not include_neural:
        scripts_to_run.remove(
            "scripts/run_neural.py"
        )

    for script in scripts_to_run:
        run_project_script(
            script
        )

    comparison = (
        build_model_comparison()
    )

    save_model_comparison_outputs(
        comparison
    )

    return comparison