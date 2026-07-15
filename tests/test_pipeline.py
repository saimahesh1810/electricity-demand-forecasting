import pandas as pd

from electricity_demand.pipeline import (
    CORE_METRICS,
)


def test_core_metrics_are_defined():
    assert CORE_METRICS == [
        "MAE",
        "RMSE",
        "MASE",
        "Bias",
        "sMAPE",
    ]


def test_metric_sorting_by_mae():
    metrics = pd.DataFrame(
        {
            "model": [
                "model_a",
                "model_b",
                "model_c",
            ],
            "MAE": [
                3.0,
                1.0,
                2.0,
            ],
            "RMSE": [
                4.0,
                2.0,
                3.0,
            ],
        }
    )

    sorted_metrics = (
        metrics
        .sort_values(
            [
                "MAE",
                "RMSE",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    assert sorted_metrics[
        "model"
    ].tolist() == [
        "model_b",
        "model_c",
        "model_a",
    ]