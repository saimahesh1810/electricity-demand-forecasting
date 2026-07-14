import numpy as np
import pandas as pd

from electricity_demand.evaluation import (
    evaluate_forecast,
    forecast_bias,
    mean_absolute_error,
    mean_absolute_scaled_error,
    root_mean_squared_error,
)


def test_mae_is_zero_for_perfect_forecast():
    actual = pd.Series(
        [1.0, 2.0, 3.0]
    )

    predicted = actual.copy()

    assert mean_absolute_error(
        actual,
        predicted,
    ) == 0.0


def test_rmse_is_zero_for_perfect_forecast():
    actual = pd.Series(
        [1.0, 2.0, 3.0]
    )

    predicted = actual.copy()

    assert root_mean_squared_error(
        actual,
        predicted,
    ) == 0.0


def test_mase_is_zero_for_perfect_forecast():
    train = pd.Series(
        np.arange(
            1,
            20,
            dtype=float,
        )
    )

    actual = pd.Series(
        [20.0, 21.0, 22.0]
    )

    predicted = actual.copy()

    mase = mean_absolute_scaled_error(
        y_true=actual,
        y_pred=predicted,
        y_train=train,
        seasonality=4,
    )

    assert mase == 0.0


def test_positive_bias_means_overforecasting():
    actual = pd.Series(
        [10.0, 10.0]
    )

    predicted = pd.Series(
        [12.0, 14.0]
    )

    assert forecast_bias(
        actual,
        predicted,
    ) == 3.0


def test_evaluate_forecast_returns_required_metrics():
    train = pd.Series(
        np.arange(
            1,
            120,
            dtype=float,
        )
    )

    actual = pd.Series(
        [120.0, 121.0],
        index=pd.date_range(
            "2022-01-01",
            periods=2,
            freq="7D",
        ),
    )

    predicted = pd.Series(
        [119.0, 122.0],
        index=actual.index,
    )

    result = evaluate_forecast(
        name="test_model",
        y_true=actual,
        y_pred=predicted,
        y_train=train,
        seasonality=52,
    )

    assert result["model"] == "test_model"

    assert {
        "MAE",
        "RMSE",
        "MASE",
        "Bias",
        "sMAPE",
    }.issubset(result.keys())