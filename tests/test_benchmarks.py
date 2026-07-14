import numpy as np
import pandas as pd
import pytest

from electricity_demand.models.benchmarks import (
    drift_forecast,
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)


def make_training_series() -> pd.Series:
    return pd.Series(
        [10.0, 12.0, 14.0, 16.0],
        index=pd.date_range(
            "2020-01-01",
            periods=4,
            freq="7D",
        ),
    )


def test_mean_forecast_length_and_values():
    train = make_training_series()

    forecast = mean_forecast(
        train,
        horizon=3,
    )

    assert len(forecast) == 3

    assert np.allclose(
        forecast.to_numpy(),
        train.mean(),
    )


def test_naive_forecast_repeats_last_value():
    train = make_training_series()

    forecast = naive_forecast(
        train,
        horizon=3,
    )

    assert np.allclose(
        forecast.to_numpy(),
        [16.0, 16.0, 16.0],
    )


def test_seasonal_naive_repeats_latest_cycle():
    train = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
    )

    forecast = seasonal_naive_forecast(
        train,
        horizon=6,
        seasonality=2,
    )

    assert np.allclose(
        forecast.to_numpy(),
        [3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
    )


def test_drift_forecast_extends_linear_trend():
    train = make_training_series()

    forecast = drift_forecast(
        train,
        horizon=2,
    )

    assert np.allclose(
        forecast.to_numpy(),
        [18.0, 20.0],
    )


def test_seasonal_naive_rejects_short_training_data():
    train = pd.Series(
        [1.0, 2.0, 3.0],
    )

    with pytest.raises(ValueError):
        seasonal_naive_forecast(
            train,
            horizon=2,
            seasonality=4,
        )