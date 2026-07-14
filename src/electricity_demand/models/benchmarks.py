from __future__ import annotations

import numpy as np
import pandas as pd


def validate_training_series(
    y_train: pd.Series,
) -> pd.Series:
    """
    Validate and return a numeric training series.
    """

    train = pd.Series(
        y_train,
        copy=True,
    ).astype(float)

    if train.empty:
        raise ValueError(
            "Training series must not be empty."
        )

    if train.isna().any():
        raise ValueError(
            "Training series contains missing values."
        )

    return train


def validate_horizon(
    horizon: int,
) -> None:
    """
    Validate forecast horizon.
    """

    if not isinstance(horizon, int):
        raise TypeError(
            "Forecast horizon must be an integer."
        )

    if horizon <= 0:
        raise ValueError(
            "Forecast horizon must be greater than zero."
        )


def create_forecast_index(
    horizon: int,
    index: pd.Index | None,
) -> pd.Index:
    """
    Create or validate the forecast index.
    """

    if index is None:
        return pd.RangeIndex(horizon)

    if len(index) != horizon:
        raise ValueError(
            "Forecast index length must match the horizon."
        )

    return index


def mean_forecast(
    y_train: pd.Series,
    horizon: int,
    index: pd.Index | None = None,
) -> pd.Series:
    """
    Forecast the historical training mean for every future period.
    """

    train = validate_training_series(
        y_train
    )

    validate_horizon(horizon)

    forecast_index = create_forecast_index(
        horizon=horizon,
        index=index,
    )

    values = np.repeat(
        train.mean(),
        horizon,
    )

    return pd.Series(
        values,
        index=forecast_index,
        name="mean",
    )


def naive_forecast(
    y_train: pd.Series,
    horizon: int,
    index: pd.Index | None = None,
) -> pd.Series:
    """
    Forecast the final training observation for every future period.
    """

    train = validate_training_series(
        y_train
    )

    validate_horizon(horizon)

    forecast_index = create_forecast_index(
        horizon=horizon,
        index=index,
    )

    values = np.repeat(
        train.iloc[-1],
        horizon,
    )

    return pd.Series(
        values,
        index=forecast_index,
        name="naive",
    )


def seasonal_naive_forecast(
    y_train: pd.Series,
    horizon: int,
    seasonality: int = 52,
    index: pd.Index | None = None,
) -> pd.Series:
    """
    Repeat the most recent complete seasonal cycle.
    """

    train = validate_training_series(
        y_train
    )

    validate_horizon(horizon)

    if seasonality <= 0:
        raise ValueError(
            "Seasonality must be greater than zero."
        )

    if len(train) < seasonality:
        raise ValueError(
            "Training data must contain at least one "
            "complete seasonal cycle."
        )

    forecast_index = create_forecast_index(
        horizon=horizon,
        index=index,
    )

    latest_season = train.iloc[
        -seasonality:
    ].to_numpy()

    repetitions = int(
        np.ceil(
            horizon / seasonality
        )
    )

    values = np.tile(
        latest_season,
        repetitions,
    )[:horizon]

    return pd.Series(
        values,
        index=forecast_index,
        name="seasonal_naive",
    )


def drift_forecast(
    y_train: pd.Series,
    horizon: int,
    index: pd.Index | None = None,
) -> pd.Series:
    """
    Forecast using a straight line between the first and last
    training observations.
    """

    train = validate_training_series(
        y_train
    )

    validate_horizon(horizon)

    if len(train) < 2:
        raise ValueError(
            "Drift forecast requires at least two "
            "training observations."
        )

    forecast_index = create_forecast_index(
        horizon=horizon,
        index=index,
    )

    slope = (
        train.iloc[-1]
        - train.iloc[0]
    ) / (
        len(train)
        - 1
    )

    steps = np.arange(
        1,
        horizon + 1,
    )

    values = (
        train.iloc[-1]
        + slope * steps
    )

    return pd.Series(
        values,
        index=forecast_index,
        name="drift",
    )