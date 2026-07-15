from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_LAGS = (
    1,
    2,
    3,
    4,
    8,
    13,
    26,
    52,
)

DEFAULT_ROLLING_WINDOWS = (
    4,
    8,
    13,
    26,
    52,
)

DEFAULT_COVARIATES = (
    "temp_mean",
    "temp_min",
    "temp_max",
    "heating_degree_days",
    "cooling_degree_days",
    "holiday_days",
    "has_holiday",
)


def add_calendar_features(
    data: pd.DataFrame,
    start_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Add seasonal calendar and trend features.

    Parameters
    ----------
    data:
        DataFrame with a DatetimeIndex.

    start_timestamp:
        Timestamp used as the beginning of the trend feature.
        When omitted, the first timestamp in data is used.
    """

    features = data.copy()

    if not isinstance(
        features.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Data must use a DatetimeIndex."
        )

    if features.empty:
        raise ValueError(
            "Data must not be empty."
        )

    if start_timestamp is None:
        start_timestamp = features.index.min()

    start_timestamp = pd.Timestamp(
        start_timestamp
    )

    iso_week = (
        features.index
        .isocalendar()
        .week
        .astype(int)
    )

    month = features.index.month

    features["week_sin"] = np.sin(
        2
        * np.pi
        * iso_week
        / 52
    )

    features["week_cos"] = np.cos(
        2
        * np.pi
        * iso_week
        / 52
    )

    features["month_sin"] = np.sin(
        2
        * np.pi
        * month
        / 12
    )

    features["month_cos"] = np.cos(
        2
        * np.pi
        * month
        / 12
    )

    features["trend"] = (
        features.index
        - start_timestamp
    ).days / 7.0

    return features


def make_ml_table(
    data: pd.DataFrame,
    target: str = "load_gw",
    lag_weeks: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    covariate_columns: Sequence[str] = DEFAULT_COVARIATES,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """
    Create a supervised weekly machine-learning table.

    All target lag and rolling-window features are based only on
    observations strictly before the prediction timestamp.

    Rolling features use target.shift(1), ensuring that the current
    target is never included.
    """

    if target not in data.columns:
        raise ValueError(
            f"Target column '{target}' was not found."
        )

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Data must use a DatetimeIndex."
        )

    available_covariates = [
        column
        for column in covariate_columns
        if column in data.columns
    ]

    feature_data = data[
        [target, *available_covariates]
    ].copy()

    target_series = feature_data[
        target
    ].astype(float)

    for lag in lag_weeks:
        if lag <= 0:
            raise ValueError(
                "All lag values must be positive."
            )

        feature_data[
            f"lag_{lag}"
        ] = target_series.shift(
            lag
        )

    past_target = target_series.shift(1)

    for window in rolling_windows:
        if window <= 1:
            raise ValueError(
                "Rolling windows must be greater than one."
            )

        feature_data[
            f"rolling_mean_{window}"
        ] = (
            past_target
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
        )

        feature_data[
            f"rolling_std_{window}"
        ] = (
            past_target
            .rolling(
                window=window,
                min_periods=window,
            )
            .std()
        )

    feature_data = add_calendar_features(
        feature_data,
        start_timestamp=data.index.min(),
    )

    if drop_missing:
        feature_data = (
            feature_data
            .dropna()
            .copy()
        )

    return feature_data


def make_single_feature_row(
    timestamp: pd.Timestamp,
    load_history: pd.Series,
    covariate_row: pd.Series,
    feature_columns: Sequence[str],
    start_timestamp: pd.Timestamp,
    lag_weeks: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    covariate_columns: Sequence[str] = DEFAULT_COVARIATES,
) -> pd.DataFrame:
    """
    Construct one feature row for recursive forecasting.

    Parameters
    ----------
    timestamp:
        Timestamp being forecast.

    load_history:
        Historical actual and recursively predicted target values
        available before timestamp.

    covariate_row:
        Known or conditionally observed covariates for timestamp.

    feature_columns:
        Feature columns used when fitting the model.
    """

    history = pd.Series(
        load_history,
        copy=True,
    ).astype(float)

    if history.empty:
        raise ValueError(
            "Load history must not be empty."
        )

    maximum_required_history = max(
        max(lag_weeks),
        max(rolling_windows),
    )

    if len(history) < maximum_required_history:
        raise ValueError(
            "Insufficient load history to construct "
            f"features. At least "
            f"{maximum_required_history} observations "
            "are required."
        )

    feature_values: dict[str, float] = {}

    for column in covariate_columns:
        if column in feature_columns:
            if column not in covariate_row.index:
                raise ValueError(
                    f"Covariate '{column}' is missing "
                    f"for timestamp {timestamp}."
                )

            feature_values[column] = float(
                covariate_row[column]
            )

    for lag in lag_weeks:
        feature_values[
            f"lag_{lag}"
        ] = float(
            history.iloc[-lag]
        )

    for window in rolling_windows:
        past_window = history.iloc[
            -window:
        ]

        feature_values[
            f"rolling_mean_{window}"
        ] = float(
            past_window.mean()
        )

        feature_values[
            f"rolling_std_{window}"
        ] = float(
            past_window.std()
        )

    timestamp = pd.Timestamp(
        timestamp
    )

    week_of_year = int(
        timestamp.isocalendar().week
    )

    month = timestamp.month

    feature_values["week_sin"] = float(
        np.sin(
            2
            * np.pi
            * week_of_year
            / 52
        )
    )

    feature_values["week_cos"] = float(
        np.cos(
            2
            * np.pi
            * week_of_year
            / 52
        )
    )

    feature_values["month_sin"] = float(
        np.sin(
            2
            * np.pi
            * month
            / 12
        )
    )

    feature_values["month_cos"] = float(
        np.cos(
            2
            * np.pi
            * month
            / 12
        )
    )

    feature_values["trend"] = float(
        (
            timestamp
            - pd.Timestamp(
                start_timestamp
            )
        ).days
        / 7.0
    )

    feature_row = pd.DataFrame(
        [feature_values],
        index=pd.DatetimeIndex(
            [timestamp]
        ),
    )

    missing_features = [
        column
        for column in feature_columns
        if column
        not in feature_row.columns
    ]

    if missing_features:
        raise ValueError(
            "Recursive feature row is missing "
            f"columns: {missing_features}"
        )

    feature_row = feature_row[
        list(feature_columns)
    ]

    if feature_row.isna().any().any():
        raise ValueError(
            "Recursive feature row contains "
            "missing values."
        )

    return feature_row