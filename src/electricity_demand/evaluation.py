from __future__ import annotations

import numpy as np
import pandas as pd


def mean_absolute_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calculate mean absolute error.
    """

    return float(
        np.mean(
            np.abs(
                np.asarray(y_true, dtype=float)
                - np.asarray(y_pred, dtype=float)
            )
        )
    )


def root_mean_squared_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calculate root mean squared error.
    """

    errors = (
        np.asarray(y_true, dtype=float)
        - np.asarray(y_pred, dtype=float)
    )

    return float(
        np.sqrt(
            np.mean(errors**2)
        )
    )


def mean_absolute_scaled_error(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    seasonality: int = 52,
) -> float:
    """
    Calculate seasonal mean absolute scaled error.

    The scaling denominator is based only on the training data.

    A MASE value below 1 means the model performs better than the
    seasonal naive benchmark on average.
    """

    y_train_array = np.asarray(
        y_train,
        dtype=float,
    )

    if len(y_train_array) <= seasonality:
        raise ValueError(
            "Training data must contain more observations "
            "than the seasonal period."
        )

    scale = np.mean(
        np.abs(
            y_train_array[seasonality:]
            - y_train_array[:-seasonality]
        )
    )

    if np.isclose(scale, 0):
        raise ValueError(
            "MASE cannot be calculated because the "
            "training-series scaling denominator is zero."
        )

    mae = mean_absolute_error(
        y_true=y_true,
        y_pred=y_pred,
    )

    return float(mae / scale)


def forecast_bias(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calculate average forecast bias.

    Positive bias means the model overforecasts.
    Negative bias means the model underforecasts.
    """

    return float(
        np.mean(
            np.asarray(y_pred, dtype=float)
            - np.asarray(y_true, dtype=float)
        )
    )


def symmetric_mean_absolute_percentage_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calculate symmetric mean absolute percentage error.
    """

    actual = np.asarray(
        y_true,
        dtype=float,
    )

    predicted = np.asarray(
        y_pred,
        dtype=float,
    )

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    valid = denominator != 0

    if not np.any(valid):
        return 0.0

    smape = np.mean(
        200
        * np.abs(
            predicted[valid]
            - actual[valid]
        )
        / denominator[valid]
    )

    return float(smape)


def align_forecast_inputs(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Align true and predicted values on a common index.
    """

    actual = pd.Series(
        y_true,
        copy=True,
    ).astype(float)

    predicted = pd.Series(
        y_pred,
        copy=True,
    ).astype(float)

    predicted = predicted.reindex(
        actual.index
    )

    if predicted.isna().any():
        missing_dates = predicted[
            predicted.isna()
        ].index.tolist()

        raise ValueError(
            "Forecast contains missing values after alignment. "
            f"Missing timestamps include: {missing_dates[:5]}"
        )

    return actual, predicted


def evaluate_forecast(
    name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    seasonality: int = 52,
) -> dict[str, float | str]:
    """
    Evaluate a forecast using the common project metrics.
    """

    actual, predicted = align_forecast_inputs(
        y_true=y_true,
        y_pred=y_pred,
    )

    return {
        "model": name,
        "MAE": mean_absolute_error(
            actual,
            predicted,
        ),
        "RMSE": root_mean_squared_error(
            actual,
            predicted,
        ),
        "MASE": mean_absolute_scaled_error(
            y_true=actual,
            y_pred=predicted,
            y_train=y_train,
            seasonality=seasonality,
        ),
        "Bias": forecast_bias(
            actual,
            predicted,
        ),
        "sMAPE": symmetric_mean_absolute_percentage_error(
            actual,
            predicted,
        ),
    }