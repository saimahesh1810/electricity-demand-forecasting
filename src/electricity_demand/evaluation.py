from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


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

def augmented_dickey_fuller_test(
    series: pd.Series,
    significance_level: float = 0.05,
) -> dict[str, float | int | bool]:
    """
    Perform the Augmented Dickey-Fuller stationarity test.

    The null hypothesis is that the series contains a unit root and is
    therefore non-stationary.

    Parameters
    ----------
    series:
        Time series to test.

    significance_level:
        Threshold used to reject the null hypothesis.

    Returns
    -------
    dict
        Test statistic, p-value, lag information, critical values and
        stationarity decision.
    """

    values = pd.Series(
        series,
        copy=True,
    ).dropna().astype(float)

    if len(values) < 20:
        raise ValueError(
            "At least 20 observations are required for the ADF test."
        )

    result = adfuller(
        values,
        autolag="AIC",
    )

    test_statistic = float(result[0])
    p_value = float(result[1])
    used_lags = int(result[2])
    number_observations = int(result[3])
    critical_values = result[4]

    return {
        "test_statistic": test_statistic,
        "p_value": p_value,
        "used_lags": used_lags,
        "number_observations": number_observations,
        "critical_value_1_percent": float(
            critical_values["1%"]
        ),
        "critical_value_5_percent": float(
            critical_values["5%"]
        ),
        "critical_value_10_percent": float(
            critical_values["10%"]
        ),
        "significance_level": significance_level,
        "is_stationary": p_value < significance_level,
    }