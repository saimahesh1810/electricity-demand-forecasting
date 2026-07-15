from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarimax(
    y_train: pd.Series,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 52),
    X_train: pd.DataFrame | None = None,
    trend: str | None = None,
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
    maxiter: int = 1000,
):
    """
    Fit a SARIMA or SARIMAX model.

    When X_train is None, this is a SARIMA model.
    When X_train is supplied, this is a SARIMAX model.
    """

    y = pd.Series(
        y_train,
        copy=True,
    ).astype(float)

    if y.empty:
        raise ValueError(
            "Training target must not be empty."
        )

    if y.isna().any():
        raise ValueError(
            "Training target contains missing values."
        )

    exog = None

    if X_train is not None:
        exog = pd.DataFrame(
            X_train,
            copy=True,
        ).astype(float)

        exog = exog.reindex(y.index)

        if exog.isna().any().any():
            raise ValueError(
                "Training exogenous variables contain "
                "missing values after alignment."
            )

    model = SARIMAX(
        endog=y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fitted_model = model.fit(
          method="lbfgs",
          disp=False,
          maxiter=maxiter,
        )
    return fitted_model


def forecast_sarimax(
    model_fit,
    horizon: int,
    X_test: pd.DataFrame | None = None,
    index: pd.Index | None = None,
    alpha: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Generate point forecasts and prediction intervals.
    """

    if horizon <= 0:
        raise ValueError(
            "Forecast horizon must be greater than zero."
        )

    exog = None

    if X_test is not None:
        exog = pd.DataFrame(
            X_test,
            copy=True,
        ).astype(float)

        if len(exog) != horizon:
            raise ValueError(
                "X_test length must match the forecast horizon."
            )

        if exog.isna().any().any():
            raise ValueError(
                "X_test contains missing values."
            )

    forecast_result = model_fit.get_forecast(
        steps=horizon,
        exog=exog,
    )

    forecast_values = np.asarray(
        forecast_result.predicted_mean,
        dtype=float,
    )

    confidence_values = np.asarray(
        forecast_result.conf_int(
            alpha=alpha
        ),
        dtype=float,
    )

    if index is None:
        forecast_index = pd.RangeIndex(
            horizon
        )
    else:
        if len(index) != horizon:
            raise ValueError(
                "Forecast index length must match horizon."
            )

        forecast_index = index

    forecast = pd.Series(
        forecast_values,
        index=forecast_index,
        name="sarima",
    )

    intervals = pd.DataFrame(
        {
            "lower": confidence_values[:, 0],
            "upper": confidence_values[:, 1],
        },
        index=forecast_index,
    )

    return forecast, intervals


def search_sarima_orders(
    y_train: pd.Series,
    p_values: Iterable[int] = range(7),
    d_values: Iterable[int] = range(3),
    q_values: Iterable[int] = range(7),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 52),
) -> pd.DataFrame:
    """
    Search SARIMA non-seasonal orders using training-set AIC.

    The seasonal order is held fixed while p, d, and q are searched.

    Failed or non-converged models are retained in the output with
    an appropriate status.
    """

    y = pd.Series(
        y_train,
        copy=True,
    ).astype(float)

    if y.isna().any():
        raise ValueError(
            "Training target contains missing values."
        )

    results: list[dict] = []

    combinations = [
        (p, d, q)
        for p in p_values
        for d in d_values
        for q in q_values
    ]

    total_models = len(combinations)

    for position, order in enumerate(
        combinations,
        start=1,
    ):
        print(
            f"\rFitting model {position}/{total_models}: "
            f"SARIMA{order}x{seasonal_order}",
            end="",
        )

        try:
            fitted = fit_sarimax(
                y_train=y,
                order=order,
                seasonal_order=seasonal_order,
            )

            converged = bool(
                fitted.mle_retvals.get(
                    "converged",
                    False,
                )
            )

            results.append(
                {
                    "p": order[0],
                    "d": order[1],
                    "q": order[2],
                    "P": seasonal_order[0],
                    "D": seasonal_order[1],
                    "Q": seasonal_order[2],
                    "seasonal_period": seasonal_order[3],
                    "aic": float(fitted.aic),
                    "bic": float(fitted.bic),
                    "log_likelihood": float(fitted.llf),
                    "converged": converged,
                    "status": (
                        "success"
                        if converged
                        else "not_converged"
                    ),
                    "error": "",
                }
            )

        except Exception as error:
            results.append(
                {
                    "p": order[0],
                    "d": order[1],
                    "q": order[2],
                    "P": seasonal_order[0],
                    "D": seasonal_order[1],
                    "Q": seasonal_order[2],
                    "seasonal_period": seasonal_order[3],
                    "aic": np.nan,
                    "bic": np.nan,
                    "log_likelihood": np.nan,
                    "converged": False,
                    "status": "failed",
                    "error": str(error),
                }
            )

    print()

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by=[
            "converged",
            "aic",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    return results_df


def select_best_sarima_order(
    search_results: pd.DataFrame,
) -> tuple[int, int, int]:
    """
    Select the converged SARIMA order with the lowest AIC.
    """

    required_columns = {
        "p",
        "d",
        "q",
        "aic",
        "converged",
    }

    missing = (
        required_columns
        - set(search_results.columns)
    )

    if missing:
        raise ValueError(
            f"Search results are missing columns: "
            f"{sorted(missing)}"
        )

    valid_results = search_results[
        search_results["converged"]
        & search_results["aic"].notna()
    ]

    if valid_results.empty:
        raise ValueError(
            "No converged SARIMA models were found."
        )

    best_row = valid_results.sort_values(
        "aic"
    ).iloc[0]

    return (
        int(best_row["p"]),
        int(best_row["d"]),
        int(best_row["q"]),
    )

def is_stable_forecast(
    forecast: pd.Series,
    intervals: pd.DataFrame,
    y_train: pd.Series,
) -> bool:
    """
    Check that forecasts are finite and within a broadly plausible
    range for the observed electricity-demand series.
    """

    forecast_values = np.asarray(
        forecast,
        dtype=float,
    )

    interval_values = np.asarray(
        intervals,
        dtype=float,
    )

    if not np.isfinite(forecast_values).all():
        return False

    if not np.isfinite(interval_values).all():
        return False

    if (intervals["lower"] > intervals["upper"]).any():
        return False

    train_min = float(y_train.min())

    train_max = float(y_train.max())

    train_range = train_max - train_min

    plausible_lower = max(
        0.0,
        train_min - 2 * train_range,
    )

    plausible_upper = (
        train_max + 2 * train_range
    )

    if forecast_values.min() < plausible_lower:
        return False

    if forecast_values.max() > plausible_upper:
        return False

    return True

def diagnose_sarima_fit(
    model_fit,
    order: tuple[int, int, int],
) -> dict:
    """
    Extract numerical-stability diagnostics from a fitted SARIMA model.
    """

    parameters = np.asarray(
        model_fit.params,
        dtype=float,
    )

    parameter_standard_errors = np.asarray(
        model_fit.bse,
        dtype=float,
    )

    ar_roots = np.asarray(
        model_fit.arroots,
        dtype=complex,
    )

    ma_roots = np.asarray(
        model_fit.maroots,
        dtype=complex,
    )

    minimum_ar_root = (
        float(np.min(np.abs(ar_roots)))
        if len(ar_roots) > 0
        else np.nan
    )

    minimum_ma_root = (
        float(np.min(np.abs(ma_roots)))
        if len(ma_roots) > 0
        else np.nan
    )

    residuals = np.asarray(
        model_fit.resid,
        dtype=float,
    )

    return {
        "order": str(order),
        "aic": float(model_fit.aic),
        "bic": float(model_fit.bic),
        "log_likelihood": float(model_fit.llf),
        "converged": bool(
            model_fit.mle_retvals.get(
                "converged",
                False,
            )
        ),
        "finite_parameters": bool(
            np.isfinite(parameters).all()
        ),
        "finite_standard_errors": bool(
            np.isfinite(
                parameter_standard_errors
            ).all()
        ),
        "minimum_ar_root": minimum_ar_root,
        "minimum_ma_root": minimum_ma_root,
        "residual_std": float(
            np.nanstd(residuals)
        ),
    }