from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from electricity_demand.config import (
    RANDOM_STATE,
)
from electricity_demand.features import (
    DEFAULT_COVARIATES,
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    make_single_feature_row,
)


def fit_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> GradientBoostingRegressor:
    """
    Fit a Gradient Boosting regression model.

    Fixed hyperparameters are used to avoid selecting the model based
    on test-period performance.
    """

    X = pd.DataFrame(
        X_train,
        copy=True,
    ).astype(float)

    y = pd.Series(
        y_train,
        copy=True,
    ).astype(float)

    if X.empty:
        raise ValueError(
            "Training features must not be empty."
        )

    if len(X) != len(y):
        raise ValueError(
            "X_train and y_train must have equal lengths."
        )

    if X.isna().any().any():
        raise ValueError(
            "Training features contain missing values."
        )

    if y.isna().any():
        raise ValueError(
            "Training target contains missing values."
        )

    model = GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=2,
        min_samples_split=8,
        min_samples_leaf=4,
        subsample=0.9,
        loss="huber",
        random_state=random_state,
    )

    model.fit(
        X,
        y,
    )

    return model


def predict_feature_model(
    model,
    X_test: pd.DataFrame,
    index: pd.Index | None = None,
) -> pd.Series:
    """
    Generate direct predictions from a fitted feature model.

    This function is suitable when all X_test features are genuinely
    available at prediction time. For long fixed-origin forecasts
    containing load lags, use recursive_feature_forecast instead.
    """

    X = pd.DataFrame(
        X_test,
        copy=True,
    ).astype(float)

    predictions = np.asarray(
        model.predict(X),
        dtype=float,
    )

    if index is None:
        prediction_index = X.index
    else:
        if len(index) != len(predictions):
            raise ValueError(
                "Prediction index length must match predictions."
            )

        prediction_index = index

    return pd.Series(
        predictions,
        index=prediction_index,
        name="feature_model",
    )


def recursive_feature_forecast(
    model,
    y_history: pd.Series,
    future_covariates: pd.DataFrame,
    feature_columns: Sequence[str],
    start_timestamp: pd.Timestamp,
    lag_weeks: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    covariate_columns: Sequence[str] = DEFAULT_COVARIATES,
) -> pd.Series:
    """
    Produce a fixed-origin recursive weekly forecast.

    Forecasted values are appended to the load history and used when
    creating lag and rolling features for later forecast steps.
    Actual future load observations are never used.
    """

    history = pd.Series(
        y_history,
        copy=True,
    ).astype(float)

    covariates = pd.DataFrame(
        future_covariates,
        copy=True,
    )

    if not isinstance(
        covariates.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Future covariates must use a DatetimeIndex."
        )

    predictions: list[float] = []

    prediction_index: list[pd.Timestamp] = []

    for timestamp, covariate_row in covariates.iterrows():
        feature_row = make_single_feature_row(
            timestamp=timestamp,
            load_history=history,
            covariate_row=covariate_row,
            feature_columns=feature_columns,
            start_timestamp=start_timestamp,
            lag_weeks=lag_weeks,
            rolling_windows=rolling_windows,
            covariate_columns=covariate_columns,
        )

        prediction = float(
            model.predict(
                feature_row
            )[0]
        )

        if not np.isfinite(
            prediction
        ):
            raise ValueError(
                "Feature model produced a "
                "non-finite prediction."
            )

        predictions.append(
            prediction
        )

        prediction_index.append(
            timestamp
        )

        history.loc[
            timestamp
        ] = prediction

    return pd.Series(
        predictions,
        index=pd.DatetimeIndex(
            prediction_index
        ),
        name="feature_model",
    )


def get_feature_importance(
    model,
    feature_columns: Sequence[str],
) -> pd.Series:
    """
    Return Gradient Boosting feature importances.
    """

    if not hasattr(
        model,
        "feature_importances_",
    ):
        raise TypeError(
            "Model does not expose feature importances."
        )

    importances = pd.Series(
        model.feature_importances_,
        index=list(feature_columns),
        name="importance",
    )

    return importances.sort_values(
        ascending=False
    )