from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM


def set_random_seeds(
    seed: int = 42,
) -> None:
    """
    Set reproducible random seeds where possible.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_lstm_sequences(
    values: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2D time-series array into supervised LSTM sequences.

    Parameters
    ----------
    values:
        Array shaped as (time_steps, features).

    lookback:
        Number of past weeks used to predict the next week.

    Returns
    -------
    X:
        Array shaped as (samples, lookback, features).

    y:
        One-step-ahead target values from the first column.
    """

    array = np.asarray(
        values,
        dtype=np.float32,
    )

    if array.ndim != 2:
        raise ValueError(
            "values must be a 2D array."
        )

    if lookback <= 0:
        raise ValueError(
            "lookback must be positive."
        )

    if len(array) <= lookback:
        raise ValueError(
            "Not enough observations to create sequences."
        )

    X_sequences: list[np.ndarray] = []

    y_values: list[float] = []

    for position in range(
        lookback,
        len(array),
    ):
        X_sequences.append(
            array[
                position - lookback:
                position
            ]
        )

        y_values.append(
            float(
                array[position, 0]
            )
        )

    return (
        np.asarray(
            X_sequences,
            dtype=np.float32,
        ),
        np.asarray(
            y_values,
            dtype=np.float32,
        ),
    )


def build_lstm_model(
    lookback: int,
    number_of_features: int,
    seed: int = 42,
) -> tf.keras.Model:
    """
    Build a compact LSTM suitable for a small weekly dataset.
    """

    set_random_seeds(seed)

    model = Sequential(
        [
            tf.keras.Input(
                shape=(
                    lookback,
                    number_of_features,
                )
            ),
            LSTM(
                units=24,
                return_sequences=False,
            ),
            Dropout(0.2),
            Dense(
                units=12,
                activation="relu",
            ),
            Dense(
                units=1,
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def fit_lstm_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    epochs: int = 300,
    batch_size: int = 16,
) -> tf.keras.callbacks.History:
    """
    Fit the LSTM with early stopping.
    """

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
        min_delta=1e-5,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[
            early_stopping,
        ],
        verbose=1,
    )

    return history


def recursive_lstm_forecast(
    model: tf.keras.Model,
    scaled_history: np.ndarray,
    scaled_future_covariates: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """
    Produce a fixed-origin recursive LSTM forecast.

    The first feature is assumed to be the scaled target.
    Later test-period target values are replaced by predictions.
    """

    history = np.asarray(
        scaled_history,
        dtype=np.float32,
    ).copy()

    future_covariates = np.asarray(
        scaled_future_covariates,
        dtype=np.float32,
    )

    if history.ndim != 2:
        raise ValueError(
            "scaled_history must be two-dimensional."
        )

    if future_covariates.ndim != 2:
        raise ValueError(
            "scaled_future_covariates must be two-dimensional."
        )

    predictions: list[float] = []

    for covariate_row in future_covariates:
        input_window = history[
            -lookback:
        ].reshape(
            1,
            lookback,
            history.shape[1],
        )

        predicted_target = float(
            model.predict(
                input_window,
                verbose=0,
            )[0, 0]
        )

        if not np.isfinite(
            predicted_target
        ):
            raise ValueError(
                "LSTM produced a non-finite prediction."
            )

        predictions.append(
            predicted_target
        )

        next_row = np.concatenate(
            [
                np.array(
                    [predicted_target],
                    dtype=np.float32,
                ),
                covariate_row.astype(
                    np.float32
                ),
            ]
        )

        history = np.vstack(
            [
                history,
                next_row,
            ]
        )

    return np.asarray(
        predictions,
        dtype=np.float32,
    )


def inverse_target_scale(
    scaled_target: np.ndarray,
    scaler: StandardScaler,
    number_of_features: int,
) -> np.ndarray:
    """
    Invert scaling for the target column only.
    """

    scaled_target = np.asarray(
        scaled_target,
        dtype=float,
    ).reshape(
        -1,
        1,
    )

    placeholder = np.zeros(
        (
            len(scaled_target),
            number_of_features,
        ),
        dtype=float,
    )

    placeholder[:, 0] = (
        scaled_target[:, 0]
    )

    inverted = scaler.inverse_transform(
        placeholder
    )

    return inverted[:, 0]