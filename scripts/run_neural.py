import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from electricity_demand.config import (
    FIGURE_DIR,
    FORECAST_DIR,
    METRICS_DIR,
    MODEL_OBJECT_DIR,
    SEASONAL_PERIOD,
    TEST_WEEKS,
)
from electricity_demand.data import (
    load_processed_data,
)
from electricity_demand.evaluation import (
    evaluate_forecast,
)
from electricity_demand.models.neural import (
    build_lstm_model,
    fit_lstm_model,
    inverse_target_scale,
    make_lstm_sequences,
    recursive_lstm_forecast,
    set_random_seeds,
)
from electricity_demand.plotting import (
    plot_forecasts,
)


LOOKBACK = 52

FEATURE_COLUMNS = [
    "load_gw",
    "temp_mean",
    "holiday_days",
]


def main() -> None:
    """
    Fit and evaluate a recursive weekly LSTM forecast.
    """

    set_random_seeds(42)

    data = load_processed_data()

    train_data = data.iloc[
        :-TEST_WEEKS
    ].copy()

    test_data = data.iloc[
        -TEST_WEEKS:
    ].copy()

    train_values = train_data[
        FEATURE_COLUMNS
    ].astype(float)

    test_values = test_data[
        FEATURE_COLUMNS
    ].astype(float)

    scaler = StandardScaler()

    scaled_train = scaler.fit_transform(
        train_values
    )

    validation_weeks = 26

    sequence_X, sequence_y = (
        make_lstm_sequences(
            scaled_train,
            lookback=LOOKBACK,
        )
    )

    if len(sequence_X) <= validation_weeks:
        raise ValueError(
            "Insufficient training sequences for validation split."
        )

    X_train = sequence_X[
        :-validation_weeks
    ]

    y_train = sequence_y[
        :-validation_weeks
    ]

    X_validation = sequence_X[
        -validation_weeks:
    ]

    y_validation = sequence_y[
        -validation_weeks:
    ]

    print(
        f"\nRaw training weeks: {len(train_data)}"
    )

    print(
        f"Lookback window: {LOOKBACK} weeks"
    )

    print(
        f"Training sequences: {len(X_train)}"
    )

    print(
        f"Validation sequences: {len(X_validation)}"
    )

    print(
        f"Number of input features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    model = build_lstm_model(
        lookback=LOOKBACK,
        number_of_features=len(
            FEATURE_COLUMNS
        ),
        seed=42,
    )

    history = fit_lstm_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        epochs=300,
        batch_size=16,
    )

    scaled_test_all = scaler.transform(
        test_values
    )

    scaled_future_covariates = (
        scaled_test_all[:, 1:]
    )

    scaled_forecast = (
        recursive_lstm_forecast(
            model=model,
            scaled_history=scaled_train,
            scaled_future_covariates=(
                scaled_future_covariates
            ),
            lookback=LOOKBACK,
        )
    )

    forecast_values = inverse_target_scale(
        scaled_target=scaled_forecast,
        scaler=scaler,
        number_of_features=len(
            FEATURE_COLUMNS
        ),
    )

    forecast = pd.Series(
        forecast_values,
        index=test_data.index,
        name="lstm",
    )

    train_target = train_data[
        "load_gw"
    ]

    test_target = test_data[
        "load_gw"
    ]

    print(
        f"\nForecast range: "
        f"{forecast.min():.3f} to "
        f"{forecast.max():.3f} GW"
    )

    print(
        f"Training range: "
        f"{train_target.min():.3f} to "
        f"{train_target.max():.3f} GW"
    )

    metrics = evaluate_forecast(
        name="lstm",
        y_true=test_target,
        y_pred=forecast,
        y_train=train_target,
        seasonality=SEASONAL_PERIOD,
    )

    metrics[
        "forecast_strategy"
    ] = "recursive_fixed_origin"

    metrics[
        "conditional_forecast"
    ] = True

    metrics[
        "lookback_weeks"
    ] = LOOKBACK

    metrics_df = pd.DataFrame(
        [metrics]
    )

    for directory in [
        FIGURE_DIR,
        FORECAST_DIR,
        METRICS_DIR,
        MODEL_OBJECT_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    forecast_output_path = (
        FORECAST_DIR
        / "lstm_forecast.csv"
    )

    pd.DataFrame(
        {
            "actual": test_target,
            "lstm": forecast,
        }
    ).to_csv(
        forecast_output_path
    )

    metrics_output_path = (
        METRICS_DIR
        / "lstm_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_output_path,
        index=False,
    )

    model_output_path = (
        MODEL_OBJECT_DIR
        / "lstm_model.keras"
    )

    model.save(
        model_output_path
    )

    scaler_output_path = (
        MODEL_OBJECT_DIR
        / "lstm_scaler.json"
    )

    scaler_payload = {
        "feature_columns": FEATURE_COLUMNS,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "lookback": LOOKBACK,
    }

    with scaler_output_path.open(
        "w",
        encoding="utf-8",
    ) as scaler_file:
        json.dump(
            scaler_payload,
            scaler_file,
            indent=2,
        )

    forecast_figure = plot_forecasts(
        train=train_target,
        test=test_target,
        forecasts={
            "lstm": forecast,
        },
    )

    forecast_figure_path = (
        FIGURE_DIR
        / "lstm_forecast.png"
    )

    forecast_figure.savefig(
        forecast_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    loss_figure, loss_axis = (
        plt.subplots(
            figsize=(10, 6)
        )
    )

    loss_axis.plot(
        history.history["loss"],
        label="Training loss",
    )

    loss_axis.plot(
        history.history["val_loss"],
        label="Validation loss",
    )

    loss_axis.set_title(
        "LSTM Training and Validation Loss"
    )

    loss_axis.set_xlabel("Epoch")

    loss_axis.set_ylabel(
        "Mean squared error"
    )

    loss_axis.legend()

    loss_axis.grid(
        alpha=0.3
    )

    loss_figure.tight_layout()

    loss_figure_path = (
        FIGURE_DIR
        / "lstm_training_loss.png"
    )

    loss_figure.savefig(
        loss_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        "\nLSTM test metrics:"
    )

    print(
        metrics_df
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        f"\nEpochs completed: "
        f"{len(history.history['loss'])}"
    )

    print(
        f"\nForecast saved to:\n"
        f"{forecast_output_path}"
    )

    print(
        f"\nMetrics saved to:\n"
        f"{metrics_output_path}"
    )

    print(
        f"\nModel saved to:\n"
        f"{model_output_path}"
    )

    print(
        "\nImportant: the LSTM forecast is recursive. "
        "Actual test-period load values were not used. "
        "Realised test-period weather and holiday values "
        "were used, so this is a conditional forecast."
    )


if __name__ == "__main__":
    main()