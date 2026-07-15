import pickle

import pandas as pd

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
from electricity_demand.features import (
    make_ml_table,
)
from electricity_demand.models.feature_models import (
    fit_gradient_boosting,
    get_feature_importance,
    recursive_feature_forecast,
)
from electricity_demand.plotting import (
    plot_feature_importance,
    plot_forecasts,
)


def main() -> None:
    """
    Fit and evaluate a recursive Gradient Boosting forecast.
    """

    data = load_processed_data()

    train_data = data.iloc[
        :-TEST_WEEKS
    ].copy()

    test_data = data.iloc[
        -TEST_WEEKS:
    ].copy()

    train_target = train_data[
        "load_gw"
    ]

    test_target = test_data[
        "load_gw"
    ]

    print(
        "\nCreating leakage-safe training features..."
    )

    ml_train = make_ml_table(
        train_data
    )

    target_column = "load_gw"

    feature_columns = [
        column
        for column in ml_train.columns
        if column != target_column
    ]

    X_train = ml_train[
        feature_columns
    ]

    y_train = ml_train[
        target_column
    ]

    print(
        f"Raw training weeks: "
        f"{len(train_data)}"
    )

    print(
        f"Usable supervised rows: "
        f"{len(ml_train)}"
    )

    print(
        f"Number of features: "
        f"{len(feature_columns)}"
    )

    model = fit_gradient_boosting(
        X_train=X_train,
        y_train=y_train,
    )

    future_covariates = test_data.drop(
        columns=[
            target_column,
        ]
    )

    forecast = recursive_feature_forecast(
        model=model,
        y_history=train_target,
        future_covariates=future_covariates,
        feature_columns=feature_columns,
        start_timestamp=data.index.min(),
    )

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
        name="feature_model",
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

    metrics_df = pd.DataFrame(
        [metrics]
    )

    feature_importance = (
        get_feature_importance(
            model=model,
            feature_columns=feature_columns,
        )
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
        / "feature_model_forecast.csv"
    )

    forecast_df = pd.DataFrame(
        {
            "actual": test_target,
            "feature_model": forecast,
        }
    )

    forecast_df.to_csv(
        forecast_output_path
    )

    metrics_output_path = (
        METRICS_DIR
        / "feature_model_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_output_path,
        index=False,
    )

    importance_output_path = (
        METRICS_DIR
        / "feature_importance.csv"
    )

    feature_importance.to_csv(
        importance_output_path,
        header=True,
    )

    forecast_figure = plot_forecasts(
        train=train_target,
        test=test_target,
        forecasts={
            "feature_model": forecast,
        },
    )

    forecast_figure_path = (
        FIGURE_DIR
        / "feature_model_forecast.png"
    )

    forecast_figure.savefig(
        forecast_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    importance_figure = (
        plot_feature_importance(
            feature_importance,
            top_n=20,
        )
    )

    importance_figure_path = (
        FIGURE_DIR
        / "feature_importance.png"
    )

    importance_figure.savefig(
        importance_figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    model_output_path = (
        MODEL_OBJECT_DIR
        / "gradient_boosting_model.pkl"
    )

    with model_output_path.open(
        "wb"
    ) as model_file:
        pickle.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "training_start": data.index.min(),
            },
            model_file,
        )

    print(
        "\nFeature-model metrics:"
    )

    print(
        metrics_df
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nTop 15 features:"
    )

    print(
        feature_importance
        .head(15)
        .round(5)
        .to_string()
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
        "\nImportant: load lags were forecast recursively, "
        "so actual test-period load values were not used. "
        "Realised test-period weather was used, making this "
        "a conditional forecast."
    )


if __name__ == "__main__":
    main()