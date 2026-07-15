import pickle

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
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
from electricity_demand.models.sarimax import (
    fit_sarimax,
    forecast_sarimax,
    is_stable_forecast,
)
from electricity_demand.plotting import (
    plot_sarimax_forecast,
)


SARIMA_ORDER = (
    1,
    1,
    1,
)

SEASONAL_ORDER = (
    1,
    1,
    1,
    SEASONAL_PERIOD,
)


# Predefined covariate sets.
# Selection is based on training AIC, not test accuracy.
COVARIATE_SETS = {
    "temperature": [
        "temp_mean",
    ],
    "temperature_holiday": [
        "temp_mean",
        "holiday_days",
    ],
}


def main() -> None:
    """
    Fit and evaluate weekly SARIMAX models using external covariates.

    Future test-period weather values are realised observations.
    The forecasts are therefore conditional forecasts.
    """

    data = load_processed_data()

    y = data["load_gw"]

    train = y.iloc[:-TEST_WEEKS]

    test = y.iloc[-TEST_WEEKS:]

    horizon = len(test)

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

    candidate_results = []

    fitted_candidates = {}

    print(
        "\nFitting predefined SARIMAX covariate models..."
    )

    for candidate_name, columns in COVARIATE_SETS.items():
        print(
            f"\nCandidate: {candidate_name}"
        )

        print(
            f"Covariates: {columns}"
        )

        X = data[columns].astype(float)

        X_train = X.iloc[:-TEST_WEEKS].copy()
        X_test = X.iloc[-TEST_WEEKS:].copy()

        continuous_columns = [
             column
             for column in columns
             if column not in [
                "holiday_days",
                "has_holiday",
                 ]
            ]

        if continuous_columns:
           scaler = StandardScaler()

           X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns,
            )

           X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns,
            )

           X_train = X_train_scaled

           X_test = X_test_scaled

        try:
            model_fit = fit_sarimax(
            y_train=train,
            X_train=X_train,
            order=SARIMA_ORDER,
            seasonal_order=SEASONAL_ORDER,
            trend=None,
            maxiter=1000,
            )

            converged = bool(
                model_fit.mle_retvals.get(
                    "converged",
                    False,
                )
            )

            forecast, intervals = forecast_sarimax(
                model_fit=model_fit,
                horizon=horizon,
                X_test=X_test,
                index=test.index,
            )

            print(
                f"Forecast range: "
                f"{forecast.min():.3f} to "
                f"{forecast.max():.3f} GW"
            )

            print(
                f"Training range: "
                f"{train.min():.3f} to "
                f"{train.max():.3f} GW"
            )

            stable = is_stable_forecast(
                forecast=forecast,
                intervals=intervals,
                y_train=train,
            )

            candidate_results.append(
                {
                    "candidate": candidate_name,
                    "covariates": ", ".join(columns),
                    "aic": float(model_fit.aic),
                    "bic": float(model_fit.bic),
                    "converged": converged,
                    "stable_forecast": stable,
                    "status": (
                        "usable"
                        if converged and stable
                        else "rejected"
                    ),
                    "error": "",
                }
            )

            if converged and stable:
                fitted_candidates[
                    candidate_name
                ] = {
                    "model_fit": model_fit,
                    "forecast": forecast,
                    "intervals": intervals,
                    "columns": columns,
                }

                print(
                    f"AIC: {model_fit.aic:.3f}"
                )

                print(
                    "Forecast status: usable"
                )

            else:
                print(
                    "Forecast status: rejected"
                )

        except Exception as error:
            candidate_results.append(
                {
                    "candidate": candidate_name,
                    "covariates": ", ".join(columns),
                    "aic": float("nan"),
                    "bic": float("nan"),
                    "converged": False,
                    "stable_forecast": False,
                    "status": "failed",
                    "error": str(error),
                }
            )

            print(
                f"Candidate failed: {error}"
            )

    candidate_results_df = pd.DataFrame(
        candidate_results
    ).sort_values(
        "aic",
        na_position="last",
    )

    candidate_output_path = (
        METRICS_DIR
        / "sarimax_candidate_comparison.csv"
    )

    candidate_results_df.to_csv(
        candidate_output_path,
        index=False,
    )

    usable_candidates = candidate_results_df[
        candidate_results_df["status"]
        == "usable"
    ].sort_values(
        "aic"
    )
    print(
        "\nFull candidate diagnostics:"
    )

    print(
        candidate_results_df[
            [
                "candidate",
                "aic",
                "bic",
                "converged",
                "stable_forecast",
                "status",
                "error",
            ]
        ].to_string(index=False)
    )
    if usable_candidates.empty:
        raise ValueError(
            "No usable SARIMAX covariate model was found."
        )

    selected_name = usable_candidates.iloc[
        0
    ]["candidate"]

    selected = fitted_candidates[
        selected_name
    ]

    model_fit = selected["model_fit"]

    forecast = selected["forecast"]

    intervals = selected["intervals"]

    selected_columns = selected["columns"]

    forecast.name = "sarimax"

    print(
        f"\nSelected SARIMAX candidate: "
        f"{selected_name}"
    )

    print(
        f"Selected covariates: "
        f"{selected_columns}"
    )

    print(
        f"Training AIC: "
        f"{model_fit.aic:.3f}"
    )

    metrics = evaluate_forecast(
        name="sarimax",
        y_true=test,
        y_pred=forecast,
        y_train=train,
        seasonality=SEASONAL_PERIOD,
    )

    metrics["covariate_set"] = (
        selected_name
    )

    metrics["conditional_forecast"] = True

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_output_path = (
        METRICS_DIR
        / "sarimax_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_output_path,
        index=False,
    )

    forecast_df = pd.DataFrame(
        {
            "actual": test,
            "sarimax": forecast,
            "sarimax_lower_95": intervals["lower"],
            "sarimax_upper_95": intervals["upper"],
        }
    )

    for column in selected_columns:
        forecast_df[column] = data.loc[
            test.index,
            column,
        ]

    forecast_output_path = (
        FORECAST_DIR
        / "sarimax_forecast.csv"
    )

    forecast_df.to_csv(
        forecast_output_path
    )

    residuals = pd.Series(
        model_fit.resid,
        index=train.index,
        name="residual",
    ).dropna()

    requested_lags = [
        lag
        for lag in [
            12,
            26,
            52,
        ]
        if lag < len(residuals)
    ]

    ljung_box = acorr_ljungbox(
        residuals,
        lags=requested_lags,
        return_df=True,
    )

    ljung_box.index.name = "lag"

    ljung_box_output_path = (
        METRICS_DIR
        / "sarimax_ljung_box.csv"
    )

    ljung_box.to_csv(
        ljung_box_output_path
    )

    figure = plot_sarimax_forecast(
        train=train,
        test=test,
        forecast=forecast,
        intervals=intervals,
        model_label=(
            "SARIMAX "
            f"({selected_name.replace('_', ' ')})"
        ),
    )

    figure_output_path = (
        FIGURE_DIR
        / "sarimax_forecast.png"
    )

    figure.savefig(
        figure_output_path,
        dpi=300,
        bbox_inches="tight",
    )

    model_output_path = (
        MODEL_OBJECT_DIR
        / "sarimax_model.pkl"
    )

    with model_output_path.open(
        "wb"
    ) as model_file:
        pickle.dump(
            model_fit,
            model_file,
        )

    print(
        "\nSARIMAX candidate comparison:"
    )

    print(
        candidate_results_df[
            [
                "candidate",
                "aic",
                "bic",
                "converged",
                "stable_forecast",
                "status",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nSARIMAX test metrics:"
    )

    print(
        metrics_df.round(4).to_string(
            index=False
        )
    )

    print(
        "\nLjung-Box residual test:"
    )

    print(
        ljung_box.round(4).to_string()
    )

    print(
        "\nImportant: this is a conditional forecast "
        "because realised future temperature covariates "
        "were used during the test period."
    )

    print(
        f"\nCandidate comparison saved to:\n"
        f"{candidate_output_path}"
    )

    print(
        f"\nForecast saved to:\n"
        f"{forecast_output_path}"
    )

    print(
        f"\nModel saved to:\n"
        f"{model_output_path}"
    )


if __name__ == "__main__":
    main()