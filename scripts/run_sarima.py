import pickle

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

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
    search_sarima_orders,
)
from electricity_demand.plotting import (
    plot_residual_diagnostics,
    plot_sarima_forecast,
)


SEASONAL_ORDER = (
    1,
    0,
    1,
    SEASONAL_PERIOD,
)


def main() -> None:
    """
    Search, fit and evaluate a weekly SARIMA model.
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

    search_output_path = (
        METRICS_DIR
        / "sarima_order_search.csv"
    )

    if search_output_path.exists():
        print(
            "\nLoading existing SARIMA search results."
        )

        search_results = pd.read_csv(
            search_output_path
        )

    else:
        print(
            "\nSearching SARIMA orders using training data..."
        )

        search_results = search_sarima_orders(
            y_train=train,
            p_values=range(7),
            d_values=range(3),
            q_values=range(7),
            seasonal_order=SEASONAL_ORDER,
        )

        search_results.to_csv(
            search_output_path,
            index=False,
        )

    valid_candidates = search_results[
        search_results["converged"]
        & search_results["aic"].notna()
    ].sort_values(
        "aic"
    ).reset_index(drop=True)
        


    if valid_candidates.empty:
        raise ValueError(
            "No converged SARIMA candidates are available."
        )

    model_fit = None
    forecast = None
    intervals = None
    best_order = None
    selected_aic = None

    rejected_candidates = []

    print(
        "\nChecking candidates for forecast stability..."
    )

    for _, candidate in valid_candidates.iterrows():
        candidate_order = (
            int(candidate["p"]),
            int(candidate["d"]),
            int(candidate["q"]),
        )

        candidate_aic = float(
            candidate["aic"]
        )

        print(
            f"Testing SARIMA{candidate_order}"
            f"x{SEASONAL_ORDER} "
            f"(AIC={candidate_aic:.3f})"
        )

        try:
            candidate_fit = fit_sarimax(
                y_train=train,
                order=candidate_order,
                seasonal_order=SEASONAL_ORDER,
            )

            candidate_forecast, candidate_intervals = (
                forecast_sarimax(
                    model_fit=candidate_fit,
                    horizon=horizon,
                    index=test.index,
                )
            )

            stable = is_stable_forecast(
                forecast=candidate_forecast,
                intervals=candidate_intervals,
                y_train=train,
            )

            if not stable:
                rejected_candidates.append(
                    {
                        "order": str(candidate_order),
                        "aic": candidate_aic,
                        "reason": (
                            "Non-finite or implausible forecast"
                        ),
                    }
                )

                print("  Rejected: forecast contains non-finite, "
                    "reversed, or catastrophically large values."
                    )

                continue

            model_fit = candidate_fit
            forecast = candidate_forecast
            intervals = candidate_intervals
            best_order = candidate_order
            selected_aic = candidate_aic

            print(
                "  Accepted: forecast is stable."
            )

            break

        except Exception as error:
            rejected_candidates.append(
                {
                    "order": str(candidate_order),
                    "aic": candidate_aic,
                    "reason": str(error),
                }
            )

            print(
                f"  Rejected due to error: {error}"
            )

    if model_fit is None:
        raise ValueError(
            "No numerically stable SARIMA model was found."
        )

    print(
        f"\nSelected stable order: {best_order}"
    )

    print(
        f"Seasonal order: {SEASONAL_ORDER}"
    )

    print(
        f"Training AIC: {selected_aic:.3f}"
    )

    rejected_df = pd.DataFrame(
        rejected_candidates
    )

    rejected_output_path = (
        METRICS_DIR
        / "sarima_rejected_candidates.csv"
    )

    rejected_df.to_csv(
        rejected_output_path,
        index=False,
    )


    forecast.name = "sarima"

    metrics = evaluate_forecast(
        name="sarima",
        y_true=test,
        y_pred=forecast,
        y_train=train,
        seasonality=SEASONAL_PERIOD,
    )

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_output_path = (
        METRICS_DIR
        / "sarima_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_output_path,
        index=False,
    )

    forecast_df = pd.DataFrame(
        {
            "actual": test,
            "sarima": forecast,
            "sarima_lower_95": intervals["lower"],
            "sarima_upper_95": intervals["upper"],
        }
    )

    forecast_output_path = (
        FORECAST_DIR
        / "sarima_forecast.csv"
    )

    forecast_df.to_csv(
        forecast_output_path
    )

    residuals = pd.Series(
        model_fit.resid,
        index=train.index,
        name="residual",
    ).dropna()

    residual_output_path = (
        FORECAST_DIR
        / "sarima_training_residuals.csv"
    )

    residuals.to_csv(
        residual_output_path
    )

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
        / "sarima_ljung_box.csv"
    )

    ljung_box.to_csv(
        ljung_box_output_path
    )

    forecast_figure = plot_sarima_forecast(
        train=train,
        test=test,
        forecast=forecast,
        intervals=intervals,
    )

    forecast_figure.savefig(
        FIGURE_DIR
        / "sarima_forecast.png",
        dpi=300,
        bbox_inches="tight",
    )

    diagnostic_figures = (
        plot_residual_diagnostics(
            residuals
        )
    )

    for name, figure in diagnostic_figures.items():
        figure.savefig(
            FIGURE_DIR
            / f"sarima_{name}.png",
            dpi=300,
            bbox_inches="tight",
        )

    model_output_path = (
        MODEL_OBJECT_DIR
        / "sarima_model.pkl"
    )

    with model_output_path.open(
        "wb"
    ) as model_file:
        pickle.dump(
            model_fit,
            model_file,
        )

    print("\nTop 10 converged models:")

    print(
        search_results[
            search_results["converged"]
        ][
            [
                "p",
                "d",
                "q",
                "aic",
                "bic",
            ]
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    print("\nSARIMA test metrics:")

    print(
        metrics_df.round(4).to_string(
            index=False
        )
    )

    print("\nLjung-Box residual test:")

    print(
        ljung_box.round(4).to_string()
    )

    print(
        f"\nSearch results saved to:\n"
        f"{search_output_path}"
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