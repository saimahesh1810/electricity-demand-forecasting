import pandas as pd

from electricity_demand.config import (
    METRICS_DIR,
    SEASONAL_PERIOD,
    TEST_WEEKS,
)
from electricity_demand.data import (
    load_processed_data,
)
from electricity_demand.evaluation import (
    evaluate_forecast,
)
from electricity_demand.models.benchmarks import (
    drift_forecast,
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)


def main() -> None:
    """
    Evaluate benchmark forecasts on the final 104 weeks.
    """

    data = load_processed_data()

    y = data["load_gw"]

    train = y.iloc[:-TEST_WEEKS]

    test = y.iloc[-TEST_WEEKS:]

    horizon = len(test)

    forecasts = {
        "mean": mean_forecast(
            train,
            horizon,
            index=test.index,
        ),
        "naive": naive_forecast(
            train,
            horizon,
            index=test.index,
        ),
        "seasonal_naive": seasonal_naive_forecast(
            train,
            horizon,
            seasonality=SEASONAL_PERIOD,
            index=test.index,
        ),
        "drift": drift_forecast(
            train,
            horizon,
            index=test.index,
        ),
    }

    metrics = []

    for name, forecast in forecasts.items():
        result = evaluate_forecast(
            name=name,
            y_true=test,
            y_pred=forecast,
            y_train=train,
            seasonality=SEASONAL_PERIOD,
        )

        metrics.append(result)

    metrics_df = pd.DataFrame(
        metrics
    ).sort_values(
        "MASE"
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        METRICS_DIR
        / "benchmark_metrics.csv"
    )

    metrics_df.to_csv(
        output_path,
        index=False,
    )

    print("\nTraining observations:")
    print(len(train))

    print("\nTest observations:")
    print(len(test))

    print("\nTrain period:")
    print(
        train.index.min(),
        "to",
        train.index.max(),
    )

    print("\nTest period:")
    print(
        test.index.min(),
        "to",
        test.index.max(),
    )

    print("\nBenchmark results:")
    print(
        metrics_df.round(4).to_string(
            index=False
        )
    )

    print(
        f"\nSaved metrics to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()