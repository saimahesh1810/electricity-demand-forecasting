import pandas as pd

from electricity_demand.config import (
    FIGURE_DIR,
    METRICS_DIR,
    PROCESSED_WEEKLY_PATH,
    REPORT_FIGURE_DIR,
    SEASONAL_PERIOD,
)
from electricity_demand.data import (
    make_processed_weekly_data,
)
from electricity_demand.evaluation import (
    augmented_dickey_fuller_test,
)
from electricity_demand.plotting import (
    plot_autocorrelation_diagnostics,
    plot_load_temperature_relationship,
    plot_seasonal_decomposition,
    plot_week_of_year_pattern,
    plot_weekly_load,
    plot_yearly_load_profiles,
)


def save_figure(
    figure,
    filename: str,
) -> None:
    """
    Save a figure to both outputs/figures and reports/figures.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        REPORT_FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )


def main() -> None:
    """
    Create the processed weekly dataset and Part 1 analysis outputs.
    """

    weekly_data = make_processed_weekly_data()

    print(
        "\nProcessed weekly dataset created successfully."
    )

    print(
        f"\nSaved to:\n{PROCESSED_WEEKLY_PATH}"
    )

    print("\nDataset shape:")
    print(weekly_data.shape)

    print("\nDate range:")
    print(
        weekly_data.index.min(),
        "to",
        weekly_data.index.max(),
    )

    print("\nFirst five rows:")
    print(weekly_data.head())

    print("\nMissing values:")
    print(weekly_data.isna().sum())

    print("\nSummary statistics:")
    print(
        weekly_data.describe().round(3)
    )

    load = weekly_data["load_gw"]

    from electricity_demand.config import TEST_WEEKS

    training_load = load.iloc[:-TEST_WEEKS]

    weekly_load_figure = plot_weekly_load(
        load
    )

    save_figure(
        weekly_load_figure,
        "weekly_load_series.png",
    )

    seasonal_pattern_figure = (
        plot_week_of_year_pattern(
            load
        )
    )

    save_figure(
        seasonal_pattern_figure,
        "week_of_year_pattern.png",
    )

    yearly_profiles_figure = (
        plot_yearly_load_profiles(
            load
        )
    )

    save_figure(
        yearly_profiles_figure,
        "yearly_load_profiles.png",
    )

    temperature_figure = (
        plot_load_temperature_relationship(
            weekly_data
        )
    )

    save_figure(
        temperature_figure,
        "load_temperature_relationship.png",
    )

    decomposition_figure = (
        plot_seasonal_decomposition(
            load,
            period=SEASONAL_PERIOD,
        )
    )

    save_figure(
        decomposition_figure,
        "seasonal_decomposition.png",
    )

    acf_figure, pacf_figure = (
        plot_autocorrelation_diagnostics(
            load,
            acf_lags=104,
            pacf_lags=52,
        )
    )

    save_figure(
        acf_figure,
        "weekly_load_acf.png",
    )

    save_figure(
        pacf_figure,
        "weekly_load_pacf.png",
    )

    adf_original = (
        augmented_dickey_fuller_test(
            training_load
        )
    )

    first_difference = (
        training_load.diff().dropna()
    )

    adf_first_difference = (
        augmented_dickey_fuller_test(
            first_difference
        )
    )

    seasonal_difference = (
        training_load.diff(
            SEASONAL_PERIOD
        ).dropna()
    )

    adf_seasonal_difference = (
        augmented_dickey_fuller_test(
            seasonal_difference
        )
    )

    adf_results = pd.DataFrame(
        [
            {
                "series": "original",
                **adf_original,
            },
            {
                "series": "first_difference",
                **adf_first_difference,
            },
            {
                "series": "seasonal_difference_52",
                **adf_seasonal_difference,
            },
        ]
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    adf_output_path = (
        METRICS_DIR
        / "stationarity_tests.csv"
    )

    adf_results.to_csv(
        adf_output_path,
        index=False,
    )

    summary_output_path = (
        METRICS_DIR
        / "processed_data_summary.csv"
    )

    weekly_data.describe().T.to_csv(
        summary_output_path
    )

    print("\nADF stationarity results:")

    print(
        adf_results[
            [
                "series",
                "test_statistic",
                "p_value",
                "used_lags",
                "is_stationary",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print(
        f"\nStationarity results saved to:\n"
        f"{adf_output_path}"
    )

    print(
        "\nEDA figures saved to:\n"
        f"{FIGURE_DIR}"
    )

    print(
        "\nReport copies saved to:\n"
        f"{REPORT_FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()