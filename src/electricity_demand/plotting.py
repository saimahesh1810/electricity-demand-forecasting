from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.gofplots import qqplot

def plot_forecasts(
    train: pd.Series,
    test: pd.Series,
    forecasts: Mapping[str, pd.Series],
    train_weeks_to_show: int = 104,
):
    """
    Plot the training series, test observations, and model forecasts.

    Parameters
    ----------
    train:
        Training target series.

    test:
        Test target series.

    forecasts:
        Dictionary containing model names and forecast series.

    train_weeks_to_show:
        Number of final training observations shown before the test set.

    Returns
    -------
    matplotlib.figure.Figure
        Forecast comparison figure.
    """

    if train.empty:
        raise ValueError("Training series must not be empty.")

    if test.empty:
        raise ValueError("Test series must not be empty.")

    if not forecasts:
        raise ValueError("At least one forecast must be provided.")

    train_plot = train.iloc[-train_weeks_to_show:]

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        train_plot.index,
        train_plot.values,
        label="Training load",
        linewidth=1.5,
    )

    ax.plot(
        test.index,
        test.values,
        label="Actual test load",
        linewidth=2.2,
    )

    for name, forecast in forecasts.items():
        aligned_forecast = pd.Series(
            forecast,
            copy=True,
        ).reindex(test.index)

        if aligned_forecast.isna().any():
            raise ValueError(
                f"Forecast '{name}' contains missing values "
                "after alignment with the test index."
            )

        ax.plot(
            aligned_forecast.index,
            aligned_forecast.values,
            label=name.replace("_", " ").title(),
            linewidth=1.4,
            alpha=0.9,
        )

    ax.axvline(
        test.index.min(),
        linestyle="--",
        linewidth=1.2,
        label="Forecast origin",
    )

    ax.set_title(
        "German Weekly Electricity Demand: "
        "Actual Values and Forecasts"
    )

    ax.set_xlabel("Week")

    ax.set_ylabel("Average electricity demand (GW)")

    ax.legend(
        loc="best",
        ncol=2,
    )

    ax.grid(
        alpha=0.3,
    )

    fig.tight_layout()

    return fig


def plot_forecast_errors(
    y_true: pd.Series,
    forecasts: Mapping[str, pd.Series],
):
    """
    Plot forecast errors for multiple models.

    Forecast error is calculated as:

        actual - forecast

    Positive values indicate underforecasting.
    Negative values indicate overforecasting.
    """

    if y_true.empty:
        raise ValueError("Actual series must not be empty.")

    if not forecasts:
        raise ValueError("At least one forecast must be provided.")

    fig, ax = plt.subplots(figsize=(14, 6))

    for name, forecast in forecasts.items():
        aligned_forecast = pd.Series(
            forecast,
            copy=True,
        ).reindex(y_true.index)

        if aligned_forecast.isna().any():
            raise ValueError(
                f"Forecast '{name}' contains missing values "
                "after alignment."
            )

        errors = y_true - aligned_forecast

        ax.plot(
            errors.index,
            errors.values,
            label=name.replace("_", " ").title(),
            linewidth=1.3,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title("Forecast Errors Over the Test Period")

    ax.set_xlabel("Week")

    ax.set_ylabel("Error: actual minus forecast (GW)")

    ax.legend(
        loc="best",
        ncol=2,
    )

    ax.grid(
        alpha=0.3,
    )

    fig.tight_layout()

    return fig

def plot_weekly_load(
    load: pd.Series,
):
    """
    Plot the complete weekly German electricity-demand series.
    """

    if load.empty:
        raise ValueError(
            "Load series must not be empty."
        )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        load.index,
        load.values,
        linewidth=1.5,
        label="Weekly average load",
    )

    ax.set_title(
        "Weekly Average German Electricity Demand"
    )

    ax.set_xlabel("Week")

    ax.set_ylabel(
        "Average electricity demand (GW)"
    )

    ax.grid(alpha=0.3)

    ax.legend()

    fig.tight_layout()

    return fig

def plot_week_of_year_pattern(
    load: pd.Series,
):
    """
    Plot average electricity demand by ISO week of the year.
    """

    if load.empty:
        raise ValueError(
            "Load series must not be empty."
        )

    seasonal_data = pd.DataFrame(
        {
            "load_gw": load,
        }
    )

    seasonal_data["week_of_year"] = (
        seasonal_data.index.isocalendar().week.astype(int)
    )

    week_average = seasonal_data.groupby(
        "week_of_year"
    )["load_gw"].mean()

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.plot(
        week_average.index,
        week_average.values,
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    ax.set_title(
        "Average German Electricity Demand by Week of Year"
    )

    ax.set_xlabel("ISO week of year")

    ax.set_ylabel(
        "Average electricity demand (GW)"
    )

    ax.set_xlim(
        week_average.index.min(),
        week_average.index.max(),
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig

def plot_yearly_load_profiles(
    load: pd.Series,
):
    """
    Plot weekly load profiles separately for each calendar year.
    """

    if load.empty:
        raise ValueError(
            "Load series must not be empty."
        )

    profile_data = pd.DataFrame(
        {
            "load_gw": load,
        }
    )

    profile_data["year"] = (
        profile_data.index.year
    )

    profile_data["week_of_year"] = (
        profile_data.index.isocalendar().week.astype(int)
    )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    for year, group in profile_data.groupby(
        "year"
    ):
        year_profile = group.groupby(
            "week_of_year"
        )["load_gw"].mean()

        ax.plot(
            year_profile.index,
            year_profile.values,
            label=str(year),
            linewidth=1.3,
        )

    ax.set_title(
        "German Electricity-Demand Profiles by Year"
    )

    ax.set_xlabel("ISO week of year")

    ax.set_ylabel(
        "Average electricity demand (GW)"
    )

    ax.legend(
        title="Year",
        ncol=2,
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig

def plot_load_temperature_relationship(
    data: pd.DataFrame,
):
    """
    Plot weekly electricity load against weekly mean temperature.
    """

    required_columns = {
        "load_gw",
        "temp_mean",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    plotting_data = data[
        [
            "load_gw",
            "temp_mean",
        ]
    ].dropna()

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        plotting_data["temp_mean"],
        plotting_data["load_gw"],
        alpha=0.6,
    )

    coefficients = np.polyfit(
        plotting_data["temp_mean"],
        plotting_data["load_gw"],
        deg=2,
    )

    temperature_grid = np.linspace(
        plotting_data["temp_mean"].min(),
        plotting_data["temp_mean"].max(),
        200,
    )

    fitted_values = np.polyval(
        coefficients,
        temperature_grid,
    )

    ax.plot(
        temperature_grid,
        fitted_values,
        linewidth=2,
        label="Quadratic trend",
    )

    ax.set_title(
        "Weekly Electricity Demand and Mean Temperature"
    )

    ax.set_xlabel(
        "Weekly mean temperature (°C)"
    )

    ax.set_ylabel(
        "Average electricity demand (GW)"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig

def plot_seasonal_decomposition(
    load: pd.Series,
    period: int = 52,
):
    """
    Decompose weekly electricity demand into trend, seasonal and
    residual components.
    """

    if len(load.dropna()) < period * 2:
        raise ValueError(
            "At least two complete seasonal cycles are required."
        )

    decomposition = seasonal_decompose(
        load.dropna(),
        model="additive",
        period=period,
        extrapolate_trend="freq",
    )

    fig = decomposition.plot()

    fig.set_size_inches(
        14,
        10,
    )

    fig.suptitle(
        "Additive Seasonal Decomposition of Weekly Demand",
        y=1.02,
    )

    fig.tight_layout()

    return fig

def plot_autocorrelation_diagnostics(
    load: pd.Series,
    acf_lags: int = 104,
    pacf_lags: int = 52,
):
    """
    Create ACF and PACF diagnostic figures.
    """

    clean_load = load.dropna()

    if len(clean_load) <= acf_lags:
        raise ValueError(
            "Series is too short for the requested ACF lags."
        )

    maximum_pacf_lags = (
        len(clean_load) // 2
    ) - 1

    pacf_lags = min(
        pacf_lags,
        maximum_pacf_lags,
    )

    acf_figure, acf_axis = plt.subplots(
        figsize=(13, 5)
    )

    plot_acf(
        clean_load,
        lags=acf_lags,
        ax=acf_axis,
        zero=False,
    )

    acf_axis.set_title(
        "Autocorrelation of Weekly Electricity Demand"
    )

    acf_axis.set_xlabel("Lag in weeks")

    acf_axis.set_ylabel("Autocorrelation")

    acf_figure.tight_layout()

    pacf_figure, pacf_axis = plt.subplots(
        figsize=(13, 5)
    )

    plot_pacf(
        clean_load,
        lags=pacf_lags,
        ax=pacf_axis,
        method="ywm",
        zero=False,
    )

    pacf_axis.set_title(
        "Partial Autocorrelation of Weekly Electricity Demand"
    )

    pacf_axis.set_xlabel("Lag in weeks")

    pacf_axis.set_ylabel(
        "Partial autocorrelation"
    )

    pacf_figure.tight_layout()

    return acf_figure, pacf_figure

def plot_sarima_forecast(
    train: pd.Series,
    test: pd.Series,
    forecast: pd.Series,
    intervals: pd.DataFrame,
    train_weeks_to_show: int = 104,
):
    """
    Plot SARIMA forecasts with 95% prediction intervals.
    """

    aligned_forecast = forecast.reindex(
        test.index
    )

    aligned_intervals = intervals.reindex(
        test.index
    )

    if aligned_forecast.isna().any():
        raise ValueError(
            "SARIMA forecast contains missing values."
        )

    if aligned_intervals.isna().any().any():
        raise ValueError(
            "SARIMA intervals contain missing values."
        )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    train_plot = train.iloc[
        -train_weeks_to_show:
    ]

    ax.plot(
        train_plot.index,
        train_plot.values,
        label="Training load",
        linewidth=1.4,
    )

    ax.plot(
        test.index,
        test.values,
        label="Actual test load",
        linewidth=2,
    )

    ax.plot(
        aligned_forecast.index,
        aligned_forecast.values,
        label="SARIMA forecast",
        linewidth=1.8,
    )

    ax.fill_between(
        aligned_intervals.index,
        aligned_intervals["lower"],
        aligned_intervals["upper"],
        alpha=0.2,
        label="95% prediction interval",
    )

    ax.axvline(
        test.index.min(),
        linestyle="--",
        linewidth=1,
        label="Forecast origin",
    )

    ax.set_title(
        "SARIMA Forecast of Weekly German Electricity Demand"
    )

    ax.set_xlabel("Week")

    ax.set_ylabel(
        "Average electricity demand (GW)"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig


def plot_residual_diagnostics(
    residuals: pd.Series,
):
    """
    Create residual time-series, histogram, Q-Q and ACF plots.
    """

    clean_residuals = pd.Series(
        residuals,
        copy=True,
    ).dropna().astype(float)

    if clean_residuals.empty:
        raise ValueError(
            "Residual series must not be empty."
        )

    residual_figure, residual_axis = plt.subplots(
        figsize=(14, 5)
    )

    residual_axis.plot(
        clean_residuals.index,
        clean_residuals.values,
        linewidth=1.2,
    )

    residual_axis.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    residual_axis.set_title(
        "SARIMA Residuals Over Time"
    )

    residual_axis.set_xlabel("Week")

    residual_axis.set_ylabel("Residual (GW)")

    residual_axis.grid(alpha=0.3)

    residual_figure.tight_layout()

    histogram_figure, histogram_axis = plt.subplots(
        figsize=(9, 5)
    )

    histogram_axis.hist(
        clean_residuals,
        bins=25,
        edgecolor="black",
        alpha=0.75,
    )

    histogram_axis.set_title(
        "Distribution of SARIMA Residuals"
    )

    histogram_axis.set_xlabel("Residual (GW)")

    histogram_axis.set_ylabel("Frequency")

    histogram_figure.tight_layout()

    qq_figure, qq_axis = plt.subplots(
        figsize=(7, 7)
    )

    qqplot(
        clean_residuals,
        line="s",
        ax=qq_axis,
    )

    qq_axis.set_title(
        "Q-Q Plot of SARIMA Residuals"
    )

    qq_figure.tight_layout()

    acf_figure, acf_axis = plt.subplots(
        figsize=(13, 5)
    )

    maximum_lags = min(
        52,
        len(clean_residuals) // 2 - 1,
    )

    plot_acf(
        clean_residuals,
        lags=maximum_lags,
        zero=False,
        ax=acf_axis,
    )

    acf_axis.set_title(
        "Autocorrelation of SARIMA Residuals"
    )

    acf_axis.set_xlabel("Lag in weeks")

    acf_axis.set_ylabel("Autocorrelation")

    acf_figure.tight_layout()

    return {
        "residual_series": residual_figure,
        "residual_histogram": histogram_figure,
        "residual_qq": qq_figure,
        "residual_acf": acf_figure,
    }

def plot_sarimax_forecast(
    train: pd.Series,
    test: pd.Series,
    forecast: pd.Series,
    intervals: pd.DataFrame,
    model_label: str = "SARIMAX",
    train_weeks_to_show: int = 104,
):
    """
    Plot a SARIMAX conditional forecast with prediction intervals.
    """

    aligned_forecast = forecast.reindex(
        test.index
    )

    aligned_intervals = intervals.reindex(
        test.index
    )

    if aligned_forecast.isna().any():
        raise ValueError(
            "SARIMAX forecast contains missing values."
        )

    if aligned_intervals.isna().any().any():
        raise ValueError(
            "SARIMAX prediction intervals contain missing values."
        )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    train_plot = train.iloc[
        -train_weeks_to_show:
    ]

    ax.plot(
        train_plot.index,
        train_plot.values,
        label="Training load",
        linewidth=1.4,
    )

    ax.plot(
        test.index,
        test.values,
        label="Actual test load",
        linewidth=2,
    )

    ax.plot(
        aligned_forecast.index,
        aligned_forecast.values,
        label=f"{model_label} forecast",
        linewidth=1.8,
    )

    ax.fill_between(
        aligned_intervals.index,
        aligned_intervals["lower"],
        aligned_intervals["upper"],
        alpha=0.2,
        label="95% prediction interval",
    )

    ax.axvline(
        test.index.min(),
        linestyle="--",
        linewidth=1,
        label="Forecast origin",
    )

    ax.set_title(
        f"{model_label} Conditional Forecast of "
        "Weekly German Electricity Demand"
    )

    ax.set_xlabel("Week")

    ax.set_ylabel(
        "Average electricity demand (GW)"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig
