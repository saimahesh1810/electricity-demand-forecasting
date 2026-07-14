from pathlib import Path
import holidays
import numpy as np
import pandas as pd

from electricity_demand.config import (
    COOLING_BASE_TEMPERATURE,
    HEATING_BASE_TEMPERATURE,
    INTERIM_DAILY_COVARIATES_PATH,
    LOAD_COLUMN,
    LOAD_GW_COLUMN,
    LOAD_MW_COLUMN,
    PROCESSED_WEEKLY_PATH,
    RAW_LOAD_PATH,
    RAW_TEMPERATURE_PATH,
    START_DATE,
    TIME_COLUMN,
)


def load_raw_hourly_data(
    file_path: Path = RAW_LOAD_PATH,
) -> pd.DataFrame:
    """
    Load the German hourly electricity-demand series.

    Parameters
    ----------
    file_path:
        Path to the OPSD hourly CSV file.

    Returns
    -------
    pandas.DataFrame
        Hourly electricity-demand data indexed by timestamp.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file was not found:\n{file_path}\n\n"
            "Run this command first:\n"
            "python scripts/download_data.py"
        )

    use_columns = [
        TIME_COLUMN,
        LOAD_COLUMN,
    ]

    data = pd.read_csv(
        file_path,
        usecols=use_columns,
        parse_dates=[TIME_COLUMN],
    )

    data = data.rename(
        columns={
            LOAD_COLUMN: LOAD_MW_COLUMN,
        }
    )

    data = data.set_index(TIME_COLUMN)

    # Convert the OPSD UTC index to timezone-naive UTC timestamps.
    # This allows it to align with the daily weather index.
    if data.index.tz is not None:
      data.index = data.index.tz_convert("UTC").tz_localize(None)

    data = data.sort_index()

    return data


def clean_hourly_data(
    data: pd.DataFrame,
    start_date: str = START_DATE,
) -> pd.DataFrame:
    """
    Clean the hourly German electricity-demand series.

    Cleaning steps:
    1. Keep observations from the chosen start date.
    2. Remove duplicate timestamps.
    3. Convert load values to numeric.
    4. Remove missing target observations.
    5. Convert electricity load from MW to GW.

    Parameters
    ----------
    data:
        Raw hourly electricity-demand data.

    start_date:
        First date retained in the analysis.

    Returns
    -------
    pandas.DataFrame
        Clean hourly data with load in MW and GW.
    """

    cleaned = data.copy()

    cleaned = cleaned.loc[start_date:]

    cleaned = cleaned[
        ~cleaned.index.duplicated(keep="first")
    ]

    cleaned[LOAD_MW_COLUMN] = pd.to_numeric(
        cleaned[LOAD_MW_COLUMN],
        errors="coerce",
    )

    cleaned = cleaned.dropna(
        subset=[LOAD_MW_COLUMN]
    )

    cleaned[LOAD_GW_COLUMN] = (
        cleaned[LOAD_MW_COLUMN] / 1000
    )

    return cleaned


def aggregate_to_weekly(
    hourly_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate hourly electricity load to complete weekly averages.

    Weeks are created as consecutive seven-day periods beginning from
    the first available timestamp. Incomplete final weeks are removed.

    Parameters
    ----------
    hourly_data:
        Clean hourly electricity-demand data.

    Returns
    -------
    pandas.DataFrame
        Weekly average electricity demand in GW.
    """

    weekly_load = hourly_data[
        [LOAD_GW_COLUMN]
    ].resample(
        "168h",
        origin=hourly_data.index.min(),
        label="left",
        closed="left",
    ).mean()

    hourly_counts = hourly_data[
        LOAD_GW_COLUMN
    ].resample(
        "168h",
        origin=hourly_data.index.min(),
        label="left",
        closed="left",
    ).count()

    complete_week_mask = hourly_counts == 168

    weekly_load = weekly_load.loc[
        complete_week_mask
    ]

    weekly_load.index.name = "timestamp"

    return weekly_load

def load_daily_temperature(
    file_path: Path = RAW_TEMPERATURE_PATH,
) -> pd.DataFrame:
    """
    Load daily Berlin temperature data.

    Returns
    -------
    pandas.DataFrame
        Daily temperature data indexed by date.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Temperature file was not found:\n"
            f"{file_path}\n\n"
            "Run this command first:\n"
            "python scripts/download_data.py"
        )

    temperature = pd.read_csv(
        file_path,
        parse_dates=["date"],
    )

    temperature = temperature.set_index("date")

    temperature = temperature.sort_index()

    temperature = temperature[
        ~temperature.index.duplicated(keep="first")
    ]

    for column in [
        "temp_mean",
        "temp_min",
        "temp_max",
    ]:
        temperature[column] = pd.to_numeric(
            temperature[column],
            errors="coerce",
        )

    temperature = temperature.interpolate(
        method="time",
        limit_direction="both",
    )

    return temperature


def create_daily_temperature_features(
    temperature: pd.DataFrame,
    heating_base: float = HEATING_BASE_TEMPERATURE,
    cooling_base: float = COOLING_BASE_TEMPERATURE,
) -> pd.DataFrame:
    """
    Create heating-degree-day and cooling-degree-day variables.

    Heating degree days measure how far daily mean temperature falls
    below the heating base temperature.

    Cooling degree days measure how far daily mean temperature rises
    above the cooling base temperature.
    """

    features = temperature.copy()

    features["heating_degree_days"] = np.maximum(
        heating_base - features["temp_mean"],
        0,
    )

    features["cooling_degree_days"] = np.maximum(
        features["temp_mean"] - cooling_base,
        0,
    )

    return features


def create_daily_holiday_features(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Create German national public-holiday indicators.
    """

    date_index = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )

    years = sorted(date_index.year.unique())

    german_holidays = holidays.country_holidays(
        country="DE",
        years=years,
    )

    holiday_features = pd.DataFrame(
        index=date_index
    )

    holiday_features.index.name = "date"

    holiday_features["has_holiday"] = [
        int(date.date() in german_holidays)
        for date in holiday_features.index
    ]

    holiday_features["holiday_name"] = [
        german_holidays.get(date.date())
        for date in holiday_features.index
    ]

    return holiday_features


def create_daily_covariates(
    temperature: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine daily temperature and holiday variables.
    """

    temperature_features = (
        create_daily_temperature_features(
            temperature
        )
    )

    holiday_features = (
        create_daily_holiday_features(
            start_date=str(
                temperature_features.index.min().date()
            ),
            end_date=str(
                temperature_features.index.max().date()
            ),
        )
    )

    daily_covariates = temperature_features.join(
        holiday_features,
        how="left",
    )

    daily_covariates["has_holiday"] = (
        daily_covariates["has_holiday"]
        .fillna(0)
        .astype(int)
    )

    return daily_covariates


def aggregate_covariates_to_weekly(
    daily_covariates: pd.DataFrame,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """
    Aggregate daily weather and holiday variables to the same
    consecutive seven-day periods as the electricity-load data.
    """

    covariates = daily_covariates.copy()

    # Ensure the daily weather index is timezone-naive.
    if covariates.index.tz is not None:
        covariates.index = covariates.index.tz_localize(None)

    origin = pd.Timestamp(origin)

    if origin.tzinfo is not None:
        origin = origin.tz_convert("UTC").tz_localize(None)

    weekly_means = covariates[
        [
            "temp_mean",
            "temp_min",
            "temp_max",
            "heating_degree_days",
            "cooling_degree_days",
        ]
    ].resample(
        "168h",
        origin=origin,
        label="left",
        closed="left",
    ).mean()

    weekly_holidays = covariates[
        "has_holiday"
    ].resample(
        "168h",
        origin=origin,
        label="left",
        closed="left",
    ).sum()

    daily_counts = covariates[
        "temp_mean"
    ].resample(
        "168h",
        origin=origin,
        label="left",
        closed="left",
    ).count()

    complete_week_mask = daily_counts == 7

    weekly_covariates = weekly_means.loc[
        complete_week_mask
    ].copy()

    weekly_covariates["holiday_days"] = (
        weekly_holidays.loc[complete_week_mask]
        .astype(int)
    )

    weekly_covariates["has_holiday"] = (
        weekly_covariates["holiday_days"] > 0
    ).astype(int)

    weekly_covariates.index.name = "timestamp"

    return weekly_covariates

def make_processed_weekly_data(
    raw_path: Path = RAW_LOAD_PATH,
    output_path: Path = PROCESSED_WEEKLY_PATH,
) -> pd.DataFrame:
    """
    Create the complete weekly modelling dataset.

    The resulting dataset contains weekly average German electricity
    demand, Berlin temperature variables, degree-day variables, and
    German national holiday indicators.
    """

    raw_data = load_raw_hourly_data(
        file_path=raw_path
    )

    cleaned_hourly = clean_hourly_data(
        data=raw_data
    )

    weekly_load = aggregate_to_weekly(
        hourly_data=cleaned_hourly
    )

    daily_temperature = load_daily_temperature()

    daily_covariates = create_daily_covariates(
        temperature=daily_temperature
    )

    INTERIM_DAILY_COVARIATES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_covariates.to_csv(
        INTERIM_DAILY_COVARIATES_PATH
    )

    weekly_covariates = aggregate_covariates_to_weekly(
        daily_covariates=daily_covariates,
        origin=weekly_load.index.min(),
    )

    weekly_data = weekly_load.join(
        weekly_covariates,
        how="left",
    )

    required_columns = [
        LOAD_GW_COLUMN,
        "temp_mean",
        "temp_min",
        "temp_max",
        "heating_degree_days",
        "cooling_degree_days",
        "holiday_days",
        "has_holiday",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in weekly_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Processed dataset is missing columns: "
            f"{missing_columns}"
        )

    if weekly_data[required_columns].isna().any().any():
        missing_counts = (
            weekly_data[required_columns]
            .isna()
            .sum()
        )

        raise ValueError(
            "Processed weekly data contains missing "
            f"values:\n{missing_counts}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly_data.to_csv(output_path)

    return weekly_data

def load_processed_data(
    file_path: Path = PROCESSED_WEEKLY_PATH,
) -> pd.DataFrame:
    """
    Load the processed weekly modelling dataset.

    Parameters
    ----------
    file_path:
        Path to the processed weekly CSV file.

    Returns
    -------
    pandas.DataFrame
        Weekly modelling data indexed by timestamp.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed data file was not found:\n{file_path}\n\n"
            "Run this command first:\n"
            "python scripts/make_features.py"
        )

    data = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
        index_col="timestamp",
    )

    data = data.sort_index()

    return data