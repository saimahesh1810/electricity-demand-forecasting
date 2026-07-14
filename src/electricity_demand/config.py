from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_OBJECT_DIR = OUTPUT_DIR / "model_objects"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_FIGURE_DIR = REPORT_DIR / "figures"


# --------------------------------------------------
# Data source configuration
# --------------------------------------------------

OPSD_DATA_URL = (
    "https://data.open-power-system-data.org/"
    "time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)

RAW_LOAD_FILENAME = "time_series_60min_singleindex.csv"

RAW_LOAD_PATH = RAW_DATA_DIR / RAW_LOAD_FILENAME

PROCESSED_WEEKLY_FILENAME = "weekly_german_electricity_demand.csv"

PROCESSED_WEEKLY_PATH = (
    PROCESSED_DATA_DIR / PROCESSED_WEEKLY_FILENAME
)


# --------------------------------------------------
# Electricity-load configuration
# --------------------------------------------------

TIME_COLUMN = "utc_timestamp"

LOAD_COLUMN = "DE_load_actual_entsoe_transparency"

START_DATE = "2015-01-01"

LOAD_MW_COLUMN = "load_mw"

LOAD_GW_COLUMN = "load_gw"


# --------------------------------------------------
# Forecast configuration
# --------------------------------------------------

TEST_WEEKS = 104

SEASONAL_PERIOD = 52

RANDOM_STATE = 42


# --------------------------------------------------
# Temperature configuration
# --------------------------------------------------

BERLIN_LATITUDE = 52.52

BERLIN_LONGITUDE = 13.41

OPEN_METEO_ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

# --------------------------------------------------
# Covariate file configuration
# --------------------------------------------------

RAW_TEMPERATURE_FILENAME = "berlin_daily_temperature.csv"

RAW_TEMPERATURE_PATH = (
    RAW_DATA_DIR / RAW_TEMPERATURE_FILENAME
)

INTERIM_DAILY_COVARIATES_FILENAME = (
    "daily_weather_and_holidays.csv"
)

INTERIM_DAILY_COVARIATES_PATH = (
    INTERIM_DATA_DIR
    / INTERIM_DAILY_COVARIATES_FILENAME
)


# --------------------------------------------------
# Temperature-derived feature configuration
# --------------------------------------------------

HEATING_BASE_TEMPERATURE = 18.0

COOLING_BASE_TEMPERATURE = 22.0


# --------------------------------------------------
# Ensure output directories exist
# --------------------------------------------------

for directory in [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    FORECAST_DIR,
    METRICS_DIR,
    MODEL_OBJECT_DIR,
    REPORT_FIGURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)