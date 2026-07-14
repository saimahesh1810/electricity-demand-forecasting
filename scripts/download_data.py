from pathlib import Path

import pandas as pd
import requests

from electricity_demand.config import (
    BERLIN_LATITUDE,
    BERLIN_LONGITUDE,
    OPEN_METEO_ARCHIVE_URL,
    OPSD_DATA_URL,
    RAW_LOAD_PATH,
    RAW_TEMPERATURE_PATH,
    START_DATE,
)


def download_file(
    url: str,
    destination: Path,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """
    Download a file from a URL and save it locally.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        print(f"File already exists:\n{destination}")
        return destination

    print(f"Downloading:\n{url}")
    print(f"\nSaving to:\n{destination}")

    with requests.get(
        url,
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()

        with destination.open("wb") as file:
            for chunk in response.iter_content(
                chunk_size=chunk_size
            ):
                if chunk:
                    file.write(chunk)

    print("\nDownload complete.")

    return destination


def download_temperature_data(
    destination: Path = RAW_TEMPERATURE_PATH,
) -> Path:
    """
    Download daily historical temperature data for Berlin.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        print(
            f"Temperature file already exists:\n"
            f"{destination}"
        )
        return destination

    params = {
        "latitude": BERLIN_LATITUDE,
        "longitude": BERLIN_LONGITUDE,
        "start_date": START_DATE,
        "end_date": "2020-09-30",
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
        ],
        "timezone": "Europe/Berlin",
    }

    print("\nDownloading Berlin temperature data...")

    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
        timeout=180,
    )

    response.raise_for_status()

    payload = response.json()

    if "daily" not in payload:
        raise ValueError(
            "Open-Meteo response did not contain daily data."
        )

    daily = payload["daily"]

    temperature_data = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temp_mean": daily["temperature_2m_mean"],
            "temp_min": daily["temperature_2m_min"],
            "temp_max": daily["temperature_2m_max"],
        }
    )

    temperature_data.to_csv(
        destination,
        index=False,
    )

    print(
        f"Temperature data saved to:\n"
        f"{destination}"
    )

    return destination


def main() -> None:
    """
    Download electricity-load and temperature data.
    """

    print("Downloading electricity-demand data")

    download_file(
        url=OPSD_DATA_URL,
        destination=RAW_LOAD_PATH,
    )

    print("\nDownloading weather data")

    download_temperature_data()

    print("\nAll raw datasets are available.")


if __name__ == "__main__":
    main()