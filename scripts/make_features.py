from electricity_demand.config import (
    PROCESSED_WEEKLY_PATH,
)
from electricity_demand.data import (
    make_processed_weekly_data,
)


def main() -> None:
    """
    Create the processed weekly electricity-demand dataset.
    """

    weekly_data = make_processed_weekly_data()

    print("\nProcessed weekly dataset created successfully.")

    print(f"\nSaved to:\n{PROCESSED_WEEKLY_PATH}")

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
    print(weekly_data.describe().round(3))


if __name__ == "__main__":
    main()