import numpy as np
import pandas as pd

from electricity_demand.features import (
    make_ml_table,
)


def make_example_data() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-02",
        periods=70,
        freq="W-THU",
    )

    return pd.DataFrame(
        {
            "load_gw": np.arange(
                1,
                71,
                dtype=float,
            ),
            "temp_mean": np.linspace(
                2,
                20,
                70,
            ),
            "holiday_days": np.zeros(
                70
            ),
            "has_holiday": np.zeros(
                70
            ),
        },
        index=index,
    )


def test_lag_feature_uses_previous_value():
    data = make_example_data()

    table = make_ml_table(
        data,
        lag_weeks=(1, 2),
        rolling_windows=(4,),
        covariate_columns=(
            "temp_mean",
            "holiday_days",
            "has_holiday",
        ),
    )

    timestamp = table.index[0]

    original_position = data.index.get_loc(
        timestamp
    )

    expected_previous_value = data[
        "load_gw"
    ].iloc[
        original_position - 1
    ]

    assert (
        table.loc[
            timestamp,
            "lag_1",
        ]
        == expected_previous_value
    )


def test_rolling_mean_excludes_current_target():
    data = make_example_data()

    table = make_ml_table(
        data,
        lag_weeks=(1,),
        rolling_windows=(4,),
        covariate_columns=(
            "temp_mean",
        ),
    )

    timestamp = table.index[0]

    position = data.index.get_loc(
        timestamp
    )

    expected_mean = data[
        "load_gw"
    ].iloc[
        position - 4:
        position
    ].mean()

    assert np.isclose(
        table.loc[
            timestamp,
            "rolling_mean_4",
        ],
        expected_mean,
    )


def test_current_target_not_used_in_rolling_feature():
    data = make_example_data()

    original_table = make_ml_table(
        data,
        lag_weeks=(1,),
        rolling_windows=(4,),
        covariate_columns=(
            "temp_mean",
        ),
    )

    timestamp = original_table.index[
        0
    ]

    modified_data = data.copy()

    modified_data.loc[
        timestamp,
        "load_gw",
    ] = 999999.0

    modified_table = make_ml_table(
        modified_data,
        lag_weeks=(1,),
        rolling_windows=(4,),
        covariate_columns=(
            "temp_mean",
        ),
    )

    assert np.isclose(
        original_table.loc[
            timestamp,
            "rolling_mean_4",
        ],
        modified_table.loc[
            timestamp,
            "rolling_mean_4",
        ],
    )


def test_ml_table_has_no_missing_values():
    data = make_example_data()

    table = make_ml_table(
        data
    )

    assert not table.isna().any().any()