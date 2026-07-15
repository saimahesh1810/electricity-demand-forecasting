import numpy as np
from sklearn.preprocessing import StandardScaler

from electricity_demand.models.neural import (
    inverse_target_scale,
    make_lstm_sequences,
)


def test_make_lstm_sequences_shapes():
    values = np.arange(
        60,
        dtype=float,
    ).reshape(
        20,
        3,
    )

    X, y = make_lstm_sequences(
        values,
        lookback=5,
    )

    assert X.shape == (
        15,
        5,
        3,
    )

    assert y.shape == (
        15,
    )


def test_sequence_target_is_next_value():
    values = np.column_stack(
        [
            np.arange(
                10,
                dtype=float,
            ),
            np.ones(10),
        ]
    )

    X, y = make_lstm_sequences(
        values,
        lookback=3,
    )

    assert X[0, -1, 0] == 2.0

    assert y[0] == 3.0


def test_inverse_target_scale():
    values = np.array(
        [
            [10.0, 1.0],
            [20.0, 2.0],
            [30.0, 3.0],
        ]
    )

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        values
    )

    restored = inverse_target_scale(
        scaled_target=scaled[:, 0],
        scaler=scaler,
        number_of_features=2,
    )

    assert np.allclose(
        restored,
        values[:, 0],
    )