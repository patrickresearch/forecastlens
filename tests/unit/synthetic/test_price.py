import numpy as np
import pytest

from forecastlens.core.exceptions import InvalidGeneratorConfigError
from forecastlens.synthetic.price import PRICE_PRESETS, PriceRegimeGenerator


def test_deterministic_structural_break_hand_computed():
    # No noise, single regime -> the recursion is exactly hand-computable.
    # values[0]=100; before t=2, target=100 (no-op); from t=2, target=110.
    # t=2: 100 + 0.5*(110-100) = 105
    # t=3: 105 + 0.5*(110-105) = 107.5
    # t=4: 107.5 + 0.5*(110-107.5) = 108.75
    series = PriceRegimeGenerator(
        n_periods=5,
        base_level=100.0,
        mean_reversion_speed=0.5,
        volatility=0.0,
        structural_break_at=2,
        structural_break_shift=10.0,
        seed=0,
    ).generate()

    np.testing.assert_allclose(series.values, [100.0, 100.0, 105.0, 107.5, 108.75])
    assert series.changepoints == (2,)
    np.testing.assert_array_equal(series.regime_labels, [0, 0, 0, 0, 0])


def test_deterministic_alternating_regimes():
    # Degenerate transition matrix (0/1 entries) makes the regime path deterministic
    # regardless of RNG draws.
    transition = np.array([[0.0, 1.0], [1.0, 0.0]])
    series = PriceRegimeGenerator(
        n_periods=5,
        base_level=100.0,
        volatility=0.0,
        regime_transition_matrix=transition,
        seed=0,
    ).generate()

    np.testing.assert_array_equal(series.regime_labels, [0, 1, 0, 1, 0])
    assert series.changepoints == (1, 2, 3, 4)


def test_no_seasonality_or_regime_is_flat_at_equilibrium():
    series = PriceRegimeGenerator(n_periods=10, base_level=50.0, volatility=0.0, seed=0).generate()
    np.testing.assert_allclose(series.values, [50.0] * 10)
    assert series.changepoints == ()


def test_reproducibility_with_same_seed():
    a = PriceRegimeGenerator(n_periods=50, volatility=2.0, jump_probability=0.05, seed=7).generate()
    b = PriceRegimeGenerator(n_periods=50, volatility=2.0, jump_probability=0.05, seed=7).generate()
    np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"n_periods": 1},
        {"n_periods": 10, "mean_reversion_speed": 0.0},
        {"n_periods": 10, "mean_reversion_speed": 1.5},
        {"n_periods": 10, "volatility": -1.0},
        {"n_periods": 10, "jump_probability": 1.5},
        {"n_periods": 10, "vol_cluster_persistence": 1.0},
        {"n_periods": 10, "structural_break_at": 0},
        {"n_periods": 10, "structural_break_at": 10},
        {"n_periods": 10, "regime_transition_matrix": np.array([[0.5, 0.6], [0.5, 0.5]])},
        {
            "n_periods": 10,
            "regime_transition_matrix": np.array([[0.5, 0.5], [0.5, 0.5]]),
            "regime_level_multipliers": [1.0, 1.0, 1.0],
        },
    ],
)
def test_invalid_construction_raises(bad_kwargs):
    with pytest.raises(InvalidGeneratorConfigError):
        PriceRegimeGenerator(**bad_kwargs)


@pytest.mark.parametrize("name", sorted(PRICE_PRESETS))
def test_all_presets_generate_finite_series(name):
    series = PRICE_PRESETS[name]()
    assert len(series.values) > 0
    assert np.all(np.isfinite(series.values))
    for cp in series.changepoints:
        assert 0 <= cp < len(series.values)


@pytest.mark.parametrize("name", sorted(PRICE_PRESETS))
def test_all_presets_reproducible(name):
    a = PRICE_PRESETS[name]()
    b = PRICE_PRESETS[name]()
    np.testing.assert_array_equal(a.values, b.values)
    assert a.changepoints == b.changepoints
