import numpy as np
import pytest

from forecastlens.core.exceptions import InvalidGeneratorConfigError
from forecastlens.synthetic.demand import DEMAND_PRESETS, DemandGenerator


def test_seasonality_only_hand_computed():
    # demand_t = 100 * (1 + 0.5*sin(2*pi*t/4)) for t=0..3 -> 100, 150, 100, 50
    series = DemandGenerator(
        n_periods=4, baseline=100.0, seasonality_amplitude=0.5, seasonality_period=4, seed=0
    ).generate()
    np.testing.assert_allclose(series.values, [100.0, 150.0, 100.0, 50.0], atol=1e-9)


def test_disruption_hoarding_then_destock_hand_computed():
    series = DemandGenerator(
        n_periods=8,
        baseline=100.0,
        disruption_at=2,
        hoarding_window=2,
        hoarding_multiplier=2.0,
        destock_window=2,
        destock_multiplier=0.5,
        seed=0,
    ).generate()

    np.testing.assert_allclose(
        series.values, [100.0, 100.0, 200.0, 200.0, 50.0, 50.0, 100.0, 100.0]
    )
    np.testing.assert_array_equal(series.regime_labels, [0, 0, 2, 2, 3, 3, 0, 0])
    assert series.changepoints == (2, 4, 6)


def test_intermittency_zeroes_out_demand():
    series = DemandGenerator(
        n_periods=200, baseline=10.0, intermittency_probability=1.0, seed=0
    ).generate()
    np.testing.assert_array_equal(series.values, np.zeros(200))


def test_values_never_negative_even_with_large_noise():
    series = DemandGenerator(n_periods=300, baseline=1.0, noise_std=50.0, seed=0).generate()
    assert np.all(series.values >= 0.0)


def test_promo_regime_matches_deterministic_values():
    baseline = 10.0
    series = DemandGenerator(
        n_periods=200, baseline=baseline, promo_probability=0.2, promo_size_multiplier=5.0, seed=1
    ).generate()

    promo_mask = series.regime_labels == 1
    assert promo_mask.sum() > 0  # sanity: p=0.2 over 200 periods should trigger at least one
    np.testing.assert_allclose(series.values[promo_mask], baseline * 5.0)
    np.testing.assert_allclose(series.values[~promo_mask], baseline)
    assert set(series.metadata["promo_indices"]) == set(np.flatnonzero(promo_mask).tolist())


def test_lifecycle_s_curve_bump_shape():
    series = DemandGenerator(
        n_periods=500,
        baseline=100.0,
        lifecycle="s_curve",
        lifecycle_midpoint=100,
        lifecycle_decline_midpoint=400,
        lifecycle_steepness=0.05,
        seed=0,
    ).generate()

    assert series.changepoints == (100, 400)
    assert series.values[0] < series.values[250]
    assert series.values[-1] < series.values[250]


def test_reproducibility_with_same_seed():
    a = DemandGenerator(n_periods=100, promo_probability=0.1, noise_std=2.0, seed=3).generate()
    b = DemandGenerator(n_periods=100, promo_probability=0.1, noise_std=2.0, seed=3).generate()
    np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"n_periods": 1},
        {"n_periods": 10, "baseline": -1.0},
        {"n_periods": 10, "intermittency_probability": 1.5},
        {"n_periods": 10, "promo_probability": -0.1},
        {"n_periods": 10, "lifecycle": "bogus"},
        {"n_periods": 10, "lifecycle": "ramp_up"},  # missing lifecycle_midpoint
        {
            "n_periods": 10,
            "lifecycle": "s_curve",
            "lifecycle_midpoint": 5,
        },  # missing decline midpoint
        {"n_periods": 10, "disruption_at": 10},
        {"n_periods": 10, "hoarding_window": -1},
        {"n_periods": 10, "destock_window": -1},
        {"n_periods": 10, "noise_std": -1.0},
    ],
)
def test_invalid_construction_raises(bad_kwargs):
    with pytest.raises(InvalidGeneratorConfigError):
        DemandGenerator(**bad_kwargs)


@pytest.mark.parametrize("name", sorted(DEMAND_PRESETS))
def test_all_presets_generate_finite_nonnegative_series(name):
    series = DEMAND_PRESETS[name]()
    assert len(series.values) > 0
    assert np.all(np.isfinite(series.values))
    assert np.all(series.values >= 0.0)
    for cp in series.changepoints:
        assert 0 <= cp < len(series.values)


@pytest.mark.parametrize("name", sorted(DEMAND_PRESETS))
def test_all_presets_reproducible(name):
    a = DEMAND_PRESETS[name]()
    b = DEMAND_PRESETS[name]()
    np.testing.assert_array_equal(a.values, b.values)
    assert a.changepoints == b.changepoints
