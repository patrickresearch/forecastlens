import numpy as np
import pytest
from scipy import stats

from forecastlens.core.exceptions import InvalidForecastResultError
from forecastlens.metrics.crps import crps_from_quantiles, crps_from_samples


def test_crps_from_samples_hand_computed():
    # y=0, samples=[-1, 0, 1]
    # term_data = mean(|s - 0|) = (1+0+1)/3 = 2/3
    # pairwise |si-sj| matrix sums to 8 -> term_spread = 8 / (2*9) = 4/9
    # crps = 2/3 - 4/9 = 2/9
    result = crps_from_samples([0.0], [[-1.0], [0.0], [1.0]])
    np.testing.assert_allclose(result, [2.0 / 9.0])


def test_crps_from_samples_zero_for_deterministic_ensemble():
    # All ensemble members identical to y_true -> both terms cancel exactly.
    result = crps_from_samples([5.0], [[5.0], [5.0], [5.0]])
    np.testing.assert_allclose(result, [0.0], atol=1e-12)


def test_crps_from_samples_shape_mismatch_raises():
    with pytest.raises(InvalidForecastResultError):
        crps_from_samples([0.0, 1.0], [[0.0], [1.0]])


def test_crps_from_quantiles_needs_at_least_two_levels():
    with pytest.raises(InvalidForecastResultError):
        crps_from_quantiles([0.0], {0.5: [0.0]})


def test_crps_from_quantiles_matches_analytic_normal_crps():
    # CRPS(N(mu, sigma), y) has a closed form (Gneiting & Raftery, 2007):
    # sigma * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ],  z = (y - mu) / sigma
    mu, sigma, y = 0.0, 1.0, 0.0
    z = (y - mu) / sigma
    analytic = sigma * (
        z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    )

    levels = np.linspace(0.001, 0.999, 999)
    quantiles = {float(level): [stats.norm.ppf(level, loc=mu, scale=sigma)] for level in levels}

    approx = crps_from_quantiles([y], quantiles)[0]
    assert approx == pytest.approx(analytic, abs=1e-3)


def test_crps_from_samples_matches_properscoring_reference():
    properscoring = pytest.importorskip("properscoring")
    rng = np.random.default_rng(42)
    y_true = rng.normal(size=5)
    samples = rng.normal(size=(200, 5))

    ours = crps_from_samples(y_true, samples)
    theirs = np.array(
        [properscoring.crps_ensemble(y_true[i], samples[:, i]) for i in range(5)]
    )
    np.testing.assert_allclose(ours, theirs, rtol=1e-8)
