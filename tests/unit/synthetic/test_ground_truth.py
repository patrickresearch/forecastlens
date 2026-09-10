import pytest

from forecastlens.core.exceptions import ForecastLensError
from forecastlens.synthetic.demand import DEMAND_PRESETS
from forecastlens.synthetic.ground_truth import list_presets, load_preset
from forecastlens.synthetic.price import PRICE_PRESETS


def test_list_presets_covers_all_ten():
    names = list_presets()
    assert len(names) == 10
    assert set(names) == set(PRICE_PRESETS) | set(DEMAND_PRESETS)


@pytest.mark.parametrize("name", list_presets())
def test_load_preset_matches_direct_call(name):
    via_registry = load_preset(name)
    direct = (PRICE_PRESETS | DEMAND_PRESETS)[name]()
    import numpy as np

    np.testing.assert_array_equal(via_registry.values, direct.values)


def test_load_unknown_preset_raises():
    with pytest.raises(ForecastLensError):
        load_preset("does_not_exist")
