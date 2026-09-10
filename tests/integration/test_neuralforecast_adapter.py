import numpy as np
import pandas as pd
import pytest

neuralforecast = pytest.importorskip("neuralforecast")

from neuralforecast import NeuralForecast  # noqa: E402
from neuralforecast.losses.pytorch import MQLoss  # noqa: E402
from neuralforecast.models import NHITS  # noqa: E402

from forecastlens.adapters.neuralforecast import NeuralForecastAdapter  # noqa: E402
from forecastlens.core.exceptions import UnrecognizedAdapterFormatError  # noqa: E402


def _training_df(unique_id="series_1", n=60):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    values = np.sin(np.arange(n) / 5.0) * 10 + 100
    return pd.DataFrame({"unique_id": unique_id, "ds": dates, "y": values})


def test_point_forecast_becomes_point_forecast():
    df = _training_df()
    model = NHITS(h=5, input_size=20, max_steps=5, enable_progress_bar=False)
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=df)
    predictions = nf.predict()

    result = NeuralForecastAdapter.from_dataframe(predictions, model=nf.models[0])

    assert not result.has_quantiles
    assert result.point is not None
    assert result.horizon == 5
    assert result.metadata["source"] == "neuralforecast"
    assert result.metadata["model"] == "NHITS"
    assert result.metadata["unique_id"] == "series_1"


def test_quantile_forecast_becomes_quantiles():
    df = _training_df()
    model = NHITS(
        h=5,
        input_size=20,
        max_steps=5,
        enable_progress_bar=False,
        loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=df)
    predictions = nf.predict()

    result = NeuralForecastAdapter.from_dataframe(predictions, model=nf.models[0])

    assert result.has_quantiles
    assert result.quantile_levels() == [0.1, 0.5, 0.9]


def test_multiple_series_requires_explicit_unique_id():
    df = pd.concat([_training_df("series_1"), _training_df("series_2")], ignore_index=True)
    model = NHITS(h=5, input_size=20, max_steps=5, enable_progress_bar=False)
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=df)
    predictions = nf.predict()

    with pytest.raises(UnrecognizedAdapterFormatError):
        NeuralForecastAdapter.from_dataframe(predictions, model=nf.models[0])

    result = NeuralForecastAdapter.from_dataframe(
        predictions, model=nf.models[0], unique_id="series_2"
    )
    assert result.metadata["unique_id"] == "series_2"
