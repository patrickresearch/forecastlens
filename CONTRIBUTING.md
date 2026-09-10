# Contributing

ForecastLens is pre-alpha; interfaces can still change between commits. Issues and PRs are welcome regardless.

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Development principles (non-negotiable — see `CLAUDE.md` for full rationale)

- **Fail-loud, never fail-silent.** Missing quantiles, unrecognized adapter formats, missing cost matrices → raise a clear exception, never return silent NaNs or defaults.
- **Leak-safety is mandatory** for anything time-dependent (`LeakSafeWalkForwardSplitter`, rolling bucket edges, `ProcurementTimingModel` decisions): a computation at time `t` may only use data up to `t-1`. Every such component needs an explicit leak test.
- **Metric correctness before feature breadth.** CRPS/MASE/WQL/Pinball implementations must be validated against a reference implementation (`properscoring`, `sktime`/`darts.metrics`) before new features are built on top.
- Core package stays dependency-light: only `numpy`, `pandas`, `scipy` as hard dependencies. `darts`/`neuralforecast` are optional extras, never hard dependencies.

## Tests

```bash
pytest tests/unit          # fast, no heavy framework dependencies
pytest tests/integration   # requires darts/neuralforecast installed
```

## Code style

- `ruff` for linting/formatting, `mypy` for type checking — both run via pre-commit.
- Comments in English, WHY not WHAT — see `CLAUDE.md` for the exact style guidance.
