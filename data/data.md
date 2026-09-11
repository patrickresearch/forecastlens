# Data Sources

Real-world data is never bundled in this repository (see `data/real/scripts/` for download scripts) or hardcoded in the package. Every real dataset used in examples/tutorials must be listed here with source, license, and retrieval date before it is referenced anywhere in docs or examples.

Synthetic data (`forecastlens.synthetic`) needs no license entry here — its generation parameters/seeds are documented alongside the generator code instead.

## Real datasets

| Dataset | Source | License / Terms | Attribution required | Retrieved | Notes |
|---|---|---|---|---|---|
| _(none yet)_ | | | | | Planned: Brent/WTI via FRED (`DCOILBRENTEU`, `DCOILWTICO`), Henry Hub (`DHHNGSP`) + AGSI+ storage as a v0.2 extension — pending API key registration. |

## Explicitly excluded sources

- **Yahoo Finance / `yfinance`** — unofficial API, violates Yahoo's ToS. Not used.
- **TTF gas prices** — no freely redistributable daily series available (ICE Endex is proprietary; AGSI+/ALSI+ only provide storage levels, not prices).
