# Economic value layer

Statistical error metrics (MAE, RMSE, CRPS) say nothing about whether a forecast is *useful* for a specific decision. This layer translates a forecast into a decision and prices that decision in the same units the business already thinks in — currency, not error scores. The literature grounding here is the cost-loss model (Murphy & Richardson, meteorology) and directional forecast value (Blaskowitz & Herwartz, 2009); recent (2025/26) electricity-market forecasting studies find explicitly weak correlation between RMSE/MAE and realized profit, which is the whole motivation for this module.

There are two decision models in `forecastlens.decision`, and they intentionally do not share one interface — their inputs are too different to force into one shape (see `decision.base.DecisionModel` for why that Protocol stays conceptual, not enforced).

## `ProcurementTimingModel`: buy now vs. wait

The strongest business story of the two, because the output is a euro-per-unit figure with no statistics vocabulary needed to explain it.

**How the simulation works.** A single forecast is made once, before the procurement window opens (`forecast.median()` covering `deadline_horizon` periods). The model then walks forward one day at a time:

1. `current_price = realized_prices[t]` — today's own, now-known price.
2. `decision_rule.should_wait(current_price, forecast.median()[t:])` looks only at the *original* forecast's remaining horizon — never at any later day's realized price.
3. If the rule says wait, move to `t + 1`. If it says buy, or the deadline (`t == deadline_horizon - 1`) is reached, purchase at `current_price`.

This is the leak-safety guarantee ROADMAP.md calls out explicitly: the decision at time `t` uses only the forecast that was already available and today's own price, never tomorrow's. `tests/unit/decision/test_procurement_timing.py::test_decision_unaffected_by_realized_prices_after_buy_day_leak_safety` asserts this directly — mutating realized prices after the simulated buy day cannot change the outcome.

**Why a deadline is mandatory, not optional.** Without `deadline_horizon`, the simulated strategy could "wait forever," a strategy no real procurement process would allow and one that would make the model look artificially good in a backtest. `ThresholdTimingRule` can say "wait" every single day; the deadline forces a purchase regardless.

**`ThresholdTimingRule`.** Waits when the *lowest* still-forecast price within the remaining window undercuts today's price by more than `wait_if_forecast_below_current_by` (a fraction). `ExpectedValueTimingRule` — minimizing expected cost over the full quantile distribution rather than a fixed threshold — is deferred to v0.2; it needs the full quantile distribution, not just the median, and the MVP scope stays deliberately narrow.

## `DecisionRelevantBucketAccuracy`: cost-weighted confusion matrix

Classifies both the forecast and the realized value into the same decision-relevant buckets (e.g. "cheap" / "normal" / "expensive"), then scores mismatches with a `CostMatrix` instead of treating every miss as equally costly.

**Bucket edges are causal.** `BucketDefinition.compute_edges` computes rolling quantile edges from the *realized* series' own history, shifted by one period — the edge used to classify period `t` never sees data from `t` or later, the same guarantee `LeakSafeWalkForwardSplitter` provides. `tests/unit/decision/test_buckets.py::test_edges_are_prefix_invariant_leak_safety` asserts this directly: extending the series with more recent data cannot change an earlier period's edges.

**The naive baseline is a one-step persistence forecast** (`realized[t-1]`), the same convention MASE uses for its scale — not an arbitrary choice, but consistency across the codebase's various "what would a naive forecast have cost" comparisons.

**Without a `CostMatrix`, you get a confusion matrix and hit rate, nothing else.** `evaluate()` never fabricates a cost or a value score when you haven't told it what a misclassification actually costs — that would be an implicit "all errors cost the same" assumption ROADMAP.md explicitly rules out.

## `relative_value_score`: read this before citing it

```
relative_value_score = (cost_naive - cost_model) / cost_naive
```

This is a **deliberately simplified MVP metric**, not the academic Murphy value score. The real Murphy score normalizes against a perfect-foresight forecast on a 0–1 scale; ours does not, and can go negative (model worse than naive) or exceed 1 (naive cost near zero). If you're citing a "value score" from this package in a report next to literature that uses the Murphy definition, they are not the same number. A full Murphy-normalized score is planned as an opt-in for v0.2, not part of this MVP.

`cost_naive == 0` raises `ZeroNaiveCostError` rather than returning `inf` or `NaN` — the ratio is genuinely undefined there, not just numerically awkward.
