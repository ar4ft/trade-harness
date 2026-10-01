# Initial hybrid comparison

Retrospective CPU research on 1,455 historical snapshots (485 per asset) from BTCUSDT, ETHUSDT, and SOLUSDT. Each snapshot contains 100 hourly candles; entry decisions are sampled every 48 candles. TimesFM 3.0 is frozen. Three purged validation folds precede the final 20% test period. This history has been examined before and pretraining overlap is unknown.

| Candidate | Mean validation log loss ↓ | Mean net account return | Doubled-cost return | Closed validation trades |
|---|---:|---:|---:|---:|
| market_only | 1.0868 | -0.072% | -0.078% | 8 |
| market_boosted_reference | 1.0705 | -0.058% | -0.040% | 4 |
| strategy_only | 1.0891 | 0.000% | 0.000% | 0 |
| market_strategy | 1.0852 | -0.122% | -0.101% | 8 |
| forecast_only | 1.0811 | 0.000% | 0.000% | 0 |
| combined | 1.0863 | -0.165% | -0.101% | 10 |

Forecast-only wins log loss among the five logistic ablations. The separately reported boosted market-only reference has lower log loss than any of those ablations. The combined model does not demonstrate an improvement: its mean validation account return is −0.165%, and only ten validation trades close. Every candidate fails the positive-edge gates.

The combined artifact is available through `--backend hybrid` as an opt-in research model; it does not replace the default model. In the final test it has 42.61% direction accuracy and no executed trades, compared with 46.05% direction accuracy for matched market-only logistic regression. The final period did not select candidates or change settings.

A zero return from no trades is abstention, not evidence of a profitable strategy. Mean account returns average independently funded asset accounts and then folds; they are not a shared-portfolio or compounded lifetime return. Doubled costs can alter which entries pass risk checks, so account returns need not decrease monotonically.

The result supports more testing, not promotion. Next experiments should predeclare denser decision sampling and evaluate fresh prospective periods, then compare Kronos/Chronos-2. Keep strategy thresholds and learner settings fixed for each recorded experiment. Check predictive calibration and sample/trade counts as well as account returns.

[Full measured report](hybrid-comparison.json) · [Architecture and reproduction](../docs/hybrid-decisions.md)
