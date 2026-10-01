# Feature and parameter experiment results

Eight predeclared candidates used 1,455 frozen TimesFM forecast snapshots across BTCUSDT, ETHUSDT and SOLUSDT, three purged chronological validation folds, separate calibration, and normal/doubled-cost replay. Parameter selection used validation classification log loss; the final period did not select the winner.

The selected candidate was `base_c0_1_a10`: original 35 features, logistic C=0.1 and Ridge alpha=10. Its validation log loss was 1.085615, compared with 1.086299 for the untuned base. Adding four bounded causal interactions did not improve log loss. The winner's mean validation return was −0.165%, with only 10 closed trades. Every candidate failed the positive-edge gate.

On the locked final period (291 snapshots), winner direction accuracy was 45.70% and log loss 1.065478, versus 42.61% and 1.068789 for the untuned base. Neither candidate placed a trade in that period under the configured risk thresholds. These results establish neither profitability nor an execution-ready model.

The experimental artifact is separate from the default model and remains research-only. Full parameters, provenance, fold results and limitations are in [parameter-experiments.json](parameter-experiments.json).

## Language reviewer audit

The real local LoRA reviewer was evaluated on 24 BTC snapshots later than both known training cutoffs, using the same frozen forecasting and feature evidence as the numerical decision model. Agreement was 1/24 (4.17%). Numerical direction accuracy was 62.5%; reviewer accuracy was 25%. The consensus policy returned HOLD for every snapshot, also scoring 62.5% because many labels were HOLD.

This small direction audit has no cost replay and is not a trading-edge validation. It shows no benefit from adding the current reviewer. The shipped adapter was trained without this extended evidence prompt; contextual reviewed examples and fresh prospective evaluation are needed before promoting an ensemble. We retain the fixed agreement requirement rather than tuning it to produce more trades. Full observations are in [consensus-comparison.json](consensus-comparison.json).
