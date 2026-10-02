# Contextual reviewer and complete-policy results

Version 0.8 implements shared-evidence training, full-policy replay, causal feature/regime diagnostics, and stronger forward acceptance. These historical experiments establish no positive trading edge. No candidate is activated.

## Contextual reviewer

The new candidate uses BTC, ETH, and SOL with paired flat/long inputs, position-aware next-open cost labels, and exactly the runtime `shared-evidence-v2` renderer. Its training partition contains 1,452 examples: 206 BUY, 180 SELL, and 1,066 HOLD. The CPU run used 120 optimizer steps, processing 480 examples, approximately 0.33 epoch. This is a limited training budget.

Separate calibration on 48 later examples selected temperature 1.70894 and reduced calibration log loss from 0.90001 to 0.80574. The sampled validation and test sets each contain 48 examples. The reviewer predicted HOLD for every example in both: action accuracy was 77.08% and 72.92%, exactly matching the majority baseline. Directional coverage is zero. Its probability scores also fail to beat the constant training-prior log-loss baseline on these samples. Calibration improvement on its fitting sample does not establish generalization.

The adapter remains a local experimental artifact rather than the default reviewer. Full training metadata and baseline calculations are in [reviewer-v2-training.json](reviewer-v2-training.json).

## Complete decision policy

The comparison uses 291 validation snapshots across five chronological folds and 291 final-period snapshots, with an identical sparse schedule for every candidate. Numerical weights/calibration are refitted before each fold; the contextual reviewer is frozen. Every intervening bar updates the paper accounts' fills, stops, exits, and risk. Each policy is rerun at normal and doubled costs, with separate funded accounts for each asset.

| Policy | Validation closed trades | Final-action coverage | Compounded diagnostic validation return |
|---|---:|---:|---:|
| Numerical anchor | 4 | 1.37% | −0.0535% |
| Fixed momentum | 60 | 20.62% | −1.2098% |
| Fixed strategy rules | 28 | 9.62% | −0.2018% |
| Numerical plus contextual reviewer | 0 | 0% | 0% |

The return column compounds aligned daily returns averaged across the independently funded asset accounts and reset fold accounts. It is a diagnostic aggregation, not the measured return of a shared portfolio. Per-asset account returns, stress runs, decision-band/regime summaries, interval estimates, ledger paths/hashes, and final-period comparisons are in [research-evaluation.json](research-evaluation.json). Full JSONL ledgers are generated under ignored local artifacts and can be reproduced with the [workflow](../docs/research-workflow.md).

Consensus preserved cash by abstaining, but did not beat the cash baseline or demonstrate useful directional review. None of the four policies meets the acceptance criteria. The final period cannot select a policy or promote a historical result.

## Feature search

Twelve predeclared base/interaction/quant regularization trials used five purged validation folds. Base C=0.1/alpha=10 had the lowest validation log loss, 1.078704. Quant features did not improve this metric. The winner averaged −0.0713% account return across validation folds/assets with six closed trades; every candidate failed promotion. Paired fold differences are retained so a small score advantage is not mistaken for a stable parameter region. See [quant-experiments.json](quant-experiments.json).

## Acceptance and next evidence

`research-promotion-v2` requires a locked future experiment completed through its fixed endpoint, prediction-before-outcome records, sufficiently complete observations and matured labels, at least five completed windows and 100 closed trades, positive normal/stressed outcomes, positive paired time-block lower bounds against cash and momentum, per-asset checks, and sufficient directional calibration evidence across bins and windows. The current horizon requires 140 days of new observations. Legacy historical measurements cannot override these requirements.

The candidate's current all-HOLD behavior is a rejected result. Further feature/label/training changes should be separate declared experiments, judged against simple baselines, followed by a fresh locked forward period. Agreement targets or relaxed gates cannot substitute for demonstrated edge.
