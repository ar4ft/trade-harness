"""Bounded offline experiments: tune predictive quality, never agreement for its own sake."""

from .hybrid_training import evaluate_hybrid


def default_trials():
    return [
        {"name": f"{recipe}_c{str(c).replace('.', '_')}_a{alpha}",
         "features": {"recipe": recipe},
         "parameters": {"logistic_c": c, "ridge_alpha": alpha}}
        for recipe in ("base", "interactions", "quant") for c in (0.1, 0.5) for alpha in (10, 100)
    ]


def tune_hybrid(paths, cache="artifacts/hybrid/forecasts.jsonl", stride=48, context=100,
                output="reports/parameter-experiments.json",
                model_output="src/trade_harness/assets/tuned_hybrid_model.json", folds=3, trials=None,
                forecaster=None):
    return evaluate_hybrid(paths, cache, stride, context, output, model_output, folds,
                           forecaster=forecaster, experiment_trials=trials or default_trials())
