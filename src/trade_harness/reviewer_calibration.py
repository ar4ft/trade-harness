"""Measured reviewer cohorts and identity-bound, past-only temperature calibration."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import log_loss

from .diagnostics import probability_diagnostics
from .hybrid_training import digest
from .research_data import load_research
from .schemas import MarketInput, Proposal

ACTIONS = ("BUY", "SELL", "HOLD")


def scaled(probabilities, temperature):
    values = np.asarray(probabilities, dtype=float)
    if (
        values.shape[-1] != 3
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(axis=-1), 1, atol=1e-5)
        or not np.isfinite(temperature)
        or not 0.25 <= temperature <= 10
    ):
        raise ValueError("Invalid calibration probability or temperature")
    return softmax(np.log(np.maximum(values, 1e-12)) / temperature, axis=-1)


def score_reviewer(
    dataset,
    backend,
    output,
    max_samples=256,
    phases=("calibration", "validation", "test"),
    provider=None,
):
    from .models import load_model

    if (
        not 1 <= max_samples <= 10000
        or not phases
        or not set(phases) <= {"calibration", "validation", "test"}
    ):
        raise ValueError("Declare held-out phases and 1–10000 examples per phase")
    rows, source = load_research(dataset)
    provider = provider or load_model(backend)
    selected = []
    for phase in phases:
        group = [r for r in rows if r["phase"] == phase]
        times = sorted({r["as_of"] for r in group})
        # Keep all assets/positions of selected timestamps together.
        width = max(sum(r["as_of"] == t for r in group) for t in times)
        if max_samples < width:
            raise ValueError(
                "Example budget must fit a complete paired timestamp across all assets"
            )
        count = max(1, min(len(times), max_samples // width))
        chosen = {times[i] for i in np.linspace(0, len(times) - 1, count, dtype=int)}
        selected.extend(r for r in group if r["as_of"] in chosen)
    destination = Path(output)
    manifest_path = Path(str(output) + ".manifest.json")
    if destination.exists() or manifest_path.exists():
        raise ValueError("Reviewer cohort exists; use a new immutable output")
    plan = {
        "contract": "reviewer-cohort-v1",
        "backend": backend,
        "model_version": provider.version,
        "dataset_sha256": source["dataset_sha256"],
        "symbols": source["symbols"],
        "timeframe": source["timeframe"],
        "horizon": source["horizon"],
        "feature_names": source["feature_names"],
        "max_examples_per_phase": max_samples,
        "cohort_sha256": digest([(r["symbol"], r["as_of"], r["position"]) for r in selected]),
        "calibration_boundary": source["boundaries"]["calibration"],
        "validation_boundary": source["boundaries"]["validation"],
        "foundation_trained_until": getattr(provider, "trained_until", None),
        "input_contract": "shared-evidence-v2",
        "prospective": False,
        "automatic_activation": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    with destination.open("x") as stream:
        for row in selected:
            started = time.monotonic()
            result = {
                "id": f"{row['symbol']}:{row['as_of']}:{row['position']}",
                "symbol": row["symbol"],
                "as_of": row["as_of"],
                "observed_at": row["outcome"]["observed_at"],
                "position": row["position"],
                "phase": row["phase"],
                "target": row["label"],
                "input_sha256": digest(
                    {
                        "market": row["market"],
                        "history": row["history"],
                        "evidence": row["evidence"],
                    }
                ),
                "model_version": provider.version,
            }
            try:
                value = Proposal.model_validate(
                    provider.predict_with_evidence(
                        MarketInput.model_validate(row["market"]), row["history"], row["evidence"]
                    ).model_dump()
                )
                if value.model_version != provider.version or not value.probabilities:
                    raise ValueError("Cohort requires a stable probabilistic reviewer")
                result.update(
                    status="ok",
                    probabilities=value.probabilities,
                    action=value.action,
                    review_details=value.review_details,
                )
            except Exception:  # noqa: BLE001 - failures retained; credentials never written
                result["status"] = "unavailable_or_invalid"
            result["inference_ms"] = (time.monotonic() - started) * 1000
            results.append(result)
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
    from .clef_runtime import sha256

    plan.update(
        {
            "scores_sha256": sha256(destination),
            "attempted": len(results),
            "succeeded": sum(r["status"] == "ok" for r in results),
        }
    )
    manifest_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def fit_calibration(scores, output):
    from .clef_runtime import sha256

    path, destination = Path(scores), Path(output)
    if destination.exists():
        raise ValueError("Calibration exists; use a new immutable artifact")
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if manifest.get("contract") != "reviewer-cohort-v1" or sha256(path) != manifest.get(
        "scores_sha256"
    ):
        raise ValueError("Calibration cohort contract/digest mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    calibration = [r for r in rows if r["phase"] == "calibration"]
    if any(r["status"] != "ok" for r in calibration):
        raise ValueError(
            "Calibration cohort contains failed members; retain failures and rerun a declared cohort"
        )
    if (
        not calibration
        or len({r["as_of"] for r in calibration}) < 30
        or any(r["model_version"] != manifest["model_version"] for r in calibration)
        or any(
            not manifest["calibration_boundary"]
            <= r["as_of"]
            < r["observed_at"]
            < manifest["validation_boundary"]
            for r in calibration
        )
    ):
        raise ValueError(
            "Need >=30 complete, independent calibration timestamps with strictly purged labels"
        )
    probability = np.asarray([[r["probabilities"][a] for a in ACTIONS] for r in calibration])
    truth = np.asarray([ACTIONS.index(r["target"]) for r in calibration])
    if set(truth.tolist()) != {0, 1, 2}:
        raise ValueError("Calibration needs all three position-aware target classes")
    temperatures = sorted(set([1.0, *np.geomspace(0.25, 10, 81).tolist()]))
    temperature = min(
        temperatures, key=lambda t: log_loss(truth, scaled(probability, t), labels=[0, 1, 2])
    )
    artifact = {
        "contract": "reviewer-temperature-v1",
        "source_model_version": manifest["model_version"],
        "cohort_sha256": manifest["scores_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "temperature": float(temperature),
        "samples": len(calibration),
        "independent_timestamps": len({r["as_of"] for r in calibration}),
        "labels_last": max(r["observed_at"] for r in calibration),
        "symbols": manifest["symbols"],
        "timeframe": manifest["timeframe"],
        "horizon": manifest["horizon"],
        "feature_names": manifest["feature_names"],
        "fit_phase": "calibration_only",
        "raw_metrics": probability_diagnostics(probability, truth),
        "calibrated_fit_metrics": probability_diagnostics(scaled(probability, temperature), truth),
        "held_out_metrics": {},
        "automatic_activation": False,
        "validation_status": "research_only",
        "real_execution_enabled": False,
    }
    # These diagnostics never choose temperature or a policy threshold.
    for phase in ("validation", "test"):
        group = [r for r in rows if r["phase"] == phase and r["status"] == "ok"]
        if any(
            r["as_of"] <= artifact["labels_last"] or r["model_version"] != manifest["model_version"]
            for r in group
        ):
            raise ValueError("Held-out observations overlap calibration or changed model identity")
        if group:
            p = [[r["probabilities"][a] for a in ACTIONS] for r in group]
            y = [ACTIONS.index(r["target"]) for r in group]
            artifact["held_out_metrics"][phase] = probability_diagnostics(scaled(p, temperature), y)
    artifact["artifact_sha256"] = digest(artifact)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    return artifact


class CalibratedReviewer:
    def __init__(self, model, path):
        self.model = model
        self.calibration = json.loads(Path(path).read_text())
        c = self.calibration
        expected = c.get("artifact_sha256")
        if (
            c.get("contract") != "reviewer-temperature-v1"
            or c.get("source_model_version") != model.version
            or digest({k: v for k, v in c.items() if k != "artifact_sha256"}) != expected
        ):
            raise ValueError("Calibration artifact differs from source model or content identity")
        self.name, self.version = (
            model.name + "-temperature",
            digest({"model": model.version, "calibration": expected})[:16],
        )
        self.model_family = getattr(model, "model_family", None)
        self.metadata = getattr(model, "metadata", {})
        self.supports_locked_forward = getattr(model, "supports_locked_forward", True)
        cutoff = getattr(model, "trained_until", None)
        self.trained_until = max(cutoff, c["labels_last"]) if cutoff is not None else None

    def predict_with_evidence(self, market, history, evidence):
        c = self.calibration
        if (
            market.timestamps[-1] <= c["labels_last"]
            or market.symbol not in c["symbols"]
            or (market.timeframe, market.horizon) != (c["timeframe"], c["horizon"])
            or sorted(evidence.get("features", {})) != sorted(c["feature_names"])
        ):
            raise ValueError(
                "Calibration requires a matching, strictly later shared-evidence snapshot"
            )
        value = self.model.predict_with_evidence(market, history, evidence)
        if value.model_version != c["source_model_version"] or not value.probabilities:
            raise ValueError("Reviewer drifted from calibrated model")
        probability = scaled([value.probabilities[a] for a in ACTIONS], c["temperature"])
        value = value.model_copy(deep=True)
        value.probabilities = dict(zip(ACTIONS, map(float, probability)))
        value.action = ACTIONS[int(probability.argmax())]
        value.confidence = float(probability.max())
        value.probability_calibration = "past-only temperature fit; research_only"
        value.model_version = self.version
        value.review_details["calibration_sha256"] = c["artifact_sha256"]
        return value
