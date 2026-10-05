"""Native Cloudflare Clef decision review; forecasts remain owned by the harness."""

import hashlib
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .clef_contract import CONTRACT, MAX_REQUEST_BYTES, QUESTIONS, RELEASES
from .risk import RiskConfig
from .schemas import Forecast, ForecastEvidence, Proposal
from .typed import ChoiceResult, NoulResult, ScoreResult


def review_state(market, history, evidence, reference_costs=None, context_candles=100):
    from .orchestrator import safe_history

    if not 21 <= context_candles <= 100:
        raise ValueError("Declare a raw Clef context of 21–100 closed candles")
    # A declared raw window is distinct from the numerical forecaster's history.
    bounded = market.model_dump()
    for field in ("timestamps", "ohlc", "volume"):
        bounded[field] = bounded[field][-context_candles:]
    for indicator in bounded["indicators"]:
        indicator["values"] = indicator["values"][-context_candles:]
    risk = reference_costs or RiskConfig()
    return {
        "market": bounded,
        "past_decisions": safe_history(history, market),
        "shared_evidence": evidence,
        "input_scope": {
            "available_candles": len(market.ohlc),
            "raw_context_candles": min(context_candles, len(market.ohlc)),
            "semantics": "Declared raw trailing window; features and forecasts retain their independently recorded causal history.",
        },
        "reference_costs": {
            "fee_bps_per_side": risk.fee_bps,
            "slippage_bps_per_side": risk.slippage_bps,
            "minimum_net_edge": risk.min_net_edge,
            "semantics": "Research reference only; actual account risk/costs are enforced separately.",
        },
    }


def distribution(answer, keys):
    values = answer.get("probabilities")
    if not isinstance(values, dict) or set(values) != set(keys):
        raise ValueError("Clef returned a different option schema")
    if any(
        isinstance(p, bool)
        or not isinstance(p, (int, float))
        or not math.isfinite(p)
        or not 0 <= p <= 1
        for p in values.values()
    ):
        raise ValueError("Invalid Clef option probabilities")
    total = sum(values.values())
    # The upstream SystemOne converter rounds every option to four decimal places.
    if abs(total - 1) > 0.0005:
        raise ValueError("Clef probabilities do not sum to one")
    return {key: values[key] / total for key in keys}


class ClefModel:
    uses_history = True
    model_family = "clef"

    def __init__(self, variant="clef", forecast_model=None):
        if variant not in RELEASES:
            raise ValueError("Select clef or clef-flash")
        self.model, self.name = variant, variant + "-typed-review-v1"
        self.timeout = float(os.environ.get("CLEF_TIMEOUT_SECONDS", "30"))
        if not 5 <= self.timeout <= 600:
            raise ValueError("Clef timeout must be between 5 and 600 seconds")
        self.context_candles = int(os.environ.get("CLEF_CONTEXT_CANDLES", "100"))
        if not 21 <= self.context_candles <= 100:
            raise ValueError("Declare a raw Clef context of 21–100 closed candles")
        self.forecast_model = forecast_model
        self.base_url = os.environ.get("CLEF_BASE_URL", "").rstrip("/")
        self.headers = {}
        self.manifest = None
        self.supports_locked_forward = bool(self.base_url)
        if self.base_url:
            url = urlsplit(self.base_url)
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    "CLEF_BASE_URL must be an HTTP service URL without credentials/query"
                )
            if os.environ.get("CLEF_API_KEY"):
                self.headers["Authorization"] = "Bearer " + os.environ["CLEF_API_KEY"]
            self.endpoint = self.base_url + "/v1/systemone"
            self.manifest = self._manifest()
            self.provider = "self-hosted"
        else:
            account, token = (
                os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""),
                os.environ.get("CLOUDFLARE_AUTH_TOKEN", ""),
            )
            if not re.fullmatch(r"[a-fA-F0-9]{32}", account) or not token:
                raise ValueError("Set CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_AUTH_TOKEN or CLEF_BASE_URL")
            self.endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{variant}"
            self.headers["Authorization"] = "Bearer " + token
            self.provider = "workers-ai"
        identity = {
            "model": variant,
            "provider": self.provider,
            "questions": QUESTIONS,
            "contract": CONTRACT,
            "manifest": self.manifest,
            "reference_costs": RiskConfig().model_dump(),
            "raw_context_candles": self.context_candles,
            "timeout_seconds": self.timeout,
        }
        self.version = (
            variant
            + "-"
            + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        )
        # Neither a public release date nor the separate numerical model's cutoff is a known foundation fitting cutoff.
        self.trained_until = None
        self.custom_head = (self.manifest or {}).get("custom_head")
        if self.custom_head and self.custom_head["raw_context_candles"] != self.context_candles:
            raise ValueError("Native head raw context differs from the trained contract")

    def _manifest(self):
        with httpx.Client(timeout=5) as client:
            response = client.get(self.base_url + "/metadata", headers=self.headers)
            response.raise_for_status()
        value = response.json()
        expected = os.environ.get("CLEF_REVISION", RELEASES[self.model][1])
        if (
            value.get("model") != self.model
            or value.get("repository") != RELEASES[self.model][0]
            or value.get("revision") != expected
            or not re.fullmatch(r"[a-f0-9]{40}", expected)
            or value.get("contract") != CONTRACT
            or value.get("input_truncation") != "reject"
            or not re.fullmatch(r"[a-f0-9]{64}", value.get("server_sha256", ""))
            or not isinstance(value.get("max_length"), int)
            or not 1 <= value["max_length"] <= 65536
        ):
            raise ValueError("Clef server differs from pinned release or complete-input contract")
        return value

    def predict(self, market, history, evidence=None):
        if self.manifest is not None and self._manifest() != self.manifest:
            raise ValueError("Clef service changed after model declaration")
        if self.custom_head and (
            evidence is None or market.symbol not in self.custom_head["symbols"]
            or (market.timeframe, market.horizon) != (self.custom_head["timeframe"], self.custom_head["horizon"])
            or market.timestamps[-1] <= self.custom_head["fine_tuned_until"]
            or sorted(evidence.get("features", {})) != sorted(self.custom_head["feature_names"])
        ):
            raise ValueError("Market or shared evidence differs from the native head training contract")
        if evidence is None:
            if self.forecast_model is None:
                from .learning import DecisionModel

                self.forecast_model = DecisionModel(
                    str(Path(__file__).parent / "assets/decision_model.json")
                )
            primary = self.forecast_model.predict(market, [])
            forecast, interval = primary.forecast, primary.forecast_interval
            source = {"numerical_forecast": forecast.model_dump()}
        else:
            source = ForecastEvidence.model_validate(evidence["forecast"])
            expected = {
                "as_of": market.timestamps[-1],
                "symbol": market.symbol,
                "timeframe": market.timeframe,
                "horizon": market.horizon,
                "position": market.position,
            }
            if any(evidence.get(key) != value for key, value in expected.items()) or (
                source.as_of != expected["as_of"] or source.horizon != market.horizon
            ):
                raise ValueError("Clef shared evidence contract mismatch")
            forecast = Forecast(
                horizon=market.horizon,
                expected_return=source.expected_return,
                projected_close=market.ohlc[-1][3] * (1 + source.expected_return),
                method=source.source,
            )
            interval, source = source.return_interval, evidence
        payload = {
            "model": self.model,
            "state": review_state(market, history, source,
                                  reference_costs=RiskConfig.model_validate(self.custom_head["risk_config"]) if self.custom_head else None,
                                  context_candles=self.context_candles),
            "questions": QUESTIONS,
        }
        raw = json.dumps(
            payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ).encode()
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Clef input exceeds the complete-state byte budget")
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.endpoint,
                headers={**self.headers, "Content-Type": "application/json"},
                content=raw,
            )
            response.raise_for_status()
        body = response.json()
        if self.provider == "workers-ai":
            if body.get("success") is not True or body.get("errors"):
                raise ValueError("Workers AI inference failed")
            body = body["result"]
        if body.get("model") != self.model or set(body.get("answers", {})) != set(QUESTIONS):
            raise ValueError("Clef returned a different model or question schema")
        if "state_truncated" in body and body["state_truncated"] is not False:
            raise ValueError("Clef truncated the input state")
        if self.manifest is not None and (
            body.get("provenance") != self.manifest or body.get("state_truncated") is not False
        ):
            raise ValueError("Clef response lacks pinned complete-input provenance")
        direction, enough, risk = (body["answers"][key] for key in QUESTIONS)
        if (direction.get("type"), enough.get("type"), risk.get("type")) != (
            "choice",
            "noul",
            "score",
        ):
            raise ValueError("Clef answer types differ from the declared questions")
        typed = ChoiceResult(
            choice=direction["choice"],
            probabilities=distribution(direction, ("BUY", "SELL", "HOLD")),
            source=self.model,
            calibrated=False,
        )
        if isinstance(enough.get("noul"), bool):
            raise ValueError("Clef evidence answer must be a probability")
        sufficient = NoulResult(noul=enough["noul"], source=self.model)
        probabilities = distribution(risk, ("0", "1", "2", "3"))
        expected_score = sum(int(key) * p for key, p in probabilities.items())
        if (
            isinstance(risk.get("score"), bool)
            or not isinstance(risk.get("score"), (int, float))
            or not math.isfinite(risk["score"])
            or abs(risk["score"] - expected_score) > 0.001
        ):
            raise ValueError("Clef score differs from the expected ordinal level")
        ordered = ScoreResult(score=expected_score, probabilities=probabilities, source=self.model)
        return Proposal(
            action=typed.choice,
            confidence=typed.probabilities[typed.choice],
            probabilities=typed.probabilities,
            forecast=forecast,
            forecast_interval=interval,
            model_version=self.version,
            probability_calibration="uncalibrated",
            validation_status="research_only",
            rationale=f"{self.model} native typed research review. Evidence and risk scores are advisory; numerical forecast and deterministic risk checks remain separate.",
            review_details={
                "provider": self.provider,
                "model": self.model,
                "weights_pinned": self.manifest is not None,
                "manifest": self.manifest,
                "evidence_sufficient": sufficient.noul,
                "risk_level": ordered.model_dump(),
                "request_sha256": hashlib.sha256(raw).hexdigest(),
                "input_context_candles": min(self.context_candles, len(market.ohlc)),
                "score_semantics": "per-option probabilities; uncalibrated for trading",
                "usage": body.get("usage", {}),
            },
        )

    def predict_with_evidence(self, market, history, evidence):
        return self.predict(market, history, evidence=evidence)


def export_clef(dataset, output, variant="clef"):
    """Export native requests and separate position/cost targets; no adapter is trained."""
    from .research_data import load_research
    from .schemas import MarketInput

    if variant not in RELEASES:
        raise ValueError("Select clef or clef-flash")
    records, source = load_research(dataset)
    path, manifest_path = Path(output), Path(str(output) + ".manifest.json")
    if path.exists() or manifest_path.exists():
        raise ValueError("Clef export exists; use a new immutable output")
    config = RiskConfig.model_validate(source["risk_config"])
    context_candles = int(os.environ.get("CLEF_CONTEXT_CANDLES", "100"))
    lines = []
    for row in records:
        request = {
            "model": variant,
            "state": review_state(
                MarketInput.model_validate(row["market"]),
                row["history"],
                row["evidence"],
                config,
                context_candles,
            ),
            "questions": QUESTIONS,
        }
        if (
            len(
                json.dumps(
                    request, ensure_ascii=True, allow_nan=False, separators=(",", ":")
                ).encode()
            )
            > MAX_REQUEST_BYTES
        ):
            raise ValueError("Clef example exceeds the complete-state byte budget")
        lines.append(
            json.dumps(
                {
                    "id": f"{row['symbol']}:{row['as_of']}:{row['position']}",
                    "phase": row["phase"],
                    "request": request,
                    "targets": {"direction": row["label"]},
                    "outcome": row["outcome"],
                },
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    content = "\n".join(lines) + "\n"
    manifest = {
        "contract": CONTRACT,
        "model": variant,
        "examples": len(records),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "source_dataset_sha256": source["dataset_sha256"],
        "label_contract": source["label_contract"],
        "risk_config": source["risk_config"],
        "boundaries": source["boundaries"],
        "targeted_questions": ["direction"],
        "partitions": source["partitions"],
        "raw_context_candles": context_candles,
        "limitations": [
            "Automatic cost/position outcome targets are not expert or path-dependent policy labels.",
            "No labels are invented for evidence_sufficient or risk_level.",
            "This is a native request/target export, not a Clef training implementation.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(content)
    with manifest_path.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
