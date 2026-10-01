"""Optional TypeSafe-shaped Nimble service adapter; forecasts remain numerical."""

import json
import os

import httpx

from .learning import DecisionModel
from .schemas import Forecast, ForecastEvidence, Proposal
from .typed import ChoiceResult, NoulResult


class NimbleModel:
    name = "nimble-with-numerical-forecast-v1"

    def __init__(self, forecast_model=None):
        from pathlib import Path

        self.forecast_model = forecast_model or DecisionModel(
            str(Path(__file__).parent / "assets/decision_model.json")
        )
        self.trained_until = self.forecast_model.trained_until
        self.version = "nimble+" + self.forecast_model.version
        self.base_url = os.environ.get(
            "NIMBLE_BASE_URL", "https://bespokelabs--nimble-sglang-nimble.us-west.modal.direct"
        ).rstrip("/")
        self.headers = {}
        if os.environ.get("NIMBLE_API_KEY"):
            self.headers["Authorization"] = "Bearer " + os.environ["NIMBLE_API_KEY"]
        if os.environ.get("NIMBLE_MODAL_KEY") and os.environ.get("NIMBLE_MODAL_SECRET"):
            self.headers.update(
                {
                    "Modal-Key": os.environ["NIMBLE_MODAL_KEY"],
                    "Modal-Secret": os.environ["NIMBLE_MODAL_SECRET"],
                }
            )

    def predict(self, market, history, evidence=None):
        if evidence is None:
            baseline = self.forecast_model.predict(market, history)
        else:
            source = ForecastEvidence.model_validate(evidence["forecast"])
            if source.as_of != market.timestamps[-1] or source.horizon != market.horizon:
                raise ValueError("Shared forecast timestamp/horizon mismatch")
            baseline = Proposal(
                action="HOLD", confidence=0, rationale="Shared forecast, independent direction review",
                forecast=Forecast(horizon=market.horizon, expected_return=source.expected_return,
                                  projected_close=market.ohlc[-1][3] * (1 + source.expected_return),
                                  method=source.source), forecast_interval=source.return_interval,
            )
        state = {
            "market": {
                "symbol": market.symbol,
                "timeframe": market.timeframe,
                "horizon": market.horizon,
                "last_ohlc": market.ohlc[-5:],
                "last_volume": market.volume[-5:],
                "indicators": {i.name: i.values[-3:] for i in market.indicators},
                "as_of": market.timestamps[-1],
                "position": market.position,
            },
            "numerical_forecast": baseline.forecast.model_dump(),
            "numerical_direction_probabilities": baseline.probabilities,
            "past_decisions": history[-5:],
            "rules": "Long/cash only. Use only available historical evidence. HOLD when evidence does not justify transaction costs.",
        }
        if evidence is not None:
            state.pop("numerical_direction_probabilities")
            state["shared_evidence"] = evidence
        payload = {
            "model": os.environ.get("NIMBLE_MODEL", "nimble-latest"),
            "state": json.dumps(state),
            "questions": {
                "direction": {
                    "type": "choice",
                    "instructions": "Choose the trading direction for the stated horizon.",
                    "criteria": {
                        "BUY": "Evidence favors appreciation exceeding transaction costs",
                        "SELL": "Evidence favors depreciation; exit an existing long",
                        "HOLD": "No sufficient directional evidence",
                    },
                },
                "evidence_sufficient": {
                    "type": "noul",
                    "instructions": "Does the supplied evidence justify a directional trade after costs?",
                },
            },
        }
        with httpx.Client(timeout=30) as client:
            response = client.post(
                self.base_url + "/v1/systemone", headers=self.headers, json=payload
            )
            response.raise_for_status()
        answers = response.json()["answers"]
        answer = answers["direction"]
        typed = ChoiceResult(
            choice=answer["choice"],
            probabilities=answer["probabilities"],
            source="nimble",
            calibrated=False,
        )
        enough = NoulResult(noul=answers["evidence_sufficient"]["noul"], source="nimble")
        probabilities = typed.probabilities
        # A separate field is advisory. It cannot silently change the selected direction probability.
        action = typed.choice
        rationale = f"Nimble typed direction; evidence-sufficient score={enough.noul:.3f}. Numerical forecast supplied independently."
        return Proposal(
            action=action,
            confidence=probabilities[action],
            probabilities=probabilities,
            rationale=rationale,
            forecast=baseline.forecast,
            forecast_interval=baseline.forecast_interval,
            probability_calibration="uncalibrated",
            validation_status="research_only",
            model_version=self.version,
        )

    def predict_with_evidence(self, market, history, evidence):
        return self.predict(market, history, evidence=evidence)
