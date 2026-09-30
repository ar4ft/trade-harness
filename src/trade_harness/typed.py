"""Validated choice/noul/score contracts inspired by Jev and Nimble."""

from typing import Literal

from pydantic import Field, model_validator

from .schemas import StrictModel


class ChoiceResult(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    source: str
    calibrated: bool = False

    @model_validator(mode="after")
    def valid_distribution(self):
        if self.choice not in self.probabilities:
            raise ValueError("Selected choice is absent from the candidate distribution")
        if not self.probabilities or any(not 0 <= p <= 1 for p in self.probabilities.values()):
            raise ValueError("Candidate probabilities must be finite and within [0,1]")
        if abs(sum(self.probabilities.values()) - 1) > 1e-5:
            raise ValueError("Candidate probabilities must sum to one")
        if self.probabilities[self.choice] < max(self.probabilities.values()) - 1e-8:
            raise ValueError("Selected choice must maximize candidate probability")
        return self


class NoulResult(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1)
    source: str
    calibrated: bool = False


class ScoreResult(StrictModel):
    type: Literal["score"] = "score"
    score: float = Field(ge=0)
    probabilities: dict[str, float]
    source: str

    @model_validator(mode="after")
    def expected_level(self):
        if len(self.probabilities) < 2 or any(not 0 <= p <= 1 for p in self.probabilities.values()):
            raise ValueError("Invalid ordered level probabilities")
        if abs(sum(self.probabilities.values()) - 1) > 1e-5:
            raise ValueError("Level probabilities must sum to one")
        expected = sum(i * p for i, p in enumerate(self.probabilities.values()))
        if abs(self.score - expected) > 1e-5:
            raise ValueError("Score must equal the expected ordered level")
        return self


FieldResult = ChoiceResult | NoulResult | ScoreResult
