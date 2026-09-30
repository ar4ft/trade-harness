from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Indicator(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    values: list[float | None]


class MarketInput(StrictModel):
    symbol: str = Field(min_length=1, max_length=40)
    timeframe: str = Field(pattern=r"^[1-9][0-9]*[mhdw]$")
    timestamps: list[int] = Field(description="Closed candle end times, Unix milliseconds")
    ohlc: list[list[float]] = Field(min_length=21, max_length=10000)
    volume: list[float]
    indicators: list[Indicator] = Field(default_factory=list, max_length=50)
    position: Literal["flat", "long"] = "flat"
    horizon: int = Field(default=3, ge=1, le=100)

    @model_validator(mode="after")
    def validate_series(self):
        n = len(self.ohlc)
        if len(self.volume) != n or len(self.timestamps) != n:
            raise ValueError("OHLC, volume, and timestamps must have equal lengths")
        if any(b <= a for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("Timestamps must be strictly increasing")
        if any(t < 0 for t in self.timestamps):
            raise ValueError("Timestamps must be nonnegative")
        if any(v < 0 for v in self.volume):
            raise ValueError("Volume must be nonnegative")
        for row in self.ohlc:
            if len(row) != 4:
                raise ValueError("Each OHLC row is [open, high, low, close]")
            opening, high, low, close = row
            if (
                min(row) <= 0
                or low > min(opening, close)
                or high < max(opening, close)
                or low > high
            ):
                raise ValueError("Invalid OHLC price bounds")
        names = [i.name for i in self.indicators]
        if len(set(names)) != len(names):
            raise ValueError("Indicator names must be unique")
        if any(len(i.values) != n for i in self.indicators):
            raise ValueError("Every indicator must align with candle timestamps")
        return self


class Forecast(StrictModel):
    horizon: int = Field(ge=1, le=100)
    expected_return: float = Field(gt=-1, le=10)
    projected_close: float = Field(gt=0)
    method: str


class Proposal(StrictModel):
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=4000)
    forecast: Forecast
    probabilities: dict[str, float] | None = None
    probability_calibration: str = "uncalibrated"
    validation_status: Literal["validated", "research_only", "unknown"] = "unknown"
    model_version: str | None = None
    forecast_interval: list[float] | None = None

    @model_validator(mode="after")
    def validate_optional_distribution(self):
        if self.probabilities is not None:
            if set(self.probabilities) != {"BUY", "SELL", "HOLD"}:
                raise ValueError("Action probabilities must cover BUY, SELL, and HOLD")
            if (
                any(not 0 <= p <= 1 for p in self.probabilities.values())
                or abs(sum(self.probabilities.values()) - 1) > 1e-5
            ):
                raise ValueError("Action probabilities must be normalized and finite")
            if (
                not hasattr(self, "guardrails")
                and abs(self.confidence - self.probabilities[self.action]) > 1e-5
            ):
                raise ValueError("Confidence must equal the probability of the proposed action")
        if self.forecast_interval is not None:
            if (
                len(self.forecast_interval) != 2
                or self.forecast_interval[0] > self.forecast_interval[1]
            ):
                raise ValueError("Forecast interval must be ordered [lower, upper]")
        return self


class Decision(Proposal):
    id: str
    symbol: str
    timeframe: str
    as_of: int
    backend: str
    guardrails: list[str]
    proposed_action: Literal["BUY", "SELL", "HOLD"] | None = None
    execution: dict | None = None
    fields: dict = Field(default_factory=dict)


class Feedback(StrictModel):
    decision_id: str
    realized_return: float = Field(gt=-1, le=10)
    reviewed_action: Literal["BUY", "SELL", "HOLD"] | None = None
    notes: str = Field(default="", max_length=2000)
    observed_at: int = Field(ge=0, description="Unix milliseconds; must follow forecast horizon")
