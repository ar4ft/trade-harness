"""Fresh forward evidence is mandatory; historical search results cannot authorize promotion."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromotionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    policy: Literal["research-promotion-v2"] = "research-promotion-v2"
    plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_version: str | None = None
    locked_at: int | None = None
    last_examined_at: int | None = None
    evaluation_start: int | None = None
    evaluation_end: int | None = None
    predictions_recorded_before_outcomes: bool = False
    model_and_policy_locked: bool = False
    fixed_evaluation_endpoint: bool = False
    trial_count: int = Field(default=0, ge=0)
    fold_count: int = Field(default=0, ge=0)
    closed_trades: int = Field(default=0, ge=0)
    positive_folds: int = Field(default=0, ge=0)
    positive_net_return: bool = False
    positive_double_cost_return: bool = False
    paired_interval_lower: float | None = None
    cash_interval_lower: float | None = None
    effective_time_blocks: int = Field(default=0, ge=0)
    block_days: int = Field(default=0, ge=0)
    horizon_dependency_days: int = Field(default=1, ge=1)
    directional_samples: int = Field(default=0, ge=0)
    minimum_decision_bin: int = Field(default=0, ge=0)
    directional_ece: float | None = Field(default=None, ge=0, le=1)
    probability_calibrated_on_past: bool = False
    purged_boundaries: bool = False
    all_planned_assets_observed: bool = False
    per_asset_checks_passed: bool = False
    forward_observation_coverage: float = Field(default=0, ge=0, le=1)
    matured_outcome_coverage: float = Field(default=0, ge=0, le=1)
    calibration_windows: int = Field(default=0, ge=0)
    failed_gates: list[str] = Field(default_factory=list)
    accepted: bool = False

    @model_validator(mode="after")
    def measured_gates(self):
        fresh = (self.locked_at is not None and self.last_examined_at is not None and
                 self.evaluation_start is not None and self.evaluation_end is not None and
                 self.evaluation_start > max(self.locked_at, self.last_examined_at) and
                 self.evaluation_end > self.evaluation_start)
        gates = {
            "fresh_locked_forward_period": bool(self.plan_sha256) and fresh,
            "predictions_before_outcomes": self.predictions_recorded_before_outcomes,
            "fixed_model_and_policy": self.model_and_policy_locked and bool(self.model_version),
            "predeclared_evaluation_endpoint": self.fixed_evaluation_endpoint,
            "search_history_recorded": self.trial_count >= 1,
            "planned_asset_coverage": self.all_planned_assets_observed and self.forward_observation_coverage >= 0.95,
            "matured_outcome_coverage": self.matured_outcome_coverage >= 0.95,
            "per_asset_edge_checks": self.per_asset_checks_passed,
            "five_purged_folds": self.fold_count >= 5 and self.purged_boundaries,
            "100_closed_trades": self.closed_trades >= 100,
            "three_positive_folds": self.positive_folds >= 3,
            "positive_net_and_stressed_returns": self.positive_net_return and self.positive_double_cost_return,
            "beats_momentum_with_uncertainty": self.paired_interval_lower is not None and self.paired_interval_lower > 0,
            "beats_cash_with_uncertainty": self.cash_interval_lower is not None and self.cash_interval_lower > 0,
            "20_time_blocks": self.effective_time_blocks >= 20 and self.block_days >= max(7, self.horizon_dependency_days),
            "100_directional_predictions": self.directional_samples >= 100,
            "30_examples_per_decision_bin": self.minimum_decision_bin >= 30,
            "decision_band_calibration": self.directional_ece is not None and self.directional_ece <= 0.1
                                         and self.probability_calibrated_on_past,
            "three_calibration_windows": self.calibration_windows >= 3,
        }
        self.failed_gates = [key for key, passed in gates.items() if not passed]
        self.accepted = not self.failed_gates
        return self
