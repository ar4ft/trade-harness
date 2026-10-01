"""Allowlisted causal feature recipes; no generated code or fitted transforms outside training."""

from typing import Literal

import numpy as np
from pydantic import Field

from .learning import MODEL_FEATURES
from .schemas import StrictModel
from .strategies import STRATEGY_FEATURES

BASE_FEATURES = MODEL_FEATURES + STRATEGY_FEATURES + [
    "forecast_return", "forecast_lower", "forecast_upper", "forecast_width", "forecast_vs_momentum",
]
INTERACTION_FEATURES = [
    "forecast_per_volatility", "uncertainty_per_atr", "forecast_trend_alignment",
    "breakout_volume_interaction",
]
FEATURE_RECIPE_VERSION = "causal-recipes-v1"


class FeatureConfig(StrictModel):
    version: Literal["causal-recipes-v1"] = FEATURE_RECIPE_VERSION
    recipe: Literal["base", "interactions"] = "base"


class FitParameters(StrictModel):
    logistic_c: float = Field(default=0.5, ge=0.01, le=10)
    ridge_alpha: float = Field(default=10, ge=0.1, le=1000)


def feature_names(config):
    config = FeatureConfig.model_validate(config)
    return BASE_FEATURES + (INTERACTION_FEATURES if config.recipe == "interactions" else [])


def transform_features(x, config):
    config = FeatureConfig.model_validate(config)
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != len(BASE_FEATURES) or not np.isfinite(x).all():
        raise ValueError("Feature recipe input must match the finite base contract")
    if config.recipe == "base":
        return x.copy()
    column = {name: x[..., i] for i, name in enumerate(BASE_FEATURES)}
    extra = np.stack([
        np.clip(column["forecast_return"] / np.maximum(column["volatility_20"], 1e-5), -20, 20),
        np.clip(column["forecast_width"] / np.maximum(column["atr_scaled"], 1e-5), 0, 50),
        column["forecast_return"] * column["trend_direction"],
        column["breakout_direction"] * np.clip(column["volume_ratio"], 0, 10),
    ], axis=-1)
    return np.concatenate([x, extra], axis=-1)
