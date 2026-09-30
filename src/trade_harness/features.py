import numpy as np

from .schemas import MarketInput

FEATURE_NAMES = ["return_1", "return_5", "return_20", "volatility_20", "range_1", "volume_ratio"]

INDICATOR_FEATURE_NAMES = ["sma_distance", "rsi_scaled", "macd_scaled", "atr_scaled"]


def features(market: MarketInput, include_indicators=False) -> np.ndarray:
    prices = np.asarray(market.ohlc, dtype=float)
    close = prices[:, 3]
    returns = np.diff(close[-21:]) / close[-21:-1]
    volume = np.asarray(market.volume[-20:], dtype=float)
    result = np.array(
        [
            close[-1] / close[-2] - 1,
            close[-1] / close[-6] - 1,
            close[-1] / close[-21] - 1,
            returns.std(),
            (prices[-1, 1] - prices[-1, 2]) / close[-1],
            volume[-1] / max(float(volume.mean()), 1e-12),
        ]
    )

    if include_indicators:
        values = {item.name: item.values[-1] for item in market.indicators}
        required = ("sma_20", "rsi_14", "macd", "atr_14")
        if any(values.get(key) is None for key in required):
            raise ValueError("Indicator-aware model requires SMA20, RSI14, MACD, and ATR14")
        result = np.concatenate(
            [
                result,
                [
                    values["sma_20"] / close[-1] - 1,
                    values["rsi_14"] / 100,
                    values["macd"] / close[-1],
                    values["atr_14"] / close[-1],
                ],
            ]
        )
    return result


def prefix(market: MarketInput, end: int) -> MarketInput:
    return market.model_copy(
        update={
            "ohlc": market.ohlc[:end],
            "volume": market.volume[:end],
            "timestamps": market.timestamps[:end],
            "indicators": [
                i.model_copy(update={"values": i.values[:end]}) for i in market.indicators
            ],
        }
    )
