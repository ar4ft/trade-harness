import numpy as np

from .schemas import MarketInput

FEATURE_NAMES = ["return_1", "return_5", "return_20", "volatility_20", "range_1", "volume_ratio"]


def features(market: MarketInput) -> np.ndarray:
    prices = np.asarray(market.ohlc, dtype=float)
    close = prices[:, 3]
    returns = np.diff(close[-21:]) / close[-21:-1]
    volume = np.asarray(market.volume[-20:], dtype=float)
    return np.array(
        [
            close[-1] / close[-2] - 1,
            close[-1] / close[-6] - 1,
            close[-1] / close[-21] - 1,
            returns.std(),
            (prices[-1, 1] - prices[-1, 2]) / close[-1],
            volume[-1] / max(float(volume.mean()), 1e-12),
        ]
    )


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
