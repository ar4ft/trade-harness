import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from .harness import Harness
from .models import load_model
from .schemas import Decision, Feedback, MarketInput
from .storage import Store

app = FastAPI(title="Trade Harness", version="0.1.0")


@lru_cache
def runtime():
    return Harness(load_model(), Store(os.environ.get("TRADING_DB", "decisions.sqlite")))


@app.get("/health")
def health():
    return {"status": "ok", "mode": "research/paper"}


@app.post("/decisions", response_model=Decision)
def decide(market: MarketInput):
    return runtime().decide(market)


@app.post("/feedback")
def feedback(value: Feedback):
    try:
        runtime().store.feedback(value)
    except KeyError:
        raise HTTPException(status_code=404, detail="Decision not found") from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    return {"status": "recorded"}


@app.post("/backtest")
def run_backtest(market: MarketInput):
    from .backtest import backtest

    try:
        return backtest(market, runtime().model)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
