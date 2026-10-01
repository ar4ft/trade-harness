import os
import secrets
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import Field

from .harness import Harness
from .live import load_risk_config
from .models import load_model
from .risk import Quote, RiskConfig
from .schemas import Decision, Feedback, MarketInput, StrictModel
from .storage import Store

app = FastAPI(title="Trade Harness", version="0.6.1")


@lru_cache
def runtime():
    config = load_risk_config(os.environ.get("TRADING_RISK_CONFIG"))
    return Harness(
        load_model(),
        Store(os.environ.get("TRADING_DB", "decisions.sqlite")),
        minimum_confidence=config.min_confidence,
        risk_config=config,
    )


@app.get("/health")
def health():
    return {"status": "ok", "mode": "decision_only", "real_execution_enabled": False}


@app.get("/validation")
def trading_validation(symbol: str | None = None):
    from .validation import validation_report

    return validation_report(runtime().model, symbol)


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


@app.middleware("http")
async def api_auth(request, call_next):
    key = os.environ.get("TRADING_API_KEY")
    if key and request.url.path not in ("/", "/health", "/docs", "/openapi.json"):
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied.encode(), ("Bearer " + key).encode()):
            return JSONResponse(status_code=401, content={"detail": "Bearer token required"})
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "assets/monitor.html").read_text()


@lru_cache
def paper_store():
    return Store(os.environ.get("TRADING_PAPER_DB", "paper.sqlite"))


@app.get("/paper/runs")
def paper_runs():
    result = paper_store().runs()
    for run in result:
        run["latest_decision"] = paper_store().latest_decision(run["run_id"])
    return {"mode": "paper", "runs": result}


@app.get("/paper/{run_id}/events")
def paper_events(run_id: str, limit: int = Query(default=200, ge=1, le=1000)):
    if paper_store().paper_account(run_id) is None:
        raise HTTPException(status_code=404, detail="Unknown paper run")
    return {"events": paper_store().events(run_id, limit)}


class PaperTick(StrictModel):
    run_id: str = Field(default="paper-BTCUSDT-1h", pattern=r"^[A-Za-z0-9_-]{1,80}$")
    market: MarketInput
    quote: Quote
    risk: RiskConfig = Field(default_factory=RiskConfig)


@app.post("/paper/tick")
def paper_tick(request: PaperTick):
    from .paper import PaperEngine

    try:
        engine = PaperEngine(
            runtime().model,
            paper_store(),
            request.run_id,
            request.market.symbol,
            request.market.timeframe,
            request.risk,
        )
        return engine.tick(request.market, request.quote, int(time.time() * 1000))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


class TradeQuestion(StrictModel):
    type: str
    instructions: str = ""
    criteria: dict[str, str] | list[str] | None = None


class SystemOneRequest(StrictModel):
    model: str = "trade-harness"
    state: MarketInput | str
    questions: dict[str, TradeQuestion] = Field(min_length=1, max_length=6)


@app.post("/v1/systemone")
def systemone(request: SystemOneRequest):
    """Typed trade fields, using TypeSafe-shaped transport; not arbitrary NLP questions."""
    if request.model != "trade-harness":
        raise HTTPException(
            status_code=422, detail="Use model trade-harness; backend is configured on the server"
        )
    definitions = {
        "direction": ("choice", {"BUY", "SELL", "HOLD"}),
        "execution": ("choice", {"BUY", "SELL", "HOLD"}),
        "regime": ("choice", {"UPTREND", "DOWNTREND", "SIDEWAYS"}),
        "risk_allowed": ("noul", None),
        "data_valid": ("noul", None),
        "risk_level": ("score", ["LOW", "MEDIUM", "HIGH"]),
    }
    for name, question in request.questions.items():
        if name not in definitions:
            raise HTTPException(status_code=422, detail="Unsupported trade question: " + name)
        kind, criteria = definitions[name]
        if question.type != kind:
            raise HTTPException(
                status_code=422, detail="Question type differs from fixed trade field"
            )
        if question.instructions:
            raise HTTPException(
                status_code=422,
                detail="Local trading questions have fixed semantics; omit instructions",
            )
        actual = (
            set(question.criteria) if isinstance(question.criteria, dict) else question.criteria
        )
        if actual != criteria:
            raise HTTPException(
                status_code=422, detail="Candidates must match the fixed trade field"
            )
    try:
        market = (
            MarketInput.model_validate_json(request.state)
            if isinstance(request.state, str)
            else request.state
        )
    except ValueError:
        raise HTTPException(
            status_code=422, detail="State must contain a valid MarketInput object or JSON"
        ) from None
    decision = runtime().decide(market)
    if "direction" in request.questions and "direction" not in decision.fields:
        raise HTTPException(
            status_code=503, detail="Configured model did not supply direction probabilities"
        )
    return {
        "model": "trade-harness",
        "model_version": decision.model_version,
        "decision_id": decision.id,
        "answers": {key: decision.fields[key] for key in request.questions},
        "execution": decision.execution,
        "guardrails": decision.guardrails,
    }
