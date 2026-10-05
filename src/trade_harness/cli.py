import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from .backtest import backtest
from .harness import Harness
from .models import load_model
from .schemas import MarketInput
from .storage import Store
from .training import export_finetuning, train

DEFAULT_MARKET = Path(__file__).parent / "assets/latest.json"


def main():
    parser = argparse.ArgumentParser(description="Trained trading decision CLI")
    parser.add_argument(
        "--version", action="version", version=importlib.metadata.version("trade-harness")
    )
    parser.add_argument(
        "command",
        choices=[
            "decide",
            "backtest",
            "train",
            "fetch",
            "train-language",
            "export-finetuning",
            "evaluate",
            "train-hybrid",
            "tune-hybrid",
            "paper",
            "paper-replay",
            "status",
            "update",
            "serve",
            "validate",
            "export-research",
            "export-clef",
            "review-cases",
            "score-reviewer",
            "calibrate-reviewer",
            "scorecard",
            "evaluate-research",
            "lock-research",
            "evaluate-forward",
        ],
    )
    parser.add_argument("--input")
    parser.add_argument("--dataset", help="Immutable shared-evidence research JSONL")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--base-revision")
    parser.add_argument("--feature-recipe", choices=["base", "interactions", "quant"], default="base")
    parser.add_argument("--max-decisions", type=int, default=0, help="Predeclare a bounded audit schedule; 0 uses all")
    parser.add_argument("--allow-unknown-cutoff", action="store_true", help="Bounded retrospective audit only; cannot certify reviewer chronology")
    parser.add_argument("--max-samples", type=int, default=256, help="Reviewer cohort budget per held-out phase")
    parser.add_argument("--per-case", type=int, default=8)
    parser.add_argument("--reviewer-path", help="Candidate local reviewer adapter directory")
    parser.add_argument("--prospective-plan", help="Immutable declaration for future paper observations")
    parser.add_argument("--audit-output", help="Append-only forward paper journal JSONL")
    parser.add_argument("--trial-count", type=int, default=1, help="Declare prior research trials when locking a policy")
    parser.add_argument("--output")
    parser.add_argument("--db", default=None)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2025-09", help="Archive start month YYYY-MM")
    parser.add_argument("--end", default="2026-08", help="Archive end month YYYY-MM")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--with-indicators", action="store_true")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=96)
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--position", choices=["flat", "long"])
    parser.add_argument(
        "--backend",
        choices=["decision", "baseline", "trained", "local-llm", "llm", "nimble", "ollaya", "clef", "clef-flash", "llamafile", "timesfm", "hybrid", "strategy", "orchestrator"],
    )
    parser.add_argument("--data-dir", default="data/markets")
    parser.add_argument("--forecast-cache", default="artifacts/hybrid/forecasts.jsonl")
    parser.add_argument("--forecast-stride", type=int, default=48)
    parser.add_argument("--forecast-context", type=int, default=100)
    parser.add_argument("--model-output")
    parser.add_argument("--trial-plan", help="JSON list of bounded feature/parameter trials")
    review_config = parser.add_mutually_exclusive_group()
    review_config.add_argument("--reviewers", help="Comma-separated language review backends")
    review_config.add_argument("--orchestrator-config", help="JSON consensus policy configuration")
    parser.add_argument("--run-id", default="paper-BTCUSDT-1h")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Paper ticks (default 1, 0 until stopped), or language training steps (default 120)",
        dest="paper_steps",
    )
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--risk-config")
    parser.add_argument(
        "--research", action="store_true", help="Allow simulated entries for an unvalidated model"
    )
    parser.add_argument("--folds", type=int)
    updates = parser.add_mutually_exclusive_group()
    updates.add_argument(
        "--check", action="store_true", help="Verify and check for a release (default)"
    )
    updates.add_argument(
        "--apply",
        action="store_true",
        help="Install a verified update into the managed environment",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.db is None:
        args.db = (
            os.environ.get("TRADING_PAPER_DB", "paper.sqlite")
            if args.command in ("paper", "paper-replay", "status")
            else os.environ.get("TRADING_DB", "decisions.sqlite")
        )
    if args.reviewers:
        os.environ["TRADING_ORCHESTRATOR_REVIEWERS"] = args.reviewers
    if args.orchestrator_config:
        os.environ["TRADING_ORCHESTRATOR_CONFIG"] = args.orchestrator_config
    if args.backend:
        os.environ["TRADING_BACKEND"] = args.backend
    if args.command == "update":
        from .updater import UpdateError, Updater

        try:
            result = Updater().update(apply=args.apply)
        except UpdateError as error:
            parser.exit(1, f"{error}\n")
        print(json.dumps(result, indent=2))
        return
    if args.command == "serve":
        import uvicorn

        uvicorn.run("trade_harness.api:app", host=args.host, port=args.port)
        return
    if args.command == "review-cases":
        from .casebook import export_cases

        if not args.dataset or not args.output:
            parser.error("--dataset (comma-separated paths) and --output are required")
        print(json.dumps(export_cases(args.dataset.split(","), args.output, args.per_case), indent=2))
        return
    if args.command == "score-reviewer":
        from .reviewer_calibration import score_reviewer

        if not args.dataset or not args.output or args.backend not in ("local-llm", "llamafile", "llm", "nimble", "ollaya", "clef", "clef-flash"):
            parser.error("--dataset, --output and an explicit reviewer --backend are required")
        if args.reviewer_path:
            os.environ["TRADING_LORA_PATH"] = args.reviewer_path
        print(json.dumps(score_reviewer(args.dataset, args.backend, args.output, args.max_samples), indent=2))
        return
    if args.command == "calibrate-reviewer":
        from .reviewer_calibration import fit_calibration

        if not args.input or not args.output:
            parser.error("--input reviewer cohort and --output are required")
        print(json.dumps(fit_calibration(args.input, args.output), indent=2))
        return
    if args.command == "scorecard":
        from .operations import export_scorecard

        if not args.input or not args.output:
            parser.error("--input journals/SQLite paths and --output are required")
        print(json.dumps(export_scorecard(args.input.split(","), args.output), indent=2))
        return
    if args.command == "export-research":
        from .research_data import export_research

        paths = [str(p) for p in Path(args.data_dir).glob("*.json")
                 if not p.name.endswith(".provenance.json")]
        result = export_research(paths, args.forecast_cache,
                                 args.output or "artifacts/research/examples.jsonl",
                                 args.forecast_stride, args.forecast_context,
                                 {"recipe": args.feature_recipe})
        print(json.dumps(result, indent=2))
        return
    if args.command == "export-clef":
        from .clef import export_clef

        if not args.dataset:
            parser.error("--dataset is required")
        if args.backend not in (None, "clef", "clef-flash"):
            parser.error("Use --backend clef or clef-flash for native exports")
        result = export_clef(args.dataset, args.output or "artifacts/research/clef-examples.jsonl",
                             args.backend or "clef")
        print(json.dumps(result, indent=2))
        return
    if args.command == "evaluate-forward":
        from .prospective import summarize_forward

        if not args.prospective_plan or not args.input:
            parser.error("--prospective-plan and --input forward journal are required")
        paths = [str(p) for p in Path(args.data_dir).glob("*.json")
                 if not p.name.endswith(".provenance.json")]
        result = summarize_forward(args.prospective_plan, args.input.split(","), paths,
                                   args.output or "reports/forward-evaluation.json")
        print(json.dumps({"promotion": result["promotion"],
                          "matured_decisions": result["matured_decisions"]}, indent=2))
        return
    if args.command in ("evaluate-research", "lock-research"):
        from .research_evaluation import evaluate_research, lock_research

        if not args.dataset:
            parser.error("--dataset is required")
        reviewers = [v.strip() for v in (args.reviewers or "local-llm").split(",")]
        if args.reviewer_path:
            os.environ["TRADING_LORA_PATH"] = args.reviewer_path
        if args.command == "evaluate-research":
            evaluate_research(args.dataset, reviewers, args.output or "reports/research-evaluation.json",
                              folds=args.folds if args.folds is not None else 5,
                              max_decisions=args.max_decisions, allow_unknown_cutoff=args.allow_unknown_cutoff)
        else:
            result = lock_research(args.dataset, reviewers,
                                   args.output or "artifacts/research/prospective-plan.json", args.trial_count)
            print(json.dumps(result, indent=2))
        return
    if args.command == "validate":
        from .learning import DecisionModel
        from .validation import validation_report

        if args.input and args.backend == "hybrid":
            from .hybrid import HybridModel

            model = HybridModel(args.input)
        else:
            model = DecisionModel(args.input) if args.input else load_model()
        result = validation_report(model)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    if args.command == "tune-hybrid":
        from .experiments import tune_hybrid

        paths = [str(p) for p in Path(args.data_dir).glob("*.json")
                 if not p.name.endswith(".provenance.json")]
        tune_hybrid(
            paths, cache=args.forecast_cache, stride=args.forecast_stride,
            context=args.forecast_context, folds=args.folds if args.folds is not None else 3,
            output=args.output or "reports/parameter-experiments.json",
            model_output=args.model_output or str(Path(__file__).parent / "assets/tuned_hybrid_model.json"),
            trials=json.loads(Path(args.trial_plan).read_text()) if args.trial_plan else None,
        )
        return
    if args.command == "train-hybrid":
        from .hybrid_training import evaluate_hybrid

        paths = [str(p) for p in Path(args.data_dir).glob("*.json")
                 if not p.name.endswith(".provenance.json")]
        evaluate_hybrid(
            paths, cache=args.forecast_cache, stride=args.forecast_stride,
            context=args.forecast_context, folds=args.folds if args.folds is not None else 3,
            output=args.output or "reports/hybrid-comparison.json",
            model_output=args.model_output or str(Path(__file__).parent / "assets/hybrid_model.json"),
        )
        return
    if args.command == "evaluate":
        from .evaluation import evaluate

        paths = [
            str(p)
            for p in Path(args.data_dir).glob("*.json")
            if not p.name.endswith(".provenance.json")
        ]
        evaluate(paths, args.output or "reports/walk-forward.json",
                 model_output=args.model_output or "artifacts/research/numerical-candidate.json",
                 folds=args.folds if args.folds is not None else 3)
        return
    elif args.command == "paper":
        from .live import load_risk_config, run_live

        result = run_live(
            load_model(),
            db=args.db,
            run_id=args.run_id,
            symbol=args.symbol,
            timeframe=args.timeframe,
            horizon=args.horizon,
            steps=args.paper_steps if args.paper_steps is not None else 1,
            poll_seconds=args.poll_seconds,
            config=load_risk_config(args.risk_config, args.research),
            prospective_plan=args.prospective_plan,
            audit_output=args.audit_output,
        )
    elif args.command == "status":
        store = Store(args.db)
        result = {
            "runs": store.runs(),
            "events": store.events(args.run_id),
            "latest_decision": store.latest_decision(args.run_id),
        }
    elif args.command == "paper-replay":
        from .learning import window
        from .live import load_risk_config
        from .paper import PaperEngine

        market = MarketInput.model_validate_json(
            Path(args.input or "data/BTCUSDT-1h.json").read_text()
        )
        model = load_model()
        engine = PaperEngine(
            model,
            Store(args.db),
            args.run_id,
            market.symbol,
            market.timeframe,
            load_risk_config(args.risk_config, args.research),
        )
        for i in range(20, len(market.ohlc)):
            if market.timestamps[i] > getattr(model, "trained_until", -1):
                engine.replay(window(market, i), market.ohlc[i])
        result = engine.summary()
    elif args.command == "fetch":
        from .data import download

        result = download(
            args.symbol,
            args.timeframe,
            args.start,
            args.end,
            args.output or f"data/{args.symbol}-{args.timeframe}.json",
            args.horizon,
        )
    elif args.command == "train-language":
        from .language_training import main as train_language

        train_language(
            [
                "--input",
                args.input or "data/BTCUSDT-1h.json",
                "--output",
                args.output or "artifacts/trading-lora",
                "--steps",
                str(args.paper_steps if args.paper_steps is not None else 120),
                "--stride",
                str(args.stride),
                "--eval-samples",
                str(args.eval_samples),
                "--base-model",
                args.base_model,
                "--max-length", str(args.max_length),
            ]
            + (["--dataset", args.dataset] if args.dataset else [])
            + (["--base-revision", args.base_revision] if args.base_revision else [])
        )
        return
    elif args.command == "export-finetuning":
        result = export_finetuning(Store(args.db), args.output or "artifacts/trading-chat.jsonl")
    else:
        input_path = args.input or (
            str(DEFAULT_MARKET) if args.command == "decide" else "data/BTCUSDT-1h.json"
        )
        market = MarketInput.model_validate_json(Path(input_path).read_text())
        if args.position:
            market = market.model_copy(update={"position": args.position})
        if args.command == "train":
            result = train(market, args.output or "artifacts/forecast.json", args.with_indicators)
        elif args.command == "backtest":
            result = backtest(market, load_model())
        else:
            from .live import load_risk_config

            config = load_risk_config(args.risk_config, args.research)
            result = (
                Harness(
                    load_model(),
                    Store(args.db),
                    minimum_confidence=config.min_confidence,
                    risk_config=config,
                )
                .decide(market)
                .model_dump()
            )
    if args.output and args.command in ("decide", "paper", "paper-replay", "status", "backtest"):
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
