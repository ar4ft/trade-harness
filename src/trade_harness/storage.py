import json
import sqlite3
import threading

from .schemas import Decision, Feedback, MarketInput


def timeframe_ms(timeframe: str) -> int:
    return (
        int(timeframe[:-1])
        * {"m": 60000, "h": 3600000, "d": 86400000, "w": 604800000}[timeframe[-1]]
    )


class Store:
    def __init__(self, path="decisions.sqlite"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS decisions (
            id TEXT PRIMARY KEY, symbol TEXT, timeframe TEXT, as_of INTEGER,
            market TEXT, decision TEXT, feedback TEXT)""")
        self.db.commit()

    def save(self, market: MarketInput, decision: Decision):
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO decisions VALUES (?,?,?,?,?,?,NULL)",
                (
                    decision.id,
                    market.symbol,
                    market.timeframe,
                    decision.as_of,
                    market.model_dump_json(),
                    decision.model_dump_json(),
                ),
            )

    def history(self, market: MarketInput, limit=20):
        with self.lock:
            rows = self.db.execute(
                """SELECT decision, feedback FROM decisions
                WHERE symbol=? AND timeframe=? AND as_of<? ORDER BY as_of DESC LIMIT ?""",
                (market.symbol, market.timeframe, market.timestamps[-1], limit),
            ).fetchall()
        result = []
        for raw, feedback in reversed(rows):
            item = json.loads(raw)
            if feedback:
                observed = json.loads(feedback)
                if observed["observed_at"] < market.timestamps[-1]:
                    item["feedback"] = observed
            result.append(item)
        return result

    def feedback(self, feedback: Feedback):
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT decision FROM decisions WHERE id=?", (feedback.decision_id,)
            ).fetchone()
            if row is None:
                raise KeyError(feedback.decision_id)
            decision = json.loads(row[0])
            earliest = (
                decision["as_of"]
                + timeframe_ms(decision["timeframe"]) * decision["forecast"]["horizon"]
            )
            if feedback.observed_at < earliest:
                raise ValueError("Outcome must be observed after the forecast horizon")
            self.db.execute(
                "UPDATE decisions SET feedback=? WHERE id=?",
                (feedback.model_dump_json(), feedback.decision_id),
            )

    def reviewed_examples(self):
        with self.lock:
            rows = self.db.execute(
                "SELECT market,decision,feedback FROM decisions WHERE feedback IS NOT NULL"
            ).fetchall()
        return [
            (json.loads(m), json.loads(d), json.loads(f))
            for m, d, f in rows
            if json.loads(f).get("reviewed_action")
        ]
