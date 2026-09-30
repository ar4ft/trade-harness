import json
import sqlite3
import threading
from contextlib import contextmanager

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
        self._savepoint_sequence = 0
        self.db.execute("""CREATE TABLE IF NOT EXISTS decisions (
            id TEXT PRIMARY KEY, symbol TEXT, timeframe TEXT, as_of INTEGER,
            market TEXT, decision TEXT, feedback TEXT)""")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(decisions)")}
        if "scope" not in columns:
            self.db.execute("ALTER TABLE decisions ADD COLUMN scope TEXT NOT NULL DEFAULT ''")
        self.db.execute("""CREATE TABLE IF NOT EXISTS paper_accounts (
            run_id TEXT PRIMARY KEY, symbol TEXT, timeframe TEXT, model_version TEXT,
            config TEXT NOT NULL, state TEXT NOT NULL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS paper_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, timestamp INTEGER,
            kind TEXT, payload TEXT NOT NULL, event_key TEXT NOT NULL,
            UNIQUE(run_id, event_key))""")
        self.db.execute("CREATE INDEX IF NOT EXISTS decisions_scope_time ON decisions(scope,as_of)")
        self.db.commit()

    def save(self, market: MarketInput, decision: Decision, scope=""):
        with self.transaction():
            self.db.execute(
                "INSERT INTO decisions (id,symbol,timeframe,as_of,market,decision,feedback,scope) VALUES (?,?,?,?,?,?,NULL,?)",
                (
                    decision.id,
                    market.symbol,
                    market.timeframe,
                    decision.as_of,
                    market.model_dump_json(),
                    decision.model_dump_json(),
                    scope,
                ),
            )

    def history(self, market: MarketInput, limit=20, scope=""):
        with self.lock:
            rows = self.db.execute(
                """SELECT decision, feedback FROM decisions
                WHERE symbol=? AND timeframe=? AND as_of<? AND scope=? ORDER BY as_of DESC LIMIT ?""",
                (market.symbol, market.timeframe, market.timestamps[-1], scope, limit),
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
        with self.transaction():
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

    @contextmanager
    def transaction(self):
        """Nested savepoints preserve one atomic outer account/order/decision update."""
        with self.lock:
            nested = self.db.in_transaction
            self._savepoint_sequence += 1
            savepoint = "nested_" + str(self._savepoint_sequence)
            self.db.execute("SAVEPOINT " + savepoint if nested else "BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("RELEASE SAVEPOINT " + savepoint) if nested else self.db.commit()
            except BaseException:
                if nested:
                    self.db.execute("ROLLBACK TO SAVEPOINT " + savepoint)
                    self.db.execute("RELEASE SAVEPOINT " + savepoint)
                else:
                    self.db.rollback()
                raise

    def paper_account(self, run_id):
        with self.lock:
            row = self.db.execute(
                "SELECT symbol,timeframe,model_version,config,state FROM paper_accounts WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "symbol": row[0],
            "timeframe": row[1],
            "model_version": row[2],
            "config": json.loads(row[3]),
            "state": json.loads(row[4]),
        }

    def create_account(self, run_id, symbol, timeframe, model_version, config, state):
        with self.transaction():
            self.db.execute(
                "INSERT INTO paper_accounts VALUES (?,?,?,?,?,?)",
                (run_id, symbol, timeframe, model_version, json.dumps(config), json.dumps(state)),
            )

    def save_account(self, run_id, state):
        with self.transaction():
            self.db.execute(
                "UPDATE paper_accounts SET state=? WHERE run_id=?", (json.dumps(state), run_id)
            )

    def event(self, run_id, timestamp, kind, payload, event_key):
        with self.transaction():
            self.db.execute(
                "INSERT INTO paper_events(run_id,timestamp,kind,payload,event_key) VALUES (?,?,?,?,?)",
                (run_id, timestamp, kind, json.dumps(payload), event_key),
            )

    def events(self, run_id, limit=100):
        with self.lock:
            rows = self.db.execute(
                "SELECT id,timestamp,kind,payload FROM paper_events WHERE run_id=? ORDER BY id DESC LIMIT ?",
                (run_id, limit),
            ).fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "kind": r[2], "payload": json.loads(r[3])} for r in rows
        ]

    def runs(self):
        with self.lock:
            rows = self.db.execute(
                "SELECT run_id,symbol,timeframe,model_version,config,state FROM paper_accounts ORDER BY run_id"
            ).fetchall()
        return [
            {
                "run_id": r[0],
                "symbol": r[1],
                "timeframe": r[2],
                "model_version": r[3],
                "config": json.loads(r[4]),
                "state": json.loads(r[5]),
            }
            for r in rows
        ]

    def pending_outcomes(self, scope, earliest, latest):
        with self.lock:
            rows = self.db.execute(
                "SELECT id,as_of,decision FROM decisions WHERE scope=? AND feedback IS NULL AND as_of>=? AND as_of<?",
                (scope, earliest, latest),
            ).fetchall()
        return [
            {"id": r[0], "as_of": r[1], "horizon": json.loads(r[2])["forecast"]["horizon"]}
            for r in rows
        ]

    def latest_decision(self, scope):
        with self.lock:
            row = self.db.execute(
                "SELECT decision FROM decisions WHERE scope=? ORDER BY as_of DESC LIMIT 1", (scope,)
            ).fetchone()
        return json.loads(row[0]) if row else None
