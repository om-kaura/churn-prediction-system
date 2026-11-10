"""
Prediction history — a real, inspectable SQLite database, separate from
data/telco_churn.csv (which is the training data, not this). Every call to
/predict, /predict/{model}, or /predict/compare is logged here: what was sent
in, which model scored it, and what came back. Deliberately plain sqlite3
(stdlib) rather than an ORM/Postgres — this is a single append-only table with
no migrations to manage, so the extra weight wouldn't buy anything.

Open it directly to show your teacher real logged predictions:
    sqlite3 outputs/predictions.db "select * from predictions order by id desc limit 5;"
"""
import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_used TEXT NOT NULL,
    churn_prediction TEXT NOT NULL,
    churn_probability REAL NOT NULL,
    input_features TEXT NOT NULL
);
"""


def init_db(db_path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def log_prediction(
    model_used: str,
    churn_prediction: str,
    churn_probability: float,
    input_features: dict,
    db_path: str = DB_PATH,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO predictions "
            "(created_at, model_used, churn_prediction, churn_probability, input_features) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                model_used,
                churn_prediction,
                churn_probability,
                json.dumps(input_features),
            ),
        )
        conn.commit()


def get_history(limit: int = 50, db_path: str = DB_PATH) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, model_used, churn_prediction, churn_probability, input_features "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        record = dict(r)
        record["input_features"] = json.loads(record["input_features"])
        out.append(record)
    return out


def count_predictions(db_path: str = DB_PATH) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
