"""
history_db.py - SQLite Persistent Storage for Rotating Machinery Fault Diagnosis History.
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "diagnostic_history.db"


def get_default_db_path() -> Path:
    """Return the standard path for the persistent diagnostic history database."""
    return DEFAULT_DB_PATH


def init_db(db_path: Path = None):
    """
    Initialize SQLite database and create the diagnostic_history table if it doesn't exist.
    Creates parent directories automatically.
    """
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(target_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diagnostic_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diagnosis_ref TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                selected_condition TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                prediction_probability REAL NOT NULL,
                tree_votes INTEGER NOT NULL,
                total_trees INTEGER NOT NULL,
                health_status TEXT NOT NULL,
                severity TEXT NOT NULL,
                fault_type TEXT NOT NULL,
                fault_desc TEXT NOT NULL,
                feat_max REAL NOT NULL,
                feat_min REAL NOT NULL,
                feat_mean REAL NOT NULL,
                feat_sd REAL NOT NULL,
                feat_rms REAL NOT NULL,
                feat_skewness REAL NOT NULL,
                feat_kurtosis REAL NOT NULL,
                feat_crest REAL NOT NULL,
                feat_form REAL NOT NULL,
                class_probabilities TEXT NOT NULL
            );
        """)
        conn.commit()
    finally:
        conn.close()


def save_diagnosis(record: dict, db_path: Path = None) -> str:
    """
    Save a completed, validated diagnosis to SQLite persistent storage within an atomic transaction.
    Generates a unique reference ID in the format 'DIA-0001'.
    
    Returns:
        diagnosis_ref (str): The unique assigned reference code (e.g. 'DIA-0001').
    """
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    init_db(target_path)
    
    timestamp = record.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    class_probs = record.get("class_probabilities", {})
    if isinstance(class_probs, dict):
        class_probs_json = json.dumps(class_probs)
    else:
        class_probs_json = str(class_probs)
        
    conn = sqlite3.connect(target_path, timeout=10.0)
    try:
        cursor = conn.cursor()
        
        # Calculate next monotonic reference number atomically
        cursor.execute("SELECT MAX(id) FROM diagnostic_history")
        row = cursor.fetchone()
        next_id = (row[0] + 1) if (row and row[0] is not None) else 1
        diagnosis_ref = f"DIA-{next_id:04d}"
        
        cursor.execute("""
            INSERT INTO diagnostic_history (
                diagnosis_ref, timestamp, selected_condition, predicted_class,
                prediction_probability, tree_votes, total_trees, health_status,
                severity, fault_type, fault_desc, feat_max, feat_min,
                feat_mean, feat_sd, feat_rms, feat_skewness, feat_kurtosis,
                feat_crest, feat_form, class_probabilities
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            diagnosis_ref,
            timestamp,
            str(record["selected_condition"]),
            str(record["predicted_class"]),
            float(record["prediction_probability"]),
            int(record["tree_votes"]),
            int(record.get("total_trees", 50)),
            str(record["health_status"]),
            str(record["severity"]),
            str(record.get("fault_type", "")),
            str(record.get("fault_desc", "")),
            float(record["feat_max"]),
            float(record["feat_min"]),
            float(record["feat_mean"]),
            float(record["feat_sd"]),
            float(record["feat_rms"]),
            float(record["feat_skewness"]),
            float(record["feat_kurtosis"]),
            float(record["feat_crest"]),
            float(record["feat_form"]),
            class_probs_json
        ))
        conn.commit()
        return diagnosis_ref
    finally:
        conn.close()


def get_history(db_path: Path = None, limit: int = None) -> pd.DataFrame:
    """
    Retrieve all historical diagnoses ordered with newest records first (id DESC).
    Returns a pandas DataFrame.
    """
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not target_path.exists():
        init_db(target_path)
        
    conn = sqlite3.connect(target_path)
    try:
        query = "SELECT * FROM diagnostic_history ORDER BY id DESC"
        if limit and limit > 0:
            query += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_diagnosis_by_ref(diagnosis_ref: str, db_path: Path = None) -> dict:
    """
    Retrieve complete stored data for a single diagnosis by reference ID (e.g. 'DIA-0001').
    Returns a dictionary or None if not found.
    """
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not target_path.exists():
        return None
        
    conn = sqlite3.connect(target_path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM diagnostic_history WHERE diagnosis_ref = ?", (diagnosis_ref,))
        row = cursor.fetchone()
        if not row:
            return None
            
        rec = dict(row)
        try:
            rec["class_probabilities"] = json.loads(rec["class_probabilities"])
        except Exception:
            pass
        return rec
    finally:
        conn.close()


def filter_history(category: str = None, search_query: str = None, db_path: Path = None) -> pd.DataFrame:
    """
    Filter historical diagnoses by predicted fault category and/or diagnosis reference ID search.
    Returns matching records with newest first.
    """
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not target_path.exists():
        init_db(target_path)
        
    conditions = []
    params = []
    
    if category and category != "All Classes":
        conditions.append("predicted_class = ?")
        params.append(category)
        
    if search_query and search_query.strip():
        conditions.append("diagnosis_ref LIKE ?")
        params.append(f"%{search_query.strip()}%")
        
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT * FROM diagnostic_history {where_clause} ORDER BY id DESC"
    
    conn = sqlite3.connect(target_path)
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()


def clear_history(db_path: Path = None) -> int:
    """
    Delete all diagnostic history records from the database table.
    Preserves table structure. Returns count of deleted records.
    """
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not target_path.exists():
        return 0
        
    conn = sqlite3.connect(target_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM diagnostic_history")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM diagnostic_history")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='diagnostic_history'")
        conn.commit()
        return count
    finally:
        conn.close()
