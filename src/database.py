import sqlite3
import os
from datetime import datetime, timezone, timedelta

# IST timezone helper
IST_TZ = timezone(timedelta(hours=5, minutes=30))

def get_db_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path="mydata/system.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Create side camera queue table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS side_camera_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        side TEXT NOT NULL,
        model TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    
    # Create vehicle runs history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        fl_model TEXT,
        fr_model TEXT,
        rl_model TEXT,
        rr_model TEXT,
        status TEXT NOT NULL
    )
    """)
    
    conn.commit()
    conn.close()

def add_queue_item(side, model, db_path="mydata/system.db"):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now_ist = datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute(
        "INSERT INTO side_camera_queue (side, model, created_at) VALUES (?, ?, ?)",
        (side, model, now_ist)
    )
    conn.commit()
    conn.close()

def get_queue(side, db_path="mydata/system.db"):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT model FROM side_camera_queue WHERE side = ? ORDER BY id ASC",
        (side,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [row["model"] for row in rows]

def pop_queue_item(side, db_path="mydata/system.db"):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    # Find oldest item
    cursor.execute(
        "SELECT id, model FROM side_camera_queue WHERE side = ? ORDER BY id ASC LIMIT 1",
        (side,)
    )
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return None
    
    # Delete it
    item_id = row["id"]
    model = row["model"]
    cursor.execute("DELETE FROM side_camera_queue WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    return model

def clear_queues(db_path="mydata/system.db"):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM side_camera_queue")
    conn.commit()
    conn.close()

def save_vehicle_run(fl_model, fr_model, rl_model, rr_model, status, db_path="mydata/system.db"):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now_ist = datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute(
        """
        INSERT INTO vehicle_runs (timestamp, fl_model, fr_model, rl_model, rr_model, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (now_ist, fl_model, fr_model, rl_model, rr_model, status)
    )
    conn.commit()
    conn.close()

def get_vehicle_runs(db_path="mydata/system.db"):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicle_runs ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
