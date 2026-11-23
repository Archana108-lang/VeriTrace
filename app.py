import sqlite3
import webbrowser
from datetime import datetime
from threading import Timer
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ml_engine import FraudDetectorEngine

app = FastAPI(title="VeriTrace Fraud Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_engine = FraudDetectorEngine()
DB_PATH = "veritrace.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            amount REAL NOT NULL,
            hour INTEGER NOT NULL,
            distance REAL NOT NULL,
            tx_velocity INTEGER NOT NULL,
            foreign_tx INTEGER NOT NULL,
            risk_score REAL NOT NULL,
            top_driver TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

class TransactionRequest(BaseModel):
    amount: float
    hour: int
    distance: float
    foreign: int
    tx_velocity: int

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/logs")
def get_audit_logs():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 50")
    rows = [dict(row) for row in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) FROM audit_logs")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM audit_logs WHERE status = 'HIGH RISK - BLOCK'")
    blocked = cursor.fetchone()[0]
    conn.close()

    return {
        "logs": rows,
        "stats": {
            "total": total,
            "blocked": blocked,
            "safe": total - blocked
        }
    }

@app.post("/predict")
def predict_fraud(tx: TransactionRequest):
    result = ai_engine.analyze_transaction(
        amount=tx.amount,
        hour=tx.hour,
        distance=tx.distance,
        foreign=tx.foreign,
        tx_velocity=tx.tx_velocity
    )

    now_str = datetime.now().strftime("%d %b, %I:%M:%S %p")
    top_driver = result["explanations"][0]["feature"] if result["explanations"] else "Normal Pattern"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audit_logs (timestamp, amount, hour, distance, tx_velocity, foreign_tx, risk_score, top_driver, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        tx.amount,
        tx.hour,
        tx.distance,
        tx.tx_velocity,
        tx.foreign,
        result["risk_score"],
        top_driver,
        result["status"]
    ))
    conn.commit()
    conn.close()

    return result

@app.delete("/logs/{log_id}")
@app.post("/logs/{log_id}/delete")
def delete_single_log(log_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM audit_logs WHERE id = ?", (log_id,))
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"message": f"Deleted record #{log_id}"}

@app.delete("/logs")
@app.post("/logs/clear")
def clear_all_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM audit_logs")
    try:
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='audit_logs'")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
    return {"message": "All audit logs cleared"}

def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    Timer(2.0, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)