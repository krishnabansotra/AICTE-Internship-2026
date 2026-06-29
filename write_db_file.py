from pathlib import Path

content = '''import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions.db"


def init_db():
    """Initialize database and create table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,
            present_price REAL,
            kms_driven INTEGER,
            fuel_type TEXT,
            seller_type TEXT,
            transmission TEXT,
            predicted_price REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
    conn.commit()
    conn.close()


def save_prediction(year, present_price, kms_driven, fuel_type, seller_type, transmission, predicted_price):
    """Save a prediction record into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            year, present_price, kms_driven, fuel_type, seller_type, transmission, predicted_price
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (year, present_price, kms_driven, fuel_type, seller_type, transmission, predicted_price))
    conn.commit()
    conn.close()


def get_all_predictions():
    """Fetch all saved predictions."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
'''

Path('app/database.py').write_text(content, encoding='utf-8')
print('database.py restored')
