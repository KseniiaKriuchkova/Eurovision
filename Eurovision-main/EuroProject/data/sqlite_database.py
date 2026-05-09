import sqlite3
import os

def save_to_sqlite(df, path="storage.db"):
    os.makedirs("storage", exist_ok=True)
    conn = sqlite3.connect(path)
    df.to_sql("songs", conn, if_exists="replace", index=False)
    conn.close()
