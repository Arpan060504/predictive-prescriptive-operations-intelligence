import sqlite3
import os
from pathlib import Path

db_path = Path("database/operations.db")
print(f"Database exists: {db_path.exists()}")
if db_path.exists():
    print(f"File size: {os.path.getsize(db_path) / (1024 * 1024):.2f} MB")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name;")
    items = cur.fetchall()
    print(f"Total objects: {len(items)}")
    for name, obj_type in items:
        if obj_type == "table":
            cur.execute(f'SELECT COUNT(*) FROM "{name}"')
            cnt = cur.fetchone()[0]
            print(f"{obj_type:6} {name:35} : {cnt:,} rows")
        else:
            print(f"{obj_type:6} {name:35}")
    conn.close()
