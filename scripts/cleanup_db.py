import sqlite3

conn = sqlite3.connect("server.db")
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS _alembic_tmp_alerts")
cur.execute("DROP INDEX IF EXISTS ix_alerts_edge_pc_id")
conn.commit()
print("dropped tmp table and index if present")
conn.close()
