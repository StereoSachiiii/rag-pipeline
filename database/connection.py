import os
import psycopg2
from dotenv import load_dotenv

# 1. Load env before defining the dict
load_dotenv()

# 2. Centralized Config
CONNECTION_PARAMS = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "ragdb"),
    "user": os.getenv("DB_USER", "rag"),
    "password": os.getenv("DB_PASSWORD", "rag"),
    "port": int(os.getenv("DB_PORT", 5433))
}

def run_query(query, params=None, fetch=False):
    """Universal wrapper to handle DRY connection logic"""
    try:
        with psycopg2.connect(**CONNECTION_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if fetch:
                    return cur.fetchone()
    except Exception as e:
        print(f"Database Error: {e}")
        return None

def init_db():
    """Initializes extension and tables"""
    # Enable Vector Extension
    run_query("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create Table
    run_query("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding VECTOR(384) 
        );
    """)
    
    # Verify Connection
    version = run_query("SELECT version();", fetch=True)
    if version:
        print(f"DB Initialized. System Version: {version[0]}")

if __name__ == "__main__":
    init_db()
