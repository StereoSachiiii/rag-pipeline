import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

class DBConfig:
    @classmethod
    def get_config(cls):
        """Standardized connection parameters from Environment Variables"""
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "database": os.getenv("DB_NAME", "ragdb"),
            "user": os.getenv("DB_USER", "rag"),
            "password": os.getenv("DB_PASSWORD", "rag"),
            "port": int(os.getenv("DB_PORT", 5433))
        }

    @classmethod
    def run_query(cls, query, params=None, fetch=False, fetch_all=False):
        """Universal wrapper for database operations"""
        try:
            # Unpack the config directly into the connect function
            with psycopg2.connect(**cls.get_config()) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    if fetch_all:
                        return cur.fetchall()
                    if fetch:
                        return cur.fetchone()
        except Exception as e:
            print(f"Database Error: {e}")
            return None

    @classmethod
    def init_db(cls):
        """Bootstrap the pgvector extension and document table"""
        # 1. Check/Enable pgvector
        cls.run_query("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # 2. Setup Schema (VECTOR size 384 matches all-MiniLM-L6-v2)
        cls.run_query("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding VECTOR(384) 
            );
        """)

        # 3. Create HNSW index for fast approximate nearest-neighbor search
        cls.run_query("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents
            USING hnsw (embedding vector_cosine_ops);
        """)
        
        version = cls.run_query("SELECT version();", fetch=True)
        if version:
            print(f"✅ DB Initialized: {version[0]}")

if __name__ == "__main__":
    DBConfig.init_db()
