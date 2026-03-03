from app.database.connection import DBConfig
import os
from dotenv import load_dotenv
from app.core.llm_provider import LLMProvider
from sentence_transformers import SentenceTransformer

# for users u need to use the same embedding model as in ingest.py to ensure compatibility
# mine has 384 dimensions, so the vector length should be 384 and embedding model should reflect that
class RAGQuery:
    def __init__(self,
      llm_provider : LLMProvider,
      embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
     ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm_provider = llm_provider

    def query(self, user_query):
        vector = self.embedding_model.encode(user_query).tolist()

        results = DBConfig.run_query(
            "SELECT content, metadata FROM documents ORDER BY embedding <=> %s::vector LIMIT 3",
            params=(vector,),
            fetch_all=True
        )

        context_text = "\n\n".join([res[0] for res in results]) 
        sources = [{"content": res[0], "metadata": res[1]} for res in results]

        # Call LLM
        llm_response = self.llm_provider.generate(context_text, user_query)
        
        return {
            "answer": llm_response,
            "sources": sources
        }
        

   