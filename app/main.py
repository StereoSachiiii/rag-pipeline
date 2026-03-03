from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.query import RAGQuery
from groq import APIConnectionError, GroqError, RateLimitError, APIStatusError, AuthenticationError 
from app.core.groq_provider import GroqProvider


class QueryRequest(BaseModel):
    user_query: str



app = FastAPI()
rag_query = RAGQuery(llm_provider=GroqProvider()) #hardcoded for now, you can make this dynamic based on user input or config

@app.post("/query")
def query(request: QueryRequest):    

    try: 
        results = rag_query.query(request.user_query)    
        return {"answer": results["answer"], "results": results['sources']}
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid Groq API Key.")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Groq rate limit reached.")
    except APIConnectionError:
        raise HTTPException(status_code=503, detail="Groq service is unreachable.")
    except APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=f"Groq API error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal server error occurred.")