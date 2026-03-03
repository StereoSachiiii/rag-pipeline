import os
import glob
import json
from dotenv import load_dotenv



from app.database.connection import DBConfig 

# 2. RAG specific imports
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter
load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

def main():
    print("🚀 Starting Modular RAG Ingestion Pipeline...")

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = SentenceTransformer(model_name)
    
    headers_to_split_on = [("##", "Section"), ("###", "Subsection")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    md_files = glob.glob(os.path.join(DOCS_DIR, "*.md"))
    
    for file_path in md_files:
        filename = os.path.basename(file_path)
        print(f"📄 Processing: {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = markdown_splitter.split_text(content)
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content
            
            final_metadata = json.dumps({
                "source": filename,
                "headers": chunk.metadata
            })

            vector = embedder.encode(chunk_text).tolist()

            insert_query = """
                INSERT INTO documents (content, metadata, embedding)
                VALUES (%s, %s, %s)
            """
            DBConfig.run_query(insert_query, params=(chunk_text, final_metadata, vector))
            
            print(f"  -> Inserted chunk {i}")

    print(" Ingestion complete!")

if __name__ == "__main__":
    main()
