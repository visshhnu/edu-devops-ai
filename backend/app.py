from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

# -----------------------------
# CORS Configuration
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request Model
# -----------------------------
class ChatRequest(BaseModel):
    question: str
    course: str
    level: str
    module: str


# -----------------------------
# Load Embedding Model
# -----------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Connect to Chroma DB
# -----------------------------
client = chromadb.Client(
    Settings(persist_directory="chroma_db", is_persistent=True)
)

collection = client.get_collection("edu_knowledge")


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    try:
        # Build Chroma filter properly using $and
        where_filter = {
            "$and": [
                {"course": request.course},
                {"level": request.level},
                {"module": request.module}
            ]
        }

        # Query vector database
        results = collection.query(
            query_texts=[request.question],
            n_results=2,
            where=where_filter
        )

        documents = results.get("documents", [[]])[0]
        context = "\n".join(documents)

        if not context:
            return {"answer": "No relevant content found for selected module."}

        # Prepare LLM prompt
        prompt = f"""
You are a structured learning assistant.

Answer ONLY from the given context.
If answer is not in context, say:
"I don’t have enough information in this module."

Context:
{context}

Question:
{request.question}
"""

        # Call Ollama (phi3 recommended for speed)
        response = requests.post(
            "http://ollama:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        answer = response.json().get("response", "No response from model.")

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}
