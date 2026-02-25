from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from router import build_prompt, call_llm

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client(
    Settings(
        persist_directory="chroma_db",
        is_persistent=True,
        anonymized_telemetry=False
    )
)

collection = client.get_collection("edu_knowledge")

class ChatRequest(BaseModel):
    question: str
    domain: str | None = None
    level: str = "beginner"

@app.get("/health")
def health():
    return {"status": "AI backend running"}

@app.post("/chat")
def chat(request: ChatRequest):

    query_embedding = embedding_model.encode(request.question).tolist()

    if request.domain:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            where={"domain": request.domain}
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )

    if not results["documents"] or not results["documents"][0]:
        return {"answer": "No relevant content found."}

    context = "\n".join(results["documents"][0])

    prompt = build_prompt(context, request.question, request.level)

    answer = call_llm(prompt)

    return {"answer": answer}
