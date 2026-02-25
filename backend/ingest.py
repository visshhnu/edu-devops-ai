import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client(
    Settings(
        persist_directory="chroma_db",
        is_persistent=True,
        anonymized_telemetry=False
    )
)

collection = client.get_or_create_collection(name="edu_knowledge")

def parse_metadata(content):
    lines = content.split("\n")
    metadata = {}
    body_start = 0

    for i, line in enumerate(lines):
        if line.startswith("DOMAIN:"):
            metadata["domain"] = line.replace("DOMAIN:", "").strip()
        elif line.startswith("LEVEL:"):
            metadata["level"] = line.replace("LEVEL:", "").strip()
        elif line.startswith("MODULE:"):
            metadata["module"] = line.replace("MODULE:", "").strip()
        elif line.strip() == "":
            body_start = i + 1
            break

    body = "\n".join(lines[body_start:])
    return metadata, body

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

knowledge_path = "knowledge"

doc_id = 0

for root, dirs, files in os.walk(knowledge_path):
    for file in files:
        filepath = os.path.join(root, file)

        with open(filepath, "r") as f:
            content = f.read()

        metadata, body = parse_metadata(content)
        chunks = chunk_text(body)

        for chunk in chunks:
            embedding = embedding_model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[f"doc_{doc_id}"]
            )

            doc_id += 1

print("Multi-domain knowledge ingested successfully!")
