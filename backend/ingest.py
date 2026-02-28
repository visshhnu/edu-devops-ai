import os
import uuid
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Chroma client
client = chromadb.Client(
    Settings(
        persist_directory="chroma_db",
        is_persistent=True,
        anonymized_telemetry=False
    )
)

collection = client.get_or_create_collection("edu_knowledge")


def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


knowledge_base_path = "knowledge"

for course in os.listdir(knowledge_base_path):
    course_path = os.path.join(knowledge_base_path, course)

    if not os.path.isdir(course_path):
        continue

    for level in os.listdir(course_path):
        level_path = os.path.join(course_path, level)

        if not os.path.isdir(level_path):
            continue

        for file in os.listdir(level_path):
            if not file.endswith(".txt"):
                continue

            module = file.replace(".txt", "")
            file_path = os.path.join(level_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text)
            embeddings = embedding_model.encode(chunks).tolist()
            ids = [str(uuid.uuid4()) for _ in chunks]

            metadatas = [
                {
                    "course": course,
                    "level": level,
                    "module": module
                }
                for _ in chunks
            ]

            collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            print(f"Ingested: {course} → {level} → {module}")

print("Structured ingestion complete!")
