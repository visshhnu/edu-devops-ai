import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from router import build_prompt, call_llm

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client(
    Settings(
        persist_directory="chroma_db",
        is_persistent=True,
        anonymized_telemetry=False
    )
)

collection = client.get_collection("edu_knowledge")

domain_input = input("Select domain (devops/cloud/ai/programming/linux/medical_coding or Enter for auto): ")
level_input = input("Select level (beginner/advanced): ")
query = input("Ask your question: ")

query_embedding = embedding_model.encode(query).tolist()

if domain_input:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"domain": domain_input}
    )
else:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

if not results["documents"] or not results["documents"][0]:
    print("No relevant content found.")
    exit()

contexts = "\n".join(results["documents"][0])

prompt = build_prompt(contexts, query, level_input)

answer = call_llm(prompt)

print("\nAI Answer:\n")
print(answer)
