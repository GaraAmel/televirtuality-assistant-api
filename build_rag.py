import os
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="rag_db")
collection = client.get_or_create_collection("animals")

folder = "data/animals"

for filename in os.listdir(folder):
    if filename.endswith(".txt"):
        path = os.path.join(folder, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        embedding = embedder.encode(text).tolist()

        collection.add(
            ids=[filename],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"source": filename}]
        )

print("Base RAG créée avec succès.")