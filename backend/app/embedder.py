# embedder.py (ChromaDB v0.5+ compatible)

from sentence_transformers import SentenceTransformer
import chromadb

# Initialize model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Correct new client init (v0.5+)
chroma_client = chromadb.PersistentClient(path="./chroma_index")
collection = chroma_client.get_or_create_collection(name="document_chunks")

# Add chunks to Chroma
def add_to_chroma(doc_id, texts, metadatas):
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(texts))]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

# Search
def search_chroma(query, top_k=3):
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    matched = []
    for i in range(len(results["ids"][0])):
        matched.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": results["distances"][0][i]
        })
    return matched
