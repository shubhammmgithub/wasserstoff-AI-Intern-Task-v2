# backend/app/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Use ChromaDB's default in-memory DB
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(name="document_chunks")

def embed_chunks(texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

def embed_query(query):
    return model.encode(query, convert_to_numpy=True)

def add_to_chroma(doc_id, texts, metadatas):
    embeddings = embed_chunks(texts)
    ids = [f"{doc_id}_{i}" for i in range(len(texts))]
    chroma_collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )

def search_chroma(query, top_k=3):
    embedding = embed_query(query)
    results = chroma_collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k
    )
    return results
