from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client and collection
chroma_client = chromadb.Client()
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="document_chunks",
    embedding_function=embedding_function
)

def embed_chunks(chunks, doc_id):
    """
    Adds chunks to ChromaDB with metadata
    """
    chunk_texts = [chunk["chunk_text"] for chunk in chunks]
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{
        "doc_id": chunk["doc_id"],
        "page": chunk["page"],
        "paragraph": chunk["paragraph"]
    } for chunk in chunks]

    collection.add(documents=chunk_texts, metadatas=metadatas, ids=ids)

def embed_query(query: str, top_k=3):
    """
    Performs a similarity search in ChromaDB
    """
    results = collection.query(query_texts=[query], n_results=top_k)
    return results
