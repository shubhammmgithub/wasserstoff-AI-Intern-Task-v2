"""This is the Flask backend for AI chatbot

It handles:
- Document upload and preprocessing
- OCR, chunking, embedding, FAISS indexing
- Semantic search and citation-based answers
- Theme extraction using Groq

Initially used OpenAI but switched to Groq due to quota limits.
"""

import os
import json
import openai
import faiss
import numpy as np
import csv
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from io import StringIO, BytesIO

from .ocr_utils import extract_text_from_file
from .text_utils import chunk_text
from .embedder import embed_chunks, build_faiss_index, embed_query
from .groq_utils import chat_with_groq

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define paths
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, '..', '..', 'docs')
INDEX_FILE = os.path.join(BASE_DIR, 'faiss_index.index')
CHUNKS_FILE = os.path.join(BASE_DIR, 'chunk_metadata.json')
CHUNK_DATA_FILE = os.path.join(BASE_DIR, 'chunk_data.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Globals
faiss_index = None
stored_chunks = []
embedding_dim = 1536
last_search_result = {}

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Wasserstoff Document AI Server is Running"

@app.route("/upload", methods=["POST"])
def upload_file():
    global faiss_index, stored_chunks, embedding_dim

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    text = extract_text_from_file(filepath)
    new_chunks = chunk_text(text, doc_id=filename)
    new_texts = [chunk["chunk_text"] for chunk in new_chunks]
    new_embeddings = embed_chunks(new_texts)

    if faiss_index is None:
        faiss_index = build_faiss_index(new_embeddings)
        stored_chunks = new_chunks
    else:
        faiss_index.add(new_embeddings.astype("float32"))
        stored_chunks.extend(new_chunks)

    faiss.write_index(faiss_index, INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(stored_chunks, f, ensure_ascii=False, indent=2)

    chunk_data = []
    for chunk in new_chunks:
        chunk_data.append({
            "doc_id": chunk.get("doc_id"),
            "page": chunk.get("page"),
            "paragraph": chunk.get("paragraph"),
            "text": chunk.get("chunk_text")
        })

    if os.path.exists(CHUNK_DATA_FILE):
        with open(CHUNK_DATA_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        chunk_data = existing_data + chunk_data

    with open(CHUNK_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    embedding_dim = new_embeddings.shape[1]

    return jsonify({
        "filename": filename,
        "extracted_text_snippet": text[:300],
        "total_chunks": len(new_chunks),
        "embedding_dimension": embedding_dim,
        "status": "✅ Document embedded and FAISS index updated"
    })

@app.route("/search", methods=["POST"])
def search():
    global faiss_index, stored_chunks, last_search_result

    data = request.get_json()
    query = data.get("query", "")
    top_k = int(data.get("top_k", 3))

    if not query:
        return jsonify({"error": "No query provided"}), 400
    if faiss_index is None or not stored_chunks:
        return jsonify({"error": "Upload a document first."}), 400

    query_vector = embed_query(query)
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1).astype("float32")

    D, I = faiss_index.search(query_vector, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        chunk = stored_chunks[idx]
        results.append({
            "score": round(float(score), 4),
            "chunk": chunk["chunk_text"],
            "doc_id": chunk.get("doc_id", "N/A"),
            "page": chunk.get("page"),
            "paragraph": chunk.get("paragraph")
        })

    context_chunks = "\n\n".join([
        f"[{res['doc_id']}, Page {res.get('page')}, Para {res.get('paragraph')}]: {res['chunk']}"
        for res in results
    ])

    synthesis_prompt = f"""Hey GROQ, You are an expert summarizer. Based on the following top relevant document excerpts, identify the main themes or ideas and synthesize a short answer and please don't give irrelevant answer.

Always cite sources using the format: [DocID, Page, Para].

Excerpts:
{context_chunks}

Synthesized Answer:"""

    synthesized_answer = chat_with_groq(synthesis_prompt)

    last_search_result["query"] = query
    last_search_result["results"] = results

    return jsonify({
        "query": query,
        "results": results,
        "synthesized_answer": synthesized_answer
    })

@app.route("/themes", methods=["POST"])
def extract_themes():
    global stored_chunks

    if not stored_chunks:
        return jsonify({"error": "No documents uploaded."}), 400

    grouped = {}
    for chunk in stored_chunks:
        doc_id = chunk.get("doc_id", "Unknown")
        grouped.setdefault(doc_id, []).append(chunk)

    theme_results = {}

    for doc_id, chunks in grouped.items():
        limited_chunks = chunks[:30]

        chunk_text = "\n\n".join([
            f"[{chunk.get('doc_id')}, Page {chunk.get('page')}, Para {chunk.get('paragraph')}]: {chunk['chunk_text']}"
            for chunk in limited_chunks
        ])

        prompt = f"""Hey Groq,You are a theme summarization expert. Analyze the following document excerpts and summarize the common themes, patterns, or recurring ideas from them and give the most relevant answer.

Cite supporting passages by mentioning DocID, Page, and Paragraph wherever relevant.

Excerpts:
{chunk_text}

Theme Summary:"""

        try:
            theme_summary = chat_with_groq(prompt)
        except Exception as e:
            theme_summary = f"Groq API error: {str(e)}"

        theme_results[doc_id] = {"summary": theme_summary}

    return jsonify({
        "themes_by_document": theme_results,
        "total_documents": len(grouped)
    })

@app.route("/download_results", methods=["GET"])
def download_results():
    format = request.args.get("format", "txt")

    if not last_search_result:
        return jsonify({"error": "No search results available for download."}), 400

    query = last_search_result["query"]
    results = last_search_result["results"]

    if format == "csv":
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=["score", "doc_id", "page", "paragraph", "chunk"])
        writer.writeheader()
        for item in results:
            writer.writerow(item)
        output.seek(0)
        return send_file(BytesIO(output.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name="search_results.csv")
    else:
        content = f"Query: {query}\n\nResults:\n"
        for res in results:
            content += f"\n---\nScore: {res['score']}\nDoc: {res['doc_id']} | Page: {res['page']} | Paragraph: {res['paragraph']}\n\n{res['chunk']}\n"
        return send_file(BytesIO(content.encode()), mimetype="text/plain", as_attachment=True, download_name="search_results.txt")

def load_existing_index_and_chunks():
    global faiss_index, stored_chunks

    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("ℹ️ Loading previous FAISS index and chunk metadata...")
        faiss_index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            stored_chunks = json.load(f)
    else:
        print("ℹ️ No previous index found, starting fresh.")

if __name__ == "__main__":
    load_existing_index_and_chunks()
    app.run(debug=True)
