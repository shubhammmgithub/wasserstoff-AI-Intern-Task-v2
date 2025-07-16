"""Flask backend for AI chatbot using OCR, ChromaDB & Groq."""

import os
os.environ["CHROMA_TELEMETRY"] = "False"
import json
import csv
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from io import StringIO, BytesIO

from .ocr_utils import extract_text_from_file
from .text_utils import chunk_text
from .embedder import add_to_chroma, search_chroma
from .groq_utils import chat_with_groq

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, '..', '..', 'docs')
CHUNKS_FILE = os.path.join(BASE_DIR, 'chunk_metadata.json')
CHUNK_DATA_FILE = os.path.join(BASE_DIR, 'chunk_data.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Globals
stored_chunks = []
last_search_result = {}

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Wasserstoff Document AI Server is Running"

@app.route("/upload", methods=["POST"])
def upload_file():
    global stored_chunks

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    text = extract_text_from_file(filepath)
    new_chunks = chunk_text(text, doc_id=filename)
    new_texts = [chunk["chunk_text"] for chunk in new_chunks]

    # Store in Chroma
    add_to_chroma(doc_id=filename, texts=new_texts, metadatas=new_chunks)
    stored_chunks.extend(new_chunks)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(stored_chunks, f, ensure_ascii=False, indent=2)

    # Save text + metadata
    chunk_data = [{
        "doc_id": c.get("doc_id"),
        "page": c.get("page"),
        "paragraph": c.get("paragraph"),
        "text": c.get("chunk_text")
    } for c in new_chunks]

    if os.path.exists(CHUNK_DATA_FILE):
        with open(CHUNK_DATA_FILE, "r", encoding="utf-8") as f:
            chunk_data = json.load(f) + chunk_data

    with open(CHUNK_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    return jsonify({
        "filename": filename,
        "extracted_text_snippet": text[:300],
        "total_chunks": len(new_chunks),
        "status": "✅ Document embedded and ChromaDB index updated"
    })

@app.route("/search", methods=["POST"])
def search():
    global last_search_result

    data = request.get_json()
    query = data.get("query", "")
    top_k = int(data.get("top_k", 3))

    if not query:
        return jsonify({"error": "No query provided"}), 400
    if not stored_chunks:
        return jsonify({"error": "Upload a document first."}), 400

    results_raw = search_chroma(query, top_k=top_k)

    results = []
    for doc, score in zip(results_raw["documents"][0], results_raw["distances"][0]):
        metadata = doc["metadata"]
        results.append({
            "score": round(float(score), 4),
            "chunk": doc["content"],
            "doc_id": metadata.get("doc_id", "N/A"),
            "page": metadata.get("page"),
            "paragraph": metadata.get("paragraph")
        })

    context_chunks = "\n\n".join([
        f"[{r['doc_id']}, Page {r['page']}, Para {r['paragraph']}]: {r['chunk']}"
        for r in results
    ])

    prompt = f"""Hey GROQ, You are an expert summarizer. Based on the following top relevant document excerpts, identify the main themes or ideas and synthesize a short answer and please don't give irrelevant answer.

Always cite sources using the format: [DocID, Page, Para].

Excerpts:
{context_chunks}

Synthesized Answer:"""

    synthesized_answer = chat_with_groq(prompt)

    last_search_result = {"query": query, "results": results}

    return jsonify({
        "query": query,
        "results": results,
        "synthesized_answer": synthesized_answer
    })

@app.route("/themes", methods=["POST"])
def extract_themes():
    if not stored_chunks:
        return jsonify({"error": "No documents uploaded."}), 400

    grouped = {}
    for chunk in stored_chunks:
        doc_id = chunk.get("doc_id", "Unknown")
        grouped.setdefault(doc_id, []).append(chunk)

    theme_results = {}

    for doc_id, chunks in grouped.items():
        limited = chunks[:30]
        excerpt = "\n\n".join([
            f"[{c.get('doc_id')}, Page {c.get('page')}, Para {c.get('paragraph')}]: {c['chunk_text']}"
            for c in limited
        ])

        prompt = f"""Hey Groq,You are a theme summarization expert. Analyze the following document excerpts and summarize the common themes, patterns, or recurring ideas from them and give the most relevant answer.

Cite supporting passages by mentioning DocID, Page, and Paragraph wherever relevant.

Excerpts:
{excerpt}

Theme Summary:"""

        try:
            summary = chat_with_groq(prompt)
        except Exception as e:
            summary = f"Groq API error: {str(e)}"

        theme_results[doc_id] = {"summary": summary}

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
    global stored_chunks
    if os.path.exists(CHUNKS_FILE):
        print("ℹ️ Loading previous chunk metadata...")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            stored_chunks = json.load(f)
    else:
        print("ℹ️ No previous chunk metadata found. Starting fresh.")

if __name__ == "__main__":
    load_existing_index_and_chunks()
    app.run(debug=True)
