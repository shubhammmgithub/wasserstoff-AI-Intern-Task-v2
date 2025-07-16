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

    add_to_chroma(doc_id=filename, texts=new_texts, metadatas=new_chunks)
    stored_chunks.extend(new_chunks)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(stored_chunks, f, ensure_ascii=False, indent=2)

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

    try:
        matched = search_chroma(query, top_k=top_k)
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

    results = []
    for item in matched:
        metadata = item["metadata"]
        results.append({
            "score": round(float(item["score"]), 4),
            "chunk": item["content"],
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

    try:
        synthesized_answer = chat_with_groq(prompt)
    except Exception as e:
        synthesized_answer = f"❌ Error communicating with Groq: {str(e)}"

    last_search_result = {"query": query, "results": results}

    return jsonify({
        "query": query,
        "results": results,
        "synthesized_answer": synthesized_answer
    })


@app.route("/themes", methods=["POST"])
def generate_theme_summary():
    all_chunks = load_chunk_metadata()  # from persistent JSON
    if not all_chunks:
        return jsonify({"error": "No chunk metadata found."}), 400

    doc_themes = {}
    all_chunk_texts = []

    for doc_id, chunks in all_chunks.items():
        print(f"\n[Theme] Processing: {doc_id}")
        
        if not chunks:
            print(f"❌ No chunks found for {doc_id}")
            doc_themes[doc_id] = {"error": "No chunks found"}
            continue

        chunk_texts = [chunk['text'] for chunk in chunks if chunk.get('text', '').strip()]
        if not chunk_texts:
            print(f"❌ No valid text in chunks for {doc_id}")
            doc_themes[doc_id] = {"error": "No valid text found in document"}
            continue

        try:
            theme_prompt = "Identify the key themes or main ideas from the following document:"
            combined_text = "\n".join(chunk_texts[:10])  # Optional: limit for speed
            print(f"[Theme] Extracting from {len(chunk_texts)} chunks...")

            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": theme_prompt},
                    {"role": "user", "content": combined_text}
                ]
            )
            summary = response.choices[0].message.content.strip()
            doc_themes[doc_id] = {"theme_summary": summary}
            all_chunk_texts.extend(chunk_texts)

        except Exception as e:
            print(f"❌ Error summarizing {doc_id}: {e}")
            doc_themes[doc_id] = {"error": f"LLM error: {str(e)}"}

    if not all_chunk_texts:
        return jsonify({"error": "No valid chunk text found across documents."}), 400

    # Overall theme synthesis
    try:
        global_prompt = "Identify the most important themes that are common across all these documents:"
        global_input = "\n".join(all_chunk_texts[:30])  # Optional: limit
        global_response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": global_prompt},
                {"role": "user", "content": global_input}
            ]
        )
        theme_summary = global_response.choices[0].message.content.strip()
    except Exception as e:
        theme_summary = f"Error generating overall summary: {str(e)}"

    return jsonify({
        "theme_summary": theme_summary,
        "themes_by_document": doc_themes
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
