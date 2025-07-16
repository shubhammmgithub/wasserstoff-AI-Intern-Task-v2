"""Flask backend for AI chatbot using OCR, ChromaDB & Groq."""

import os
os.environ["CHROMA_TELEMETRY"] = "False"

import json
import csv
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from io import StringIO, BytesIO
import logging

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Globals
stored_chunks = []
last_search_result = {}

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Wasserstoff Document AI Server is Running"

def load_chunk_metadata():
    """Load chunk metadata from persistent JSON storage"""
    if not os.path.exists(CHUNK_DATA_FILE):
        return None
    
    try:
        with open(CHUNK_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Organize chunks by document ID
            organized_data = {}
            for chunk in data:
                doc_id = chunk.get("doc_id")
                if doc_id not in organized_data:
                    organized_data[doc_id] = []
                organized_data[doc_id].append(chunk)
            return organized_data
    except Exception as e:
        logger.error(f"Error loading chunk metadata: {str(e)}")
        return None

@app.route("/upload", methods=["POST"])
def upload_file():
    global stored_chunks

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded", "status": "error"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file", "status": "error"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        text = extract_text_from_file(filepath)
        new_chunks = chunk_text(text, doc_id=filename)
        new_texts = [chunk["chunk_text"] for chunk in new_chunks]

        add_to_chroma(doc_id=filename, texts=new_texts, metadatas=new_chunks)
        stored_chunks.extend(new_chunks)

        # Save chunk metadata
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(stored_chunks, f, ensure_ascii=False, indent=2)

        # Save chunk data for theme analysis
        chunk_data = [{
            "doc_id": c.get("doc_id"),
            "page": c.get("page"),
            "paragraph": c.get("paragraph"),
            "text": c.get("chunk_text")
        } for c in new_chunks]

        if os.path.exists(CHUNK_DATA_FILE):
            with open(CHUNK_DATA_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                chunk_data = existing_data + chunk_data

        with open(CHUNK_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

        return jsonify({
            "filename": filename,
            "extracted_text_snippet": text[:300],
            "total_chunks": len(new_chunks),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        return jsonify({
            "error": f"File processing failed: {str(e)}",
            "status": "error"
        }), 500

@app.route("/search", methods=["POST"])
def search():
    global last_search_result

    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        top_k = min(int(data.get("top_k", 3)), 20)  # Limit to max 20 results

        if not query:
            return jsonify({"error": "No query provided", "status": "error"}), 400
        if not stored_chunks:
            return jsonify({"error": "No documents available for search", "status": "error"}), 400

        matched = search_chroma(query, top_k=top_k)
        
        if not matched:
            return jsonify({
                "error": "No matching documents found",
                "status": "success",
                "results": []
            }), 200

        # Process results
        results = []
        for item in matched:
            metadata = item["metadata"]
            results.append({
                "score": round(float(item["score"]), 4),
                "chunk": item["content"],
                "doc_id": metadata.get("doc_id", "N/A"),
                "page": metadata.get("page", "N/A"),
                "paragraph": metadata.get("paragraph", "N/A")
            })

        # Store for download and theme analysis
        last_search_result = {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

        # Generate synthesized answer
        context_chunks = "\n\n".join([
            f"[{r['doc_id']}, Page {r['page']}, Para {r['paragraph']}]: {r['chunk']}"
            for r in results
        ])

        prompt = f"""Based on these document excerpts about '{query}', provide a concise answer:
        
        {context_chunks}
        
        Answer in 2-3 sentences, citing sources like [DocID, Page X]."""
        
        synthesized_answer = chat_with_groq(prompt)

        return jsonify({
            "query": query,
            "results": results,
            "synthesized_answer": synthesized_answer,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return jsonify({
            "error": f"Search failed: {str(e)}",
            "status": "error"
        }), 500

@app.route("/themes", methods=["POST"])
def generate_theme_summary():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({"error": "No query provided", "status": "error"}), 400

        # Get all relevant chunks from ChromaDB
        matched = search_chroma(query, top_k=20)  # Get more results for theme analysis
        
        if not matched:
            return jsonify({
                "error": "No matching documents found",
                "status": "success",
                "themes": []
            }), 200

        # Organize chunks by document
        doc_chunks = {}
        for item in matched:
            doc_id = item["metadata"].get("doc_id", "unknown")
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append({
                "text": item["content"],
                "page": item["metadata"].get("page", "N/A"),
                "paragraph": item["metadata"].get("paragraph", "N/A"),
                "score": float(item["score"])
            })

        # Analyze themes per document
        themes_by_doc = {}
        all_relevant_chunks = []
        
        for doc_id, chunks in doc_chunks.items():
            try:
                # Prepare document context
                chunk_texts = [c["text"] for c in chunks]
                context = "\n\n".join(chunk_texts[:10])  # Limit to top chunks
                
                # Generate document-level theme
                prompt = f"""Identify the key themes in this document that relate to '{query}':
                
                {context}
                
                Provide:
                1. A theme name
                2. A 2-3 sentence summary
                3. Key supporting points
                
                Format your response as:
                Theme: [theme name]
                Summary: [summary]
                Support: [bullet points]"""
                
                response = chat_with_groq(prompt)
                themes_by_doc[doc_id] = {
                    "theme_summary": response,
                    "top_chunks": chunks[:3]  # Include top chunks for reference
                }
                all_relevant_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Theme analysis failed for {doc_id}: {str(e)}")
                themes_by_doc[doc_id] = {
                    "error": f"Failed to analyze document: {str(e)}",
                    "status": "error"
                }

        # Generate overall theme synthesis
        try:
            top_chunks = sorted(all_relevant_chunks, key=lambda x: x["score"], reverse=True)[:15]
            context = "\n\n".join([c["text"] for c in top_chunks])
            
            prompt = f"""Synthesize the common themes across these documents regarding '{query}':
            
            {context}
            
            Provide:
            1. 2-3 main themes with names
            2. For each theme, a summary and supporting document references
            3. Format as:
            
            Theme 1: [name]
            - Summary: [summary]
            - Supported by: [Doc1, Page X], [Doc2, Page Y]
            
            Theme 2: [name]
            - ..."""
            
            theme_summary = chat_with_groq(prompt)
            
        except Exception as e:
            logger.error(f"Overall theme synthesis failed: {str(e)}")
            theme_summary = f"Error generating overall themes: {str(e)}"

        return jsonify({
            "query": query,
            "theme_summary": theme_summary,
            "themes_by_document": themes_by_doc,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Theme extraction failed: {str(e)}")
        return jsonify({
            "error": f"Theme extraction failed: {str(e)}",
            "status": "error"
        }), 500

@app.route("/download_results", methods=["GET"])
def download_results():
    format = request.args.get("format", "txt").lower()

    if not last_search_result or not last_search_result.get("results"):
        return jsonify({
            "error": "No search results available for download",
            "status": "error"
        }), 400

    try:
        query = last_search_result["query"]
        results = last_search_result["results"]

        if format == "csv":
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=["score", "doc_id", "page", "paragraph", "chunk"])
            writer.writeheader()
            for item in results:
                writer.writerow(item)
            output.seek(0)
            return send_file(
                BytesIO(output.getvalue().encode()),
                mimetype="text/csv",
                as_attachment=True,
                download_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        else:
            content = f"Query: {query}\n\nResults:\n"
            for res in results:
                content += f"\n---\nScore: {res['score']}\nDoc: {res['doc_id']} | Page: {res['page']} | Paragraph: {res['paragraph']}\n\n{res['chunk']}\n"
            return send_file(
                BytesIO(content.encode()),
                mimetype="text/plain",
                as_attachment=True,
                download_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return jsonify({
            "error": f"Download failed: {str(e)}",
            "status": "error"
        }), 500

def load_existing_index_and_chunks():
    global stored_chunks
    if os.path.exists(CHUNKS_FILE):
        logger.info("Loading previous chunk metadata...")
        try:
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                stored_chunks = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load existing chunks: {str(e)}")
            stored_chunks = []
    else:
        logger.info("No previous chunk metadata found. Starting fresh.")

if __name__ == "__main__":
    load_existing_index_and_chunks()
    app.run(host="0.0.0.0", port=5000, debug=True)