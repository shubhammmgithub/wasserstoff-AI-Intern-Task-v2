import re

def chunk_text(text, doc_id="Unknown_Doc", chunk_size=500, overlap=100):
    """
    Splits input text into overlapping chunks with metadata.
    
    Each chunk is about `chunk_size` characters, and overlaps with the previous by `overlap` characters.
    Metadata includes:
    - doc_id: identifier for the source document
    - page: estimated page number (based on character offset)
    - paragraph: running paragraph count
    """
    chunks = []
    start = 0
    paragraph_number = 1
    page_length = 1800  # Approximate characters per "page"

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Normalize whitespace
        cleaned_chunk = re.sub(r'\s+', ' ', chunk).strip()

        current_page = (start // page_length) + 1
        current_paragraph = paragraph_number

        chunks.append({
            "chunk_text": cleaned_chunk,
            "doc_id": doc_id,
            "page": current_page,
            "paragraph": current_paragraph
        })

        start += chunk_size - overlap
        paragraph_number += 1

    return chunks
