from PIL import Image
import pytesseract
import pdfplumber
import docx
import os

def extract_text_from_file(filepath):
    """
    Extracts text from image, PDF, or DOCX file using OCR and parsing tools.
    Supports .jpg, .jpeg, .png, .pdf, and .docx.
    """
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext in ['.jpg', '.jpeg', '.png']:
            return pytesseract.image_to_string(Image.open(filepath))

        elif ext == '.pdf':
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()

        elif ext == '.docx':
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])

        else:
            return "❌ Unsupported file type."

    except Exception as e:
        return f"❌ Error extracting text: {str(e)}"
