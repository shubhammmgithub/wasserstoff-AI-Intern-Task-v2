import streamlit as st
import requests
from datetime import datetime

# -------------------------
# Config & State
# -------------------------
BACKEND_URL = "http://127.0.0.1:5000"  # change to deployed URL on Hugging Face
st.set_page_config(page_title="Wasserstoff Document AI", page_icon="📄")

# -------------------------
# Title
# -------------------------
st.title("📄 Wasserstoff Document AI")
st.markdown("Upload documents and perform semantic search on their content using AI.")

# -------------------------
# Session State Init
# -------------------------
for key in ["query", "search_result", "synthesized_answer", "upload_info", "clear_upload", "theme_summary"]:
    st.session_state.setdefault(key, "" if key.endswith("answer") or key == "query" else [])

# -------------------------
# Upload Section
# -------------------------
st.header("📄 Upload Documents")

if st.button("🧹 Clear Uploaded Files"):
    st.session_state.upload_info = []
    st.session_state.clear_upload = True
    st.experimental_rerun()

if not st.session_state.clear_upload:
    uploaded_files = st.file_uploader(
        "Choose PDF, Image, or DOCX files",
        type=["pdf", "jpg", "png", "jpeg", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("📂 Upload and Process"):
        for file in uploaded_files:
            with st.spinner(f"Processing {file.name}..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files={"file": file}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.upload_info.append(data)
                        st.success(f"✅ {file.name} uploaded and processed successfully!")
                    else:
                        st.error(f"❌ Upload failed for {file.name}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend server is not running. Please start the Flask server.")

    if st.session_state.upload_info:
        st.subheader("🗌 Uploaded Document Info:")
        for data in st.session_state.upload_info:
            with st.container():
                st.write("**Filename:**", data.get('filename', 'N/A'))
                st.write("**Extracted Text Snippet:**")
                st.code(data.get('extracted_text_snippet', 'N/A'), language="text")
                st.write("**Total Chunks:**", data.get('total_chunks', 0))
                st.write("**Status:**", data.get('status', 'N/A'))
                st.markdown("---")

# -------------------------
# Search Section
# -------------------------
st.header("🔍 Semantic Search")

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("Enter your query", value=st.session_state.query, key="query_input")
with col2:
    if st.button("🔁 Reset Search"):
        st.session_state.query = ""
        st.session_state.search_result = []
        st.session_state.synthesized_answer = ""
        st.experimental_rerun()

num_results = st.slider("Number of results to display", min_value=1, max_value=10, value=3)

if st.button("🔎 Search") and query:
    with st.spinner("Searching documents..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/search",
                json={"query": query, "top_k": num_results}
            )
            if response.status_code == 200:
                result_data = response.json()
                st.session_state.query = query
                st.session_state.search_result = result_data.get("results", [])
                st.session_state.synthesized_answer = result_data.get("synthesized_answer", "")

                # Synthesized Answer
                if st.session_state.synthesized_answer:
                    st.subheader("🧠 Synthesized Answer:")
                    st.markdown(st.session_state.synthesized_answer)
                    st.download_button(
                        label="💾 Download Answer",
                        data=st.session_state.synthesized_answer,
                        file_name=f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

                # Top Results
                if st.session_state.search_result:
                    st.subheader("📌 Top Results:")
                    for i, res in enumerate(st.session_state.search_result, 1):
                        with st.expander(f"📄 Result {i} (Score: {res.get('score', 0):.4f})", expanded=i==1):
                            st.markdown(f"**Document:** `{res.get('doc_id', 'N/A')}`")
                            st.markdown(f"**Page:** `{res.get('page', 'N/A')}`")
                            st.markdown(f"**Paragraph:** `{res.get('paragraph', 'N/A')}`")
                            st.markdown("**Content:**")
                            st.write(res.get('chunk', ''))
            else:
                st.error(f"❌ Search failed: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Backend server is not running. Please start the Flask server.")

# -------------------------
# Theme Extraction Section
# -------------------------
st.header("🧠 Extract Common Themes")

if st.button("✨ Identify Themes"):
    if not st.session_state.get("query"):
        st.warning("Please perform a search first to identify themes")
    elif not st.session_state.get("search_result"):
        st.warning("No search results available for theme extraction")
    else:
        with st.spinner("Identifying common themes..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/themes",
                    headers={"Content-Type": "application/json"},
                    json={"query": st.session_state.query}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    theme_summary = data.get("theme_summary")
                    themes_by_doc = data.get("themes_by_document", {})
                    
                    if theme_summary:
                        st.session_state.theme_summary = theme_summary
                        st.subheader("🔍 Common Themes")
                        st.markdown(theme_summary)
                        
                        st.download_button(
                            label="💾 Download Themes",
                            data=theme_summary,
                            file_name=f"themes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    if themes_by_doc:
                        st.subheader("📚 Document-Specific Themes")
                        for doc_id, theme_data in themes_by_doc.items():
                            with st.expander(f"📄 {doc_id}"):
                                if "error" in theme_data:
                                    st.error(theme_data["error"])
                                elif "theme_summary" in theme_data:
                                    st.markdown(theme_data["theme_summary"])
                else:
                    st.error(f"❌ Theme extraction failed: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend server is not running")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# -------------------------
# Results Download Section
# -------------------------
if st.session_state.get("search_result"):
    st.header("💾 Download Results")
    
    export_format = st.radio("Export format:", ["CSV", "Text"])
    
    if st.button("📥 Export All Results"):
        if export_format == "CSV":
            csv_data = StringIO()
            writer = csv.DictWriter(csv_data, fieldnames=["score", "doc_id", "page", "paragraph", "chunk"])
            writer.writeheader()
            for res in st.session_state.search_result:
                writer.writerow(res)
            
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data.getvalue(),
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            text_data = f"Search Results for: {st.session_state.query}\n\n"
            for res in st.session_state.search_result:
                text_data += f"\nDocument: {res['doc_id']}\nPage: {res.get('page', 'N/A')}\nParagraph: {res.get('paragraph', 'N/A')}\nScore: {res['score']:.4f}\n\n{res['chunk']}\n{'='*50}\n"
            
            st.download_button(
                label="⬇️ Download Text",
                data=text_data,
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            #d