# app.py
import streamlit as st
import fitz  # PyMuPDF
import re
import nltk
from rake_nltk import Rake
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Setup
# -------------------------
nltk.download('punkt')

st.set_page_config(page_title="Research Paper Assistant", layout="wide")
st.title("📄 Research Paper Assistant Dashboard")
st.info("Upload PDFs to extract keywords, summarize, and find related papers.")

# -------------------------
# Functions
# -------------------------
def extract_text_from_pdf(file):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def preprocess_for_keywords(text):
    # Keep text more natural (don’t over-clean)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_keywords(text, top_n=10):
    r = Rake()
    r.extract_keywords_from_text(text)
    ranked = r.get_ranked_phrases()
    return ranked[:top_n]

# Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # better summarizer
embedding_model = SentenceTransformer('all-mpnet-base-v2')

def summarize_text(text, max_len=150):
    sentences = nltk.sent_tokenize(text)
    chunks, chunk_size = [], 10  # fewer sentences per chunk for better summaries
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i+chunk_size])
        chunks.append(chunk)
    summaries = []
    for chunk in chunks[:3]:  # limit to 3 chunks to save time
        try:
            summary = summarizer(
                chunk,
                max_length=max_len,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary)
        except Exception:
            continue
    return " ".join(summaries) if summaries else "Summary not available."

def find_related_papers(paper_idx, embeddings, paper_names, top_n=3):
    similarities = util.pytorch_cos_sim(embeddings[paper_idx], embeddings)[0]
    related_idx = similarities.argsort(descending=True).tolist()[1:top_n+1]
    return [paper_names[i] for i in related_idx]

# -------------------------
# Sidebar Controls
# -------------------------
num_keywords = st.sidebar.slider("Number of keywords", 5, 20, 10)
summary_length = st.sidebar.slider("Summary max length", 50, 300, 150)

# -------------------------
# Section 1: Upload PDFs
# -------------------------
st.header("📂 Upload PDFs")
uploaded_files = st.file_uploader("Upload one or more PDF files", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded. Click 'Process PDFs' below to analyze.")

    if st.button("Process PDFs"):
        paper_texts_raw, paper_texts_clean, paper_names = [], [], []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            paper_texts_raw.append(text)
            paper_texts_clean.append(preprocess_for_keywords(text))
            paper_names.append(file.name)

        # Compute embeddings for related papers
        embeddings = embedding_model.encode(paper_texts_raw, convert_to_tensor=True)

        # -------------------------
        # Section 2: Extract Keywords
        # -------------------------
        st.header("🔑 Extract Keywords")
        for i, text in enumerate(paper_texts_clean):
            keywords = extract_keywords(text, top_n=num_keywords)
            st.subheader(f"📘 {paper_names[i]}")
            st.write(", ".join(keywords))

        # -------------------------
        # Section 3: Summarize PDF
        # -------------------------
        st.header("📝 Summarize PDF")
        for i, text in enumerate(paper_texts_raw):
            st.subheader(f"📘 {paper_names[i]}")
            summary = summarize_text(text, max_len=summary_length)
            st.write(summary)

        # -------------------------
        # Section 4: Add Multiple PDFs
        # -------------------------
        st.header("📂 Add Multiple PDFs")
        st.success("✅ You can upload multiple PDFs above. All files will be analyzed together.")

        # -------------------------
        # Section 5: Suggest Related Research Papers
        # -------------------------
        st.header("📚 Suggest Related Research Papers")
        if len(uploaded_files) < 2:
            st.warning("⚠️ Upload at least 2 PDFs to see related paper suggestions.")
        else:
            for i, name in enumerate(paper_names):
                related = find_related_papers(i, embeddings, paper_names)
                st.subheader(f"📘 {name}")
                st.write(", ".join(related))
else:
    st.info("Please upload at least one PDF to begin.")
