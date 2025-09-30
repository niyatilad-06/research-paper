# -------------------------
# app.py
# -------------------------
import streamlit as st
import fitz  # PyMuPDF
import re
import nltk
from rake_nltk import Rake
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Download NLTK stopwords at runtime
# -------------------------
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# -------------------------
# Functions
# -------------------------
def extract_text_from_pdf(file):
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def extract_keywords(text, top_n=10):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()[:top_n]

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_models():
    summarizer_model = pipeline("summarization", model="t5-small")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return summarizer_model, embedding_model

summarizer, embed_model = load_models()

def summarize_text(text):
    return summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

def find_related_papers(paper_idx, embeddings, paper_names, top_n=3):
    similarities = util.pytorch_cos_sim(embeddings[paper_idx], embeddings)[0]
    related_idx = similarities.argsort(descending=True).tolist()[1:top_n+1]  # exclude self
    return [paper_names[i] for i in related_idx]

# -------------------------
# Dashboard
# -------------------------
st.title("📄 Research Paper Assistant Dashboard")
st.info("Upload multiple PDFs to extract keywords, summarize, and find related papers.")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.success(f"{len(uploaded_files)} files uploaded.")
    if st.button("Process PDFs"):
        paper_texts = []
        paper_names = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            clean_text = preprocess_text(text)
            paper_texts.append(clean_text)
            paper_names.append(file.name)
        embeddings = embed_model.encode(paper_texts, convert_to_tensor=True)
        for i, text in enumerate(paper_texts):
            st.markdown(f"### 📄 {paper_names[i]}")
            with st.expander("Show extracted text"):
                st.write(text[:5000])
            keywords = extract_keywords(text)
            st.write("**Keywords:**", keywords)
            summary = summarize_text(text)
            st.write("**Summary:**", summary)
            related = find_related_papers(i, embeddings, paper_names)
            st.write("**Related Papers:**", related)
            st.markdown("---")
