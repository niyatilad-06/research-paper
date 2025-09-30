import streamlit as st
import fitz
import re
import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Download stopwords at runtime
nltk.download('stopwords')
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

@st.cache_resource
def load_models():
    summarizer_model = pipeline("summarization", model="t5-small")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return summarizer_model, embedding_model

summarizer, embed_model = load_models()

def summarize_text(text, max_len=150):
    return summarizer(text, max_length=max_len, min_length=50, do_sample=False)[0]['summary_text']

def find_related_papers(paper_idx, embeddings, paper_names, top_n=3):
    similarities = util.pytorch_cos_sim(embeddings[paper_idx], embeddings)[0]
    related_idx = similarities.argsort(descending=True).tolist()[1:top_n+1]
    return [paper_names[i] for i in related_idx]

# -------------------------
# UI Layout
# -------------------------
st.set_page_config(page_title="Research Paper Assistant", layout="wide")

st.sidebar.title("📄 Upload & Settings")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

num_keywords = st.sidebar.slider("Number of keywords", 5, 20, 10)
summary_length = st.sidebar.slider("Summary max length", 50, 300, 150)

st.title("📄 Research Paper Assistant Dashboard")
st.info("Upload multiple PDFs to extract keywords, summarize, and find related papers.")

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
            st.markdown(f"## 📄 {paper_names[i]}")
            
            with st.expander("Show extracted text"):
                st.write(text[:5000])  # show first 5000 chars
            
            # Display keywords
            keywords = extract_keywords(text, top_n=num_keywords)
            st.markdown("**Keywords:**")
            st.write(", ".join(keywords))
            
            # Display summary
            summary = summarize_text(text, max_len=summary_length)
            st.markdown("**Summary:**")
            st.write(summary)
            
            # Display related papers
            related = find_related_papers(i, embeddings, paper_names)
            st.markdown("**Related Papers:**")
            st.write(", ".join(related))
            
            st.markdown("---")
