import streamlit as st
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Download stopwords (run once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------
# Functions
# -------------------------
def extract_text_from_pdf(file):
    """Extract text from uploaded PDF file."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text


def extract_keywords(text, num_keywords=10):
    """Extract keywords using RAKE."""
    rake = Rake(stopwords=stop_words)
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()[:num_keywords]
    return keywords


def chunk_text(text, chunk_size=800):
    """Split text into smaller chunks for summarization."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def summarize_text(text, max_length=120, min_length=40):
    """Summarize text in chunks for long PDFs."""
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    chunks = list(chunk_text(text))
    summaries = []
    for chunk in chunks[:3]:  # only summarize first 3 chunks for speed
        summary = summarizer(
            chunk, max_length=max_length, min_length=min_length, do_sample=False
        )[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)


def find_related_papers(papers, top_k=2):
    """Find related papers using semantic similarity of summaries."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([p["summary"] for p in papers], convert_to_tensor=True)

    related_results = []
    for i in range(len(papers)):
        scores = util.cos_sim(embeddings[i], embeddings)[0]
        scores[i] = -1  # exclude self
        top_indices = scores.topk(top_k).indices.tolist()
        related = [(papers[j]["title"], float(scores[j])) for j in top_indices]
        related_results.append({"paper": papers[i]["title"], "related": related})

    return related_results

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Research Paper Assistant", layout="wide")

st.title("📑 Research Paper Assistant Dashboard")
st.markdown("Upload PDFs to extract **keywords**, **summarize**, and **suggest related papers**.")

# Sidebar options
num_keywords = st.sidebar.slider("Number of keywords", 5, 20, 10)
summary_length = st.sidebar.slider("Summary max length", 50, 200, 120)

# Upload PDFs
st.header("📂 Upload PDFs")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files", type="pdf", accept_multiple_files=True
)

papers = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.success(f"Uploaded: {uploaded_file.name}")
        text = extract_text_from_pdf(uploaded_file)

        # Extract Keywords
        st.subheader(f"🔑 Extract Keywords - {uploaded_file.name}")
        keywords = extract_keywords(text, num_keywords=num_keywords)
        st.write(", ".join(keywords))

        # Summarize PDF
        st.subheader(f"📝 Summarize PDF - {uploaded_file.name}")
        summary = summarize_text(text, max_length=summary_length)
        st.write(summary)

        # Store for related paper suggestion
        papers.append({"title": uploaded_file.name, "summary": summary})

    # Suggest Related Papers
    if len(papers) > 1:
        st.header("🔍 Suggested Related Papers")
        related = find_related_papers(papers, top_k=2)
        for item in related:
            st.markdown(f"**{item['paper']}** is related to:")
            for r in item['related']:
                st.write(f"- {r[0]} (similarity: {r[1]:.2f})")
