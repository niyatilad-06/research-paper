import streamlit as st
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# -------------------------
# NLTK Resource Setup
# -------------------------
@st.cache_resource
def download_nltk_resources():
    """Ensure required NLTK resources are available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

download_nltk_resources()
stop_words = set(stopwords.words("english"))

# -------------------------
# Model Loading
# -------------------------
@st.cache_resource
def load_summarizer():
    """Load lightweight summarization model."""
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

@st.cache_resource
def load_sentence_model():
    """Load sentence transformer model for semantic similarity."""
    return SentenceTransformer("all-MiniLM-L6-v2")

summarizer = load_summarizer()
sentence_model = load_sentence_model()

# -------------------------
# Utility Functions
# -------------------------
def extract_text_from_pdf(file, max_pages=3):
    """Extract text from PDF (limited to first few pages)."""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text += page.get_text("text")
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""

def extract_keywords(text, num_keywords=10):
    """Extract keywords using RAKE."""
    try:
        text = text[:5000]  # Limit text length for faster extraction
        rake = Rake(stopwords=stop_words)
        rake.extract_keywords_from_text(text)
        return rake.get_ranked_phrases()[:num_keywords]
    except Exception as e:
        st.error(f"❌ Keyword extraction failed: {e}")
        return []

def split_into_chunks(text, max_words=250):
    """Split long text into smaller chunks for summarization."""
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk, words = [], [], 0
    for sent in sentences:
        word_count = len(sent.split())
        if words + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk, words = [], 0
        current_chunk.append(sent)
        words += word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_text(text, max_length=150, min_length=50):
    """Summarize long text by combining smaller summaries."""
    try:
        chunks = split_into_chunks(text)
        summaries = []
        for chunk in chunks[:5]:  # summarize first 5 chunks max
            summary = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(summary)
        return " ".join(summaries)
    except Exception as e:
        st.error(f"❌ Summarization failed: {e}")
        return ""

def find_related_papers(papers, top_k=2):
    """Suggest related papers based on summary similarity."""
    try:
        summaries = [p["summary"] for p in papers if p["summary"].strip()]
        embeddings = sentence_model.encode(summaries, convert_to_tensor=True)
        related_results = []
        for i in range(len(papers)):
            scores = util.cos_sim(embeddings[i], embeddings)[0]
            scores[i] = -1  # exclude self
            top_indices = scores.topk(top_k).indices.tolist()
            related = [(papers[j]["title"], float(scores[j])) for j in top_indices]
            related_results.append({"paper": papers[i]["title"], "related": related})
        return related_results
    except Exception as e:
        st.error(f"❌ Related paper suggestion failed: {e}")
        return []

# -------------------------
# Streamlit Interface
# -------------------------
st.set_page_config(page_title="Research Paper Assistant", layout="wide")

st.title("📚 Research Paper Assistant")
st.markdown(
    "Easily **summarize research papers**, **extract keywords**, and **find related topics**. "
    "Upload your PDFs and get insights instantly ⚡"
)

# Sidebar controls
num_keywords = st.sidebar.slider("Number of Keywords", 5, 20, 10)
summary_length = st.sidebar.slider("Summary Length", 100, 400, 150)
st.sidebar.info("⚡ Only the first 3 pages of each PDF are processed for faster results.")

# File upload
uploaded_files = st.file_uploader(
    "Upload one or more research papers (PDF)",
    type="pdf",
    accept_multiple_files=True
)

papers = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.success(f"✅ Uploaded: {uploaded_file.name}")

        with st.spinner("🔍 Extracting text..."):
            text = extract_text_from_pdf(uploaded_file)

        if not text.strip():
            st.error("⚠️ No readable text found in this file.")
            continue

        with st.expander("🧾 Preview extracted text (first 1000 characters)"):
            st.write(text[:1000] + "..." if len(text) > 1000 else text)

        with st.spinner("🧠 Extracting keywords..."):
            keywords = extract_keywords(text, num_keywords=num_keywords)

        with st.spinner("✍️ Generating summary..."):
            summary = summarize_text(text, max_length=summary_length)

        st.markdown(f"### 📄 {uploaded_file.name}")
        st.subheader("Summary")
        st.write(summary if summary else "No summary generated.")
        st.subheader("Keywords")
        st.write(", ".join(keywords) if keywords else "No keywords found.")

        papers.append({"title": uploaded_file.name, "summary": summary})

    # Related paper suggestions
    if len(papers) > 1:
        st.header("🔍 Suggested Related Papers")
        related = find_related_papers(papers)
        for item in related:
            st.markdown(f"**{item['paper']}** is related to:")
            for r in item["related"]:
                st.write(f"- {r[0]} (similarity: {r[1]:.2f})")

st.markdown("---")
st.caption("Developed by **Niyati Lad** | Enrollment No: 12202130501046")
