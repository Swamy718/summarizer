import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from spacy.lang.en import English
import re
from PyPDF2 import PdfReader
from io import StringIO
from typing import Optional, Tuple, List, Dict
import logging
import heapq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize stopwords
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    stop_words = set(stopwords.words("english"))
except Exception as e:
    stop_words = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'])
    logger.warning(f"Failed to load NLTK stopwords: {str(e)}")
    st.warning("Using default stopwords list. Full functionality may be limited.")

# Load spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    st.error("Failed to load spaCy model. Please ensure it's installed correctly.")
    nlp = None

@st.cache_data
def extractive_summarizer(text: str, max_sentences: int = 5) -> str:
    try:
        sentences = sent_tokenize(text)
        word_frequencies = {}

        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for word in words:
                if word not in stop_words:
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1

        sentence_scores = {}
        for sentence in sentences:
            for word, freq in word_frequencies.items():
                if word in sentence.lower():
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq

        summary_sentences = heapq.nlargest(max_sentences, sentence_scores, key=sentence_scores.get)
        summary_sentences.sort(key=lambda x: text.find(x))
        return " ".join(summary_sentences)
    except Exception as e:
        logger.error(f"Error in extractive summarizer: {str(e)}")
        return "Error generating summary"

@st.cache_data
def process_text(text: str) -> Tuple[str, int]:
    try:
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text, len(text.split())
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return text, 0

def extract_text_from_pdf(file_path: str) -> str:
    try:
        pdf_reader = PdfReader(file_path)
        return " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def count_words(text: str) -> int:
    return len(text.split())

def main():
    st.title("Text Summarization Tool")
    st.write("""
    Upload or paste your text below to generate a concise summary.
    The tool uses extractive summarization to create a meaningful summary of your content.
    """)

    st.header("Input Text")
    input_type = st.radio("Choose input method", ["Paste Text", "Upload File"])
    text = ""

    if input_type == "Paste Text":
        text = st.text_area("Paste your text here:", height=300)
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf'])
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file.name)
            else:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                text = stringio.read()

    st.header("Settings")
    max_sentences = st.slider("Number of sentences in summary", 1, 10, 5)
    use_bullets = st.checkbox("Use bullet points", value=True)

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            processed_text, word_count = process_text(text)
            summary = extractive_summarizer(processed_text, max_sentences)
            if use_bullets:
                summary = "• " + summary.replace(". ", "\n• ")
            st.session_state.summary = summary

    if 'summary' in st.session_state:
        summary = st.session_state.get('summary', '')
        st.subheader("Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.button(f"Original text: {count_words(text)} words", type="primary")
        with col2:
            st.button(f"Summary: {count_words(summary)} words", type="primary")

        st.markdown("---")
        edited_summary = st.text_area("Edit the summary", value=summary, height=200)

        st.download_button("Download Summary", edited_summary, "summary.txt", "text/plain")


if __name__ == "__main__":
    main()
