"""
File: TechMate/app/rag/ingestion.py
Description: Processes documents (PDF/TXT) and indexes them into FAISS safely.
"""

import os
from typing import List, Optional
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.config import GEMINI_API_KEY, FAISS_INDEX_PATH

from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# EMBEDDINGS MODEL — CACHED SINGLETON
# ============================================================
#
# HuggingFaceEmbeddings(...) loads an actual sentence-transformer
# model into memory. Previously this was called fresh every time
# ingest_document() or retrieve_relevant_chunks() ran, meaning
# every single Study Agent question paid the cost of reloading
# the model before it could even search. Cached here so it's
# loaded once per process and reused.

_embeddings_model: Optional[HuggingFaceEmbeddings] = None


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFaceEmbeddings instance, creating it
    on first call only. Both ingestion and retrieval import this
    same function, so they always use the same embedding model —
    critical for FAISS, since indexing and querying with
    different embedding models silently produces meaningless
    results (incompatible vector spaces).
    """

    global _embeddings_model

    if _embeddings_model is None:

        _embeddings_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    return _embeddings_model


def ingest_document(file_path: str) -> bool:
    """Loads a PDF or TXT file, splits it into chunks, and safely saves to FAISS vector DB."""
    try:
        # Case-insensitive extension check — file_path.endswith(".pdf")
        # alone would miss "Document.PDF" on case-preserving
        # filesystems (e.g. Windows).
        lower_path = file_path.lower()

        if lower_path.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif lower_path.endswith(".txt") or lower_path.endswith(".md"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError("Unsupported file format. Use PDF, TXT, or MD.")

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = get_embeddings_model()
        
        # Check both the directory path and the actual index file existence
        index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(index_file):
            vector_db = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            vector_db.add_documents(chunks)
        else:
            # Ensure parent directory exists before saving a brand-new index
            os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
            vector_db = FAISS.from_documents(chunks, embeddings)

        vector_db.save_local(FAISS_INDEX_PATH)
        return True
    except Exception as e:
        print(f"Ingestion error: {str(e)}")
        return False