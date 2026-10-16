"""
File: TechMate/app/rag/retrieval.py
Description: Vector search retriever interface for query lookup.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.rag.ingestion import get_embeddings_model
from app.config import FAISS_INDEX_PATH


def retrieve_relevant_chunks(query: str, top_k: int = 4) -> List[Document]:
    """Retrieves top_k relevant context chunks from the local FAISS index."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return []

    try:
        embeddings = get_embeddings_model()
        vector_db = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        docs = vector_db.similarity_search(query, k=top_k)
        return docs
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        return []