# rag-core/rag_core/ingestion/splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
