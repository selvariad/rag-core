# rag-core/rag_core/ingestion/loaders.py
from pathlib import Path
from typing import Protocol


class DocumentLoader(Protocol):
    def load(self, source: Path) -> list: ...


def get_loader(suffix: str):
    suffix = suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return _LangChainAdapter(PyPDFLoader)
    elif suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return _LangChainAdapter(Docx2txtLoader)
    elif suffix in (".txt", ".text"):
        from langchain_community.document_loaders import TextLoader
        return _LangChainAdapter(TextLoader)
    elif suffix == ".md":
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        return _LangChainAdapter(UnstructuredMarkdownLoader)
    elif suffix in (".html", ".htm"):
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        return _LangChainAdapter(UnstructuredHTMLLoader)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


class _LangChainAdapter:
    def __init__(self, loader_cls):
        self._loader_cls = loader_cls

    def load(self, source: Path) -> list:
        loader = self._loader_cls(str(source))
        return loader.load()
