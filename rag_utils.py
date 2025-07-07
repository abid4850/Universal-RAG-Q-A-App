import os
import pandas as pd
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from typing import List
import streamlit as st

SUPPORTED_FORMATS = ['.txt', '.csv', '.pdf', '.docx']

def load_document(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == '.txt':
            try:
                loader = TextLoader(file_path)
                return loader.load()
            except Exception as e:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = ''.join([next(f) for _ in range(5)])
                raise RuntimeError(f"TextLoader failed: {e}\nFirst lines of file:\n{lines}")
        elif ext == '.csv':
            # Try to load as CSV with pandas first for validation
            try:
                pd.read_csv(file_path)
                loader = CSVLoader(file_path)
                return loader.load()
            except Exception as e_csv:
                # If not a valid CSV, fallback to loading as plain text
                try:
                    loader = TextLoader(file_path)
                    return loader.load()
                except Exception as e_txt:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = ''.join([next(f) for _ in range(5)])
                    raise RuntimeError(f"CSVLoader failed: {e_csv}\nTextLoader also failed: {e_txt}\nFirst lines of file:\n{lines}")
        elif ext == '.pdf':
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        raise RuntimeError(f"Error loading {file_path}: {e}")

def create_vector_store(docs, embedding_model="nomic-embed-text", chunk_size=512, chunk_overlap=50, persist_dir=None):
    # Sidebar option for chunk size (default 400, min 200, max 1000)
    chunk_size = st.sidebar.slider("Text chunk size (lower = faster, less context)", 200, 1000, 400, step=100)
    chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 200, 40, step=20)
    embeddings = OllamaEmbeddings(model=embedding_model)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    if persist_dir:
        vector_store = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_dir)
    else:
        vector_store = Chroma.from_documents(split_docs, embeddings)
    return vector_store

def get_retriever(vector_store, k=5) -> VectorStoreRetriever:
    return vector_store.as_retriever(search_kwargs={"k": k})
