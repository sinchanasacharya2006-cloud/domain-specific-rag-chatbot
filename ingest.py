"""
ingest.py
---------
Loads PDFs from the `data/` folder, splits them into ~500-token chunks,
embeds them with all-MiniLM-L6-v2, and stores them in a persistent
ChromaDB collection called "rag_collection".

Run this once (or whenever you change the source documents):
    python ingest.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PDF_FOLDER = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_collection"


def load_documents(pdf_folder):
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            print(f"Loading: {file}")
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())
    print(f"\n✅ Total pages loaded: {len(documents)}")
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Total chunks: {len(chunks)}")
    return chunks


def store_chunks(chunks):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk.page_content],
            metadatas=[chunk.metadata],
        )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_PATH}/'")
    return collection


if __name__ == "__main__":
    docs = load_documents(PDF_FOLDER)
    chunks = chunk_documents(docs)
    store_chunks(chunks)