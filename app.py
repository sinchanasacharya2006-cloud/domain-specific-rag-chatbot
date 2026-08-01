import os
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.generativeai as genai

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Domain-Specific RAG Chatbot")
st.caption("Ask questions about pandas — answers are generated only from the loaded documentation (10 Minutes to pandas, Basics, Merging, Missing Data).")

# ---------------------------------------------------------
# Gemini API key setup
# ---------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Settings")
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = st.text_input("Enter your Gemini API key", type="password")
        st.caption("Get a free key at https://aistudio.google.com/app/apikey")
    else:
        st.success("Gemini API key loaded from environment ✅")

    st.markdown("---")
    st.markdown("**Corpus:** pandas documentation (195 pages, 432 chunks)")
    st.markdown("**Embedding model:** all-MiniLM-L6-v2")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("**LLM:** Gemini 2.5 Flash")

if not GEMINI_API_KEY:
    st.warning("Please enter a Gemini API key in the sidebar to enable answer generation. You can still browse retrieved chunks without it.")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-flash-latest")

# ---------------------------------------------------------
# Connect to ChromaDB (auto-build from data/ if not present yet)
# ---------------------------------------------------------
@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    try:
        collection = client.get_collection(name="rag_collection", embedding_function=embedding_function)
        if collection.count() > 0:
            return collection
    except Exception:
        pass

    # Collection doesn't exist yet (e.g. first run on a fresh deployment) - build it now
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with st.spinner("First-time setup: building the vector database from source PDFs... this can take a minute."):
        documents = []
        pdf_folder = "data"
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pdf_folder, file))
                documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        collection = client.get_or_create_collection(name="rag_collection", embedding_function=embedding_function)
        for i, chunk in enumerate(chunks):
            collection.add(
                ids=[str(i)],
                documents=[chunk.page_content],
                metadatas=[chunk.metadata],
            )

    return collection

collection = load_collection()

# ---------------------------------------------------------
# RAG prompt construction
# ---------------------------------------------------------
def build_prompt(question, chunks_with_meta):
    context_blocks = []
    for i, (doc, meta) in enumerate(chunks_with_meta, start=1):
        source = meta.get("source", "unknown source")
        page = meta.get("page", "?")
        context_blocks.append(f"[Source {i}: {source}, page {page}]\n{doc}")

    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant that answers questions ONLY using the provided context from pandas documentation.

Rules:
- Only use information found in the context below.
- If the answer is not contained in the context, respond exactly with: "I don't have enough information in the provided documents to answer that."
- When you answer, refer to sources using their [Source N] label so the user knows where the information came from.
- Be concise and clear.

Context:
{context_text}

Question: {question}

Answer:"""
    return prompt


def generate_answer(question, chunks_with_meta):
    prompt = build_prompt(question, chunks_with_meta)
    response = model.generate_content(prompt)
    return response.text


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
question = st.text_input("Ask a question about pandas:", placeholder="e.g. How do I fill missing values in a DataFrame?")

col1, col2 = st.columns([1, 5])
with col1:
    search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not question:
        st.info("Please type a question first.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            results = collection.query(query_texts=[question], n_results=5)

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        chunks_with_meta = list(zip(docs, metadatas))

        # --- Generated answer ---
        if GEMINI_API_KEY:
            with st.spinner("Generating answer with Gemini..."):
                try:
                    answer = generate_answer(question, chunks_with_meta)
                    st.subheader("💬 Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Gemini generation failed: {e}")
        else:
            st.info("Add a Gemini API key in the sidebar to generate a natural-language answer. Showing retrieved chunks only below.")

        # --- Retrieved chunks / citations ---
        st.subheader("📄 Retrieved Sources")
        for i, (doc, meta) in enumerate(chunks_with_meta, start=1):
            source = meta.get("source", "unknown source")
            page = meta.get("page", "?")
            with st.expander(f"Source {i}: {source} (page {page})"):
                st.write(doc)
