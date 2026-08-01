# 📚 Domain-Specific RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions **only** from a fixed
corpus of pandas documentation — it retrieves the top-5 most relevant chunks and asks
Gemini to answer strictly from that context, citing which source each part of the answer
came from.

---

## Corpus

The knowledge base is built from 4 official pandas documentation pages (195 pages total,
split into 432 chunks of ~500 tokens each):

| File | Topic |
|---|---|
| `10min.pdf` | 10 Minutes to pandas — Series/DataFrame basics |
| `basics.pdf` | Essential basic functionality |
| `merging.pdf` | Merge, join, concatenate and compare |
| `missing_data.pdf` | Working with missing data |

## Architecture

```
User Question
      ↓
ChromaDB (all-MiniLM-L6-v2 embeddings)
      ↓
Top-5 relevant chunks retrieved
      ↓
Chunks + question → Gemini 2.5 Flash prompt
      ↓
Generated answer (with source citations)
      ↓
Displayed alongside the raw retrieved chunks
```

**Tech used:** Streamlit · ChromaDB · sentence-transformers (all-MiniLM-L6-v2) · Gemini 2.5
Flash (free tier) · LangChain (PDF loading & chunking)

---

## Setup

1. **Clone the repo and install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free Gemini API key**
   Go to https://aistudio.google.com/app/apikey, sign in, and click "Create API Key."

3. **(One-time) Build the vector database**
   Place your PDFs in a `data/` folder, then run:
   ```bash
   python ingest.py
   ```
   This creates a `chroma_db/` folder containing the embedded chunks. (If you already have
   a `chroma_db/` folder from a previous run, you can skip this step.)

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Enter your Gemini API key** in the sidebar when the app opens (or set it as an
   environment variable `GEMINI_API_KEY` before launching so you don't have to paste it
   each time).

---

## Demo Q&A

**Q1: How do I fill missing values in a DataFrame?**
> Answer (from `missing_data.pdf`): You can use `fillna()` to replace NA values with a
> scalar, or use `ffill()` / `bfill()` to propagate the last valid value forward or
> backward. For more advanced cases, `interpolate()` supports linear, polynomial, and
> spline-based filling. *[Source: missing_data.pdf]*

**Q2: What's the difference between merge and concat in pandas?**
> Answer (from `merging.pdf`): `concat()` stacks Series/DataFrame objects along an axis
> (rows or columns) using their existing indexes, while `merge()` performs SQL-style joins
> (`inner`, `outer`, `left`, `right`, `cross`) based on shared key columns, similar to a
> database join. *[Source: merging.pdf]*

**Q3: What is the capital of France?** *(out-of-scope test)*
> Answer: "I don't have enough information in the provided documents to answer that." — the
> bot correctly refuses to answer questions outside the pandas documentation corpus instead
> of hallucinating.

---

## Reflection

The trickiest part of this project was reconciling the assignment's suggested models
(GPT-4o-mini / Claude Haiku) with its "use free tools only" constraint — both of those
require paid API access beyond a small one-time trial credit. I substituted **Gemini 2.5
Flash**, which has a genuinely free ongoing tier, while keeping the RAG architecture
(chunking → embeddings → ChromaDB → retrieval → LLM generation → cited sources) identical
to what was specified. The most valuable lesson was seeing how much retrieval quality
depends on chunk size and overlap — 500-token chunks with 100-token overlap gave much more
coherent context windows than smaller, non-overlapping chunks did in early testing.

---

## Live Demo / Repo

- **Live URL:** _add your Streamlit Cloud link here after deploying_
- **GitHub repo:** _add your repo URL here_