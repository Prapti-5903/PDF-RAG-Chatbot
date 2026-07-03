# 📄 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that lets you upload a PDF and
ask questions about it in natural language. The app retrieves the most
relevant chunks of the document and grounds the LLM's answer in that context,
reducing hallucination compared to a plain chatbot.

## How it works

1. **Load & Chunk** — the PDF is parsed with `PyPDFLoader` and split into
   overlapping chunks with `RecursiveCharacterTextSplitter`.
2. **Embed & Store** — each chunk is embedded using a local
   `sentence-transformers` model and stored in a **ChromaDB** vector store.
3. **Retrieve** — on each question, the top-k most similar chunks are
   retrieved via semantic search.
4. **Generate** — the retrieved chunks + question are passed to an LLM
   (**Groq's Llama 3.1**, free tier) through a grounded prompt, and the
   answer is returned along with the source page numbers.

```
PDF → chunks → embeddings → ChromaDB
                                 │
User question ──► retriever ────┤
                                 ▼
                    LLM (Groq / Llama 3.1) → grounded answer + sources
```

## Tech stack

- **LangChain** — orchestration
- **ChromaDB** — vector database
- **HuggingFace `all-MiniLM-L6-v2`** — embeddings (runs locally, free)
- **Groq API (Llama 3.1)** — fast, free-tier LLM inference
- **Streamlit** — chat UI

## Setup

```bash
git clone <your-repo-url>
cd rag-pdf-chatbot
pip install -r requirements.txt
cp .env.example .env   # add your free Groq API key from console.groq.com
streamlit run app.py
```

## Usage

1. Paste your Groq API key in the sidebar (or set it in `.env`).
2. Upload a PDF.
3. Click **Process PDF** and wait for indexing to finish.
4. Ask questions in the chat box — answers include the source page numbers.

## Possible extensions

- Swap Chroma for FAISS or Pinecone for larger-scale storage.
- Add multi-document support and per-document filtering.
- Add conversational memory (multi-turn follow-up questions).
- Swap Groq for a local model via Ollama for a fully offline pipeline.

## Why this project

Built to demonstrate core RAG concepts: chunking strategy, embedding models,
vector similarity search, and prompt grounding — the foundational building
blocks behind production retrieval systems.
