"""
Core RAG pipeline: load PDF -> chunk -> embed -> store in Chroma -> retrieve -> generate.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_store"

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based only on the
provided context. If the answer is not in the context, say you don't know —
do not make anything up.

Context:
{context}

Question: {question}

Answer:"""


def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Load a PDF and split it into overlapping text chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks, persist_directory: str = CHROMA_DIR):
    """Embed chunks and store them in a Chroma vector database."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


def load_existing_vectorstore(persist_directory: str = CHROMA_DIR):
    """Reload a previously persisted Chroma store (skip re-embedding)."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def format_docs(docs):
    """Join retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore, groq_api_key: str, model_name: str = "llama-3.1-8b-instant"):
    """Wire retriever + LLM into an LCEL RAG chain. Returns (chain, retriever)."""
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def answer_question(qa_chain, question: str):
    """Run a query through the RAG chain and return answer + source pages."""
    chain, retriever = qa_chain
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    sources = sorted({doc.metadata.get("page", "?") for doc in source_docs})
    return answer, sources