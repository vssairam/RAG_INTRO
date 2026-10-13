"""
Streamlit_RAG_App.py
Multi-PDF Retrieval-Augmented Generation (RAG) app using:
Streamlit + LangChain + OpenAI + FAISS.

Run:
    streamlit run Streamlit_RAG_App.py

Required environment variable:
    OPENAI_API_KEY

Optional environment variables:
    OPENAI_MODEL (default: gpt-4o-mini)
    OPENAI_EMBEDDING_MODEL (default: text-embedding-3-small)
"""

import os
from io import BytesIO
from typing import List

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# Load environment variables from a local .env file if one exists.
# OPENAI_API_KEY is still read from the environment; never hard-code it here.
load_dotenv()

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5


def extract_pdf_pages(uploaded_files) -> List[Document]:
    """Extract text from each PDF page and retain its original filename/page number."""
    pages: List[Document] = []

    for uploaded_file in uploaded_files:
        try:
            pdf_bytes = uploaded_file.getvalue()
            reader = PdfReader(BytesIO(pdf_bytes))

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    st.warning(f"Skipped encrypted PDF: {uploaded_file.name}")
                    continue

            for page_index, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    st.warning(
                        f"Could not read page {page_index + 1} of "
                        f"{uploaded_file.name}: {exc}"
                    )
                    continue

                if text.strip():
                    pages.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": uploaded_file.name,
                                "page": page_index + 1,  # Human-readable, 1-based page
                            },
                        )
                    )

        except Exception as exc:
            st.warning(f"Could not process {uploaded_file.name}: {exc}")

    return pages


def build_vector_store(uploaded_files, embeddings):
    """Extract, chunk, and embed all uploaded PDFs into one FAISS index."""
    page_documents = extract_pdf_pages(uploaded_files)
    if not page_documents:
        raise ValueError(
            "No readable text was found in the uploaded PDFs. "
            "If they are scanned PDFs, OCR is needed before indexing."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(page_documents)

    # Keep only chunks containing meaningful text.
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not chunks:
        raise ValueError("No text chunks were created from these PDFs.")

    # One combined FAISS vector store for every uploaded document.
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, chunks


def answer_question(question: str, vector_store, llm):
    """Retrieve relevant chunks and ask the model to answer from those sources."""
    retrieved_docs = vector_store.similarity_search(question, k=RETRIEVAL_K)

    if not retrieved_docs:
        return (
            "I could not find relevant information in the indexed PDFs.",
            [],
        )

    context_parts = []
    for index, doc in enumerate(retrieved_docs, start=1):
        filename = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", "Unknown")
        context_parts.append(
            f"[Source {index}: file={filename}, page={page}]\n"
            f"{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful document question-answering assistant. "
                "Answer using only the supplied PDF excerpts. If the excerpts "
                "do not contain enough information, clearly say that you could "
                "not find the answer in the uploaded documents. Do not invent "
                "facts or citations. You may refer to source labels such as "
                "'Source 1' in your explanation, but the application will show "
                "the verified file names and page numbers separately.",
            ),
            (
                "human",
                "Question:\n{question}\n\n"
                "Retrieved PDF excerpts:\n{context}\n\n"
                "Give a clear, concise answer grounded in the excerpts.",
            ),
        ]
    )
    response = (prompt | llm).invoke(
        {"question": question, "context": context}
    )
    answer = response.content
    if not isinstance(answer, str):
        answer = str(answer)

    # Deduplicate source references while preserving order.
    sources = []
    seen = set()
    for doc in retrieved_docs:
        filename = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", "Unknown")
        key = (filename, page)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "file": filename,
                    "page": page,
                    "excerpt": doc.page_content,
                }
            )

    return answer, sources


# ----------------------------- Session state ----------------------------- #
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None


# --------------------------------- UI ------------------------------------- #
st.title("📚 Multi-PDF RAG Assistant")
st.write(
    "Upload multiple PDFs, build one searchable FAISS vector database, "
    "and ask questions across all indexed documents."
)

with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="All selected PDFs are combined into one FAISS vector database.",
        key="pdf_uploader",
    )

    st.header("2. Indexing settings")
    st.write(f"Chunk size: **{CHUNK_SIZE} characters**")
    st.write(f"Chunk overlap: **{CHUNK_OVERLAP} characters**")
    st.write(f"Retrieved chunks per question: **{RETRIEVAL_K}**")

    build_clicked = st.button(
        "Build / Rebuild Vector Database",
        type="primary",
        use_container_width=True,
    )

    if st.button("Clear index and chat", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.chunk_count = 0
        st.session_state.indexed_files = []
        st.session_state.chat_history = []
        st.session_state.embeddings = None
        st.session_state.llm = None
        st.rerun()

if build_clicked:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        st.error(
            "OPENAI_API_KEY was not found in the environment. Set it and "
            "restart Streamlit before building the index."
        )
    elif not uploaded_files:
        st.warning("Upload at least one PDF before building the vector database.")
    else:
        try:
            with st.spinner(
                "Reading PDFs, creating chunks, generating embeddings, "
                "and building the FAISS index..."
            ):
                embedding_model_name = os.getenv(
                    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
                )
                chat_model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

                embeddings = OpenAIEmbeddings(
                    model=embedding_model_name,
                    api_key=api_key,
                )
                llm = ChatOpenAI(
                    model=chat_model_name,
                    temperature=0,
                    api_key=api_key,
                )

                vector_store, chunks = build_vector_store(
                    uploaded_files, embeddings
                )

                # Commit to session state only after the complete build succeeds.
                st.session_state.vector_store = vector_store
                st.session_state.chunk_count = len(chunks)
                st.session_state.indexed_files = sorted(
                    {file.name for file in uploaded_files}
                )
                st.session_state.embeddings = embeddings
                st.session_state.llm = llm
                st.session_state.chat_history = []

            st.success("Vector database built successfully.")
            st.rerun()

        except Exception as exc:
            st.error(
                "The vector database could not be built. Check your API key, "
                "API access/billing, internet connection, and PDF readability."
            )
            with st.expander("Technical details"):
                st.code(str(exc))


# Index status
if st.session_state.vector_store is not None:
    st.subheader("Index status")
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Total chunks indexed", st.session_state.chunk_count)
    metric_col2.metric("PDF files indexed", len(st.session_state.indexed_files))
    with st.expander("View indexed file names"):
        for filename in st.session_state.indexed_files:
            st.write(f"• {filename}")

    st.caption(
        "The current index remains available during this Streamlit session. "
        "If you change the uploaded files, click Build / Rebuild Vector Database "
        "to index the new selection."
    )
else:
    st.info("Upload your PDFs in the sidebar and build the vector database to begin.")

# Render chat history from session state.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources used for this answer"):
                for source_index, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**{source_index}. {source['file']} — page {source['page']}**"
                    )
                    st.caption(source["excerpt"][:1200])

# Ask questions only after indexing is complete.
if st.session_state.vector_store is not None:
    question = st.chat_input("Ask a question across your uploaded PDFs...")
    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the PDFs and preparing an answer..."):
                    answer, sources = answer_question(
                        question,
                        st.session_state.vector_store,
                        st.session_state.llm,
                    )
                st.markdown(answer)

                if sources:
                    st.markdown("**Sources**")
                    for source_index, source in enumerate(sources, start=1):
                        st.markdown(
                            f"{source_index}. **{source['file']}** — "
                            f"page **{source['page']}**"
                        )
                    with st.expander("View retrieved excerpts"):
                        for source_index, source in enumerate(sources, start=1):
                            st.markdown(
                                f"**{source_index}. {source['file']} — "
                                f"page {source['page']}**"
                            )
                            st.write(source["excerpt"][:1200])

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except Exception as exc:
                error_message = (
                    "I could not answer this question. Check your OpenAI API "
                    "key, API access/billing, model configuration, and internet "
                    "connection."
                )
                st.error(error_message)
                with st.expander("Technical details"):
                    st.code(str(exc))
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
