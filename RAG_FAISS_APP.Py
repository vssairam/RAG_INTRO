"""
Local PDF RAG app using LangChain + FAISS + OpenAI API.

What it does:
1. Reads Maths.pdf.
2. Splits the PDF text into chunks.
3. Creates embeddings locally with a Hugging Face embedding model.
4. Saves the vectors in a local FAISS index folder.
5. Retrieves relevant chunks from that local FAISS database for each question.
6. Sends only the retrieved context and question to the OpenAI API to generate an answer.

Run:
    python rag_app.py
Rebuild the index from the PDF:
    python rag_app.py --rebuild

Install packages first:
    pip install -U langchain-community langchain-text-splitters langchain-huggingface langchain-openai faiss-cpu sentence-transformers pypdf

Set your OpenAI API key in PowerShell before running:
    $env:OPENAI_API_KEY="your-api-key-here"

Security:
- Do not paste your real API key into this source file.
- Do not share your API key or commit it to a public repository.
"""

from pathlib import Path
import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------- SETTINGS --------------------

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Maths.pdf"
VECTOR_DB_PATH = BASE_DIR / "faiss_maths_index"

# Embeddings are generated locally on your computer.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# This model is accessed through the OpenAI API.
OPENAI_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 400
RETRIEVAL_K = 5


# -------------------- LOAD EMBEDDING MODEL --------------------

def create_embeddings():
    """Create a local text embedding model; no embedding API key is needed."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# -------------------- BUILD OR LOAD VECTOR DATABASE --------------------

def build_vector_db(embeddings):
    """Read the PDF, split it, and create a persistent local FAISS index."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {PDF_PATH.name}.\n"
            "Place Maths.pdf in the same folder as this Python file."
        )

    print(f"\nReading PDF: {PDF_PATH.name}")
    pages = PyPDFLoader(str(PDF_PATH)).load()

    # Remove pages that contain no extractable text.
    pages = [page for page in pages if page.page_content.strip()]
    if not pages:
        raise ValueError(
            "No selectable text was extracted from the PDF. "
            "It may be a scanned/image-only PDF and need OCR first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    print(f"Extracted {len(pages)} pages and created {len(chunks)} chunks.")
    print("Creating local embeddings and FAISS index. This may take a while...")

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(str(VECTOR_DB_PATH))

    print(f"FAISS index saved locally in: {VECTOR_DB_PATH}")
    return vector_db


def load_or_build_vector_db(embeddings, force_rebuild=False):
    """Load the saved FAISS index, or build it if it does not exist."""
    index_file = VECTOR_DB_PATH / "index.faiss"
    metadata_file = VECTOR_DB_PATH / "index.pkl"

    if not force_rebuild and index_file.exists() and metadata_file.exists():
        print(f"Loading existing local FAISS index from: {VECTOR_DB_PATH}")
        # Only load a FAISS index you created yourself and trust.
        return FAISS.load_local(
            str(VECTOR_DB_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    return build_vector_db(embeddings)


# -------------------- CREATE THE RAG QUESTION-ANSWER CHAIN --------------------

def create_rag_chain(vector_db):
    """Retrieve relevant PDF chunks locally and answer with the OpenAI API."""
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant answering questions about the provided Maths PDF.

Use the retrieved context as the primary and only source for claims about what
the PDF says. If the answer is not present in the context, say:
"I could not find that information in the Maths PDF."
Explain calculations step by step when the context supports them.
Do not invent formulas, examples, or page references.

Retrieved context:
{context}

Question:
{question}

Answer:"""
    )

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        # Reads OPENAI_API_KEY from the environment by default.
    )

    answer_chain = prompt | llm | StrOutputParser()

    def ask_question(question):
        # The vector search happens against the local FAISS database.
        documents = retriever.invoke(question)

        if not documents:
            return "I could not find relevant information in the Maths PDF.", []

        context_parts = []
        sources = []

        for doc in documents:
            page_number = doc.metadata.get("page")
            page_label = (
                str(page_number + 1)
                if isinstance(page_number, int)
                else "unknown"
            )
            context_parts.append(f"[PDF page {page_label}]\n{doc.page_content}")
            sources.append(page_label)

        context = "\n\n---\n\n".join(context_parts)

        # The question and retrieved passages are sent to OpenAI for generation.
        answer = answer_chain.invoke(
            {"context": context, "question": question}
        )

        unique_sources = list(dict.fromkeys(sources))
        return answer, unique_sources

    return ask_question


# -------------------- CHAT LOOP --------------------

def main():
    force_rebuild = "--rebuild" in sys.argv

    print("=" * 60)
    print("MATHS PDF RAG APP")
    print("LangChain + Local FAISS + Local Embeddings + OpenAI API")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\nStartup error: OPENAI_API_KEY is not set.")
        print("In PowerShell, run this command (replace the placeholder):")
        print('$env:OPENAI_API_KEY="your-api-key-here"')
        print("Then run: python rag_app.py")
        sys.exit(1)

    try:
        print("\nLoading local embedding model...")
        embeddings = create_embeddings()

        vector_db = load_or_build_vector_db(
            embeddings,
            force_rebuild=force_rebuild,
        )

        ask_question = create_rag_chain(vector_db)

        print("\nReady! Questions retrieve passages from the local FAISS database.")
        print(f"Answer generation uses the OpenAI API model: {OPENAI_MODEL}")
        print("Commands: type 'exit' or 'quit' to stop.")
        print("To recreate the index later, run: python rag_app.py --rebuild")

        while True:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not question:
                continue

            try:
                answer, source_pages = ask_question(question)
                print("\nAnswer:")
                print(answer)

                if source_pages:
                    print("\nRetrieved PDF page(s): " + ", ".join(source_pages))
            except Exception as exc:
                print(f"\nCould not answer this question: {exc}")
                print(
                    "Check your OpenAI API key, API access/billing, model name, "
                    "and internet connection."
                )

    except Exception as exc:
        print(f"\nStartup error: {exc}")
        print("Check the setup steps in the instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
