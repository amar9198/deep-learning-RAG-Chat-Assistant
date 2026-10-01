import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="Deep Learning RAG Chat Assistant", page_icon="🤖", layout="wide")

st.write(
    "AI-powered Retrieval-Augmented Generation assistant "
    "for answering questions from your documents."
)

# ---------- Cached resources ----------
@st.cache_resource
def get_embeddings():
    return MistralAIEmbeddings(model="mistral-embed")


@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)


def get_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def get_retriever():
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


# ---------- Indexing ----------
def index_pdf(uploaded_file, replace_existing: bool):
    """Load the uploaded PDF, split it, embed it and store it in Chroma."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    for p in pages:
        p.metadata["source"] = uploaded_file.name

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    vs = get_vectorstore()
    if replace_existing:
        vs.delete_collection()
        vs = get_vectorstore()

    # Add in batches so we can show progress and avoid API limits
    progress = st.progress(0.0, text="Embedding chunks...")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        vs.add_documents(chunks[i : i + batch_size])
        progress.progress(
            min((i + batch_size) / len(chunks), 1.0),
            text=f"Embedding chunks... {min(i + batch_size, len(chunks))}/{len(chunks)}",
        )
    progress.empty()
    return len(pages), len(chunks)


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_books" not in st.session_state:
    st.session_state.indexed_books = []


# ---------- Sidebar: upload ----------
with st.sidebar:
    st.header("📖 Upload your book")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    replace = st.checkbox(
        "Replace existing books",
        value=False,
        help="If checked, the old knowledge base is wiped before adding this book.",
    )

    if st.button("Process book", type="primary", disabled=uploaded is None):
        with st.spinner("Reading and indexing the book..."):
            try:
                n_pages, n_chunks = index_pdf(uploaded, replace)
                if replace:
                    st.session_state.indexed_books = []
                st.session_state.indexed_books.append(uploaded.name)
                st.success(f"Indexed {n_pages} pages into {n_chunks} chunks.")
            except Exception as e:
                st.error(f"Failed to process the book: {e}")

    if st.session_state.indexed_books:
        st.subheader("Books in this session")
        for name in st.session_state.indexed_books:
            st.write(f"• {name}")

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------- Main: chat ----------
st.title("📚 RAG Book Assistant ")
st.caption("Upload a PDF in the sidebar, then ask questions about it.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']} – page {s['page']}**")
                    st.caption(s["text"])

query = st.chat_input("Ask a question about the book...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                docs = get_retriever().invoke(query)
                context = "\n\n".join(d.page_content for d in docs)
                final_prompt = PROMPT.invoke({"context": context, "question": query})
                answer = get_llm().invoke(final_prompt).content
            except Exception as e:
                docs, answer = [], f"Something went wrong: {e}"

        st.markdown(answer)

        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", 0) + 1,
                "text": d.page_content[:300] + "...",
            }
            for d in docs
        ]
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s['source']} – page {s['page']}**")
                    st.caption(s["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )