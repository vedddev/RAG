import streamlit as st
import tempfile
import time
import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------
# Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, say:
"I could not find the answer in the uploaded documents."

Previous Conversation:
{chat_history}

Context:
{context}

Question:
{input}

Answer:
"""
)

# -----------------------------
# UI
# -----------------------------
st.title("Multi PDF RAG Chatbot")

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

# Show uploaded files
if uploaded_files:
    st.subheader("Uploaded PDFs")

    for pdf in uploaded_files:
        st.write(f"{pdf.name}")

# -----------------------------
# Create Vector Store
# -----------------------------
def create_vector_embeddings():

    documents = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.getvalue())
            tmp_file.flush()

            loader = PyPDFLoader(tmp_file.name)

            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    final_documents = text_splitter.split_documents(
        documents
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    st.session_state.vectors = FAISS.from_documents(
        final_documents,
        embeddings
    )

    st.session_state.document_chunks = final_documents

    st.success(
        f" Vector DB Created with {len(final_documents)} chunks"
    )

# -----------------------------
# Buttons
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Create Embeddings"):

        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            create_vector_embeddings()

with col2:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# -----------------------------
# User Query
# -----------------------------
user_prompt = st.text_input(
    "Ask a question from the uploaded PDFs"
)

if user_prompt:

    if "vectors" not in st.session_state:
        st.warning(
            "Please upload PDFs and create embeddings first."
        )
        st.stop()

    history = "\n".join(
        [
            f"Human: {chat['question']}\nAI: {chat['answer']}"
            for chat in st.session_state.chat_history
        ]
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = st.session_state.vectors.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.process_time()

    response = retrieval_chain.invoke(
        {
            "input": user_prompt,
            "chat_history": history
        }
    )

    answer = response["answer"]

    st.subheader("Answer")
    st.write(answer)

    response_time = time.process_time() - start

    st.caption(
        f"Response Time: {response_time:.2f} seconds"
    )

    new_chat = {
        "question": user_prompt,
        "answer": answer
    }

    if (
        len(st.session_state.chat_history) == 0
        or st.session_state.chat_history[-1] != new_chat
    ):
        st.session_state.chat_history.append(
            new_chat
        )

    # -------------------------
    # Chat History
    # -------------------------
    if st.session_state.chat_history:

        st.subheader("Chat History")

        for chat in reversed(
            st.session_state.chat_history
        ):

            st.markdown(
                f"**You:** {chat['question']}"
            )

            st.markdown(
                f"**AI:** {chat['answer']}"
            )

            st.divider()

    # -------------------------
    # Similarity Search Results
    # -------------------------
    with st.expander(
        "Document Similarity Search"
    ):

        for i, doc in enumerate(
            response["context"]
        ):

            st.markdown(
                f"### Chunk {i+1}"
            )

            st.write(
                doc.page_content[:1000]
            )

            st.divider()