import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# ===========================
# Load API Key
# ===========================
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error("NVIDIA_API_KEY not found in .env")
    st.stop()

# ===========================
# Initialize LLM
# ===========================
llm = ChatNVIDIA(
    model="meta/llama-3.2-3b-instruct",
    api_key=api_key,
    max_completion_tokens=4096
)

# ===========================
# Prompt
# ===========================
prompt = ChatPromptTemplate.from_template("""
You are an expert document assistant.

Use ONLY the information provided in the context.

If the answer is not present in the context, say:
"I couldn't find the answer in the provided documents."

Context:
{context}

Question:
{input}

Answer:
""")

# ===========================
# Create Vector Store
# ===========================
def vector_embedding():
    if "vectors" not in st.session_state:

        with st.spinner("Loading documents..."):

            embeddings = NVIDIAEmbeddings(api_key=api_key)

            loader = PyPDFDirectoryLoader("./us_census")
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=250
            )

            final_documents = splitter.split_documents(docs)
            print("Number of chunks:", len(final_documents))

            vectors = FAISS.from_documents(
                final_documents,
                embeddings
            )

            st.session_state.embeddings = embeddings
            st.session_state.vectors = vectors

        st.success("Vector Database Created Successfully!")

# ===========================
# Streamlit UI
# ===========================
st.title("📄 NVIDIA RAG Chatbot")

if st.button("Create Vector Database"):
    vector_embedding()

# ===========================
# Test LLM
# ===========================
if st.button("Test LLM"):
    with st.spinner("Testing LLM..."):
        response = llm.invoke("Say Hello")
        st.write(response.content)

# ===========================
# User Question
# ===========================
question = st.text_input("Ask a question about your documents")

if question:

    if "vectors" not in st.session_state:
        st.warning("Please create the vector database first.")
        st.stop()

    retriever = st.session_state.vectors.as_retriever(
        search_kwargs={"k": 5}
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.time()

    with st.spinner("Generating Answer..."):
        response = retrieval_chain.invoke(
            {"input": question}
        )

    end = time.time()

    st.success(f"Response Time: {end-start:.2f} sec")

    st.write("### Answer")
    st.write(response["answer"])

    with st.expander("Retrieved Documents"):
        for i, doc in enumerate(response["context"], start=1):
            st.markdown(f"### Document {i}")
            st.write(doc.page_content)
            st.divider()