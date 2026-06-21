import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>
    Question:{input}
    
    """
)

def create_vectore_embeddings():
    if "vectors" not in st.session_state:

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        st.session_state.loader = PyPDFDirectoryLoader(
            "D:\\Langchain\\QandN-Part2\\research_papers"
        )

        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(
                st.session_state.docs[:50]
            )
        )

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vectore_embeddings()
    st.write("Vectore Database is ready")
    
import time

if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("Please click Document Embedding first.")
        st.stop()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.process_time()

    response = retrieval_chain.invoke(
        {"input": user_prompt}
    )

    st.write(response["answer"])

    print(
        f"Response Time: {time.process_time()-start}"
    )

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------")