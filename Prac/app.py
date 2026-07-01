import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)



load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')

api_key=st.text_input("Enter the Groq api Key",type="password")
## Page

st.set_page_config(
    page_title='Normal Bot'
)
st.title('Normal bot')
st.write("Enter Pdf")

## Session state

if "store" not in st.session_state:
    st.session_state.store={}
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore=None
    
    
## LLM
if api_key:
    llm=ChatGroq(
        model='llama-3.3-70b-versatile',
        groq_api_key=api_key
    )

    ## Embeddings

    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    ## Session id

    session_id=st.text_input(
        "Session_ID",
        value="default_session"
    )

    ## Pdf upload

    uploded_files=st.file_uploader(
        "Upload Pdfs",
        type='pdf',
        accept_multiple_files=True
    )

    ## Process of PDF

    if st.button('Process'):
        if not uploded_files:
            st.warning('upload the pdf')
            st.stop()
            
        
        documents=[]
        
        with st.spinner("Process Pdf..."):
            
            for uploded_file in uploded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix='.pdf'
                ) as tmp_file:
                    tmp_file.write(uploded_file.getvalue())
                    tmp_file.flush()
                    
                    loader=PyPDFLoader(
                        tmp_file.name
                    )
                    
                    docs=loader.load()
                    
                    documents.extend(docs)
                    
            splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            splits=splitter.split_documents(
                documents
            )
            vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=embeddings,
                
            )
            st.session_state.vectorstore=vectorstore
            st.success(
                f"Process {len(splits)} chunks."
            )
            
    ## chat Function 
    def get_session_history(session:str)->BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session]=ChatMessageHistory()
        return st.session_state.store[session]
        
    ## Qusetion
    User_input=st.text_input('Ask a question')

    if User_input:
        if st.session_state.vectorstore is None:
            st.warning('Please process PDFs first.')
            st.stop()
        
        retriever = (
            st.session_state.vectorstore
            .as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
        )

        ## History and retriever
        contextualize_q_system_prompt = """
        Given a chat history and the latest user question
        which might reference context in the chat history,
        formulate a standalone question.

        Do NOT answer the question.
        Only reformulate it if necessary.
        """

        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever=create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt,
            
        )

        ## QA prompt

        system_prompt = """
        You are an assistant for question-answering tasks.

        Use the retrieved context to answer.

        If you don't know the answer,
        say you don't know.

        Keep answers concise.
        {context}
        """

        qa_prompt = (
                ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            system_prompt
                        ),
                        MessagesPlaceholder(
                            "chat_history"
                        ),
                        (
                            "human",
                            "{input}"
                        ),
                    ]
                )
            )


        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
        converstional_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
            )

        response=converstional_rag_chain.invoke(
            {"input":User_input},
            config={
                "configurable": {
                            "session_id": session_id
                        }
            }
        )

        st.subheader('Answer')

        st.write(response['answer'])

    ## show chat history

    history=get_session_history(
        session_id
    )
    st.subheader("chat History")

    for msg in history.messages:

            if msg.type == "human":
                st.markdown(
                    f"**You:** {msg.content}"
                )

            else:
                st.markdown(
                    f"**AI:** {msg.content}"
                )

    ## show session store

    with st.expander(
            "Session Information"
        ):
            st.write(
                st.session_state.store
            )
            
else:
    st.warning("Enter the Groq api key sir >-<")
