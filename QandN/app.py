import streamlit as st
# import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os 
from  dotenv import load_dotenv
load_dotenv()

# google_api_key=os.getenv('GOOGLE_API_KEY')

# Langsimth Tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot with GOOGLEGEMINI"

#prompt Templaet

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Please response to user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,model,temperature,max_tokens):
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    answer = chain.invoke(
        {"question": question}
    )

    return answer

# title for app
st.title('Athena-mini-0.1')

st.sidebar.title('Settings')
google_api_key=st.sidebar.text_input('Enter your Gemini API key',type="password")


model=st.sidebar.selectbox('Select an Gemini Model',['gemini-2.5-flash',"gemini-2.5-flash-lite"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=2048, value=512)

##Main interface
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,google_api_key,model,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please Provide the query.")