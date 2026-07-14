from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv

load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')
model=ChatGroq(model='llama-3.3-70b-versatile',groq_api_key=groq_api_key)

#create Prompt template

system_template="Translate into following language{language}:"
prompt_template=ChatPromptTemplate.from_messages(
    
    [("system",system_template),
    ("user","{text}")]
    
)

parser=StrOutputParser()

## Create chain
chain=prompt_template|model|parser

# Define App

app=FastAPI(
    title='Langchain Server',
    version=1.0,
    description='Simple api server using langchian runnable interfaces',
)

add_routes(
    app,
    chain,
    path='/chain'
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8000)