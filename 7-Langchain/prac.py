import traceback
import validators
import streamlit as st
import requests

from bs4 import BeautifulSoup
from urllib.parse import urlparse,parse_qs

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq



# Streamlit
st.set_page_config(
    page_title='Summarizer'
)

st.title('Context creator')


groq_api_key=st.sidebar.text_input(
    'Enter the groq key',
    type='password'
)

url=st.text_input(
    'Enter the url'
)


#LLM

prompt=PromptTemplate(
    input_variables=['text'],
    template="""
    Summarize the following content like a professional summarizer.
    
    {text}
    """
    
)

# Functions

def get_video_id(link):
    if 'youtu.be' in link:
        return link.split('/')[-1].split('?')[0]
    return parse_qs(urlparse(link).query)['v'][0]

def load_youtube(url):
    video_id=get_video_id(url)
    api=YouTubeTranscriptApi()
    transcript=api.fetch(
        video_id,
        languages=['hi','en']
        
    ).to_raw_data()
    
    text=" ".join(item['text'] for item in transcript)
    return [Document(page_content=text)]

def load_website(url):
    headers={
        "User-Agent":"Mozilla/5.0"
    }
    response=requests.get(
        url,
        headers=headers,
        timeout=20
    )
    
    soup=BeautifulSoup(response.text,'html.parser')
    
    for tag in soup(['script','style']):
        tag.decompose()
    text=soup.get_text(separator='')
    
    return [Document(page_content=text)
]
#button.
if st.button('Click me'):
    if not groq_api_key:
        st.error("Please enter Groq API Key.")
        st.stop()
        
    if not validators.url(url):
        st.error('Please enter the Url')
        st.stop()
        
    try:

        llm = ChatGroq(
            api_key=groq_api_key,
            model="openai/gpt-oss-20b",
            temperature=0
        )

        with st.spinner("Loading content..."):

            if "youtube.com" in url or "youtu.be" in url:
                docs = load_youtube(url)
            else:
                docs = load_website(url)

        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=prompt
        )

        with st.spinner("Generating summary..."):

            result = chain.invoke(docs)

        if isinstance(result, dict):

            st.success(result["output_text"])

        else:

            st.success(result)

    except Exception as e:

        st.error(str(e))
        st.code(traceback.format_exc())