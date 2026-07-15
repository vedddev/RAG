import re
import streamlit as st
import validators

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# ---------------- Streamlit UI ---------------- #

st.set_page_config(
    page_title="LangChain Summarizer",
    page_icon="🦜",
    layout="wide"
)

st.title("🦜 LangChain Summarizer")
st.write("Summarize YouTube videos or Websites using Hugging Face Llama 3.1")

# ---------------- Sidebar ---------------- #

with st.sidebar:
    st.header("Settings")

    hf_api_key = st.text_input(
        "Hugging Face API Token",
        type="password"
    )

# ---------------- Input ---------------- #

url = st.text_input(
    "Enter YouTube or Website URL"
)

# ---------------- Prompt ---------------- #

prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are an expert summarizer.

Provide a well-structured summary in approximately 300 words.

Content:
{text}
"""
)

# ---------------- Summarize Button ---------------- #

if st.button("Summarize"):

    if not hf_api_key:
        st.error("Please enter your Hugging Face API Token.")
        st.stop()

    if not url:
        st.error("Please enter a URL.")
        st.stop()

    if not validators.url(url):
        st.error("Please enter a valid URL.")
        st.stop()

    try:

        # ---------------- Load Documents ---------------- #

        with st.spinner("Loading document..."):

            # ----------- YouTube ----------- #

            if "youtube.com" in url or "youtu.be" in url:

                # Extract Video ID

                if "youtu.be/" in url:
                    video_id = url.split("/")[-1].split("?")[0]

                else:
                    match = re.search(r"v=([^&]+)", url)

                    if not match:
                        st.error("Invalid YouTube URL")
                        st.stop()

                    video_id = match.group(1)

                transcript = YouTubeTranscriptApi().fetch(video_id)

                text = " ".join(
                    chunk.text for chunk in transcript
                )

                docs = [
                    Document(page_content=text)
                ]

            # ----------- Website ----------- #

            else:

                loader = UnstructuredURLLoader(
                    urls=[url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    }
                )

                docs = loader.load()

        # ---------------- LLM ---------------- #

        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=hf_api_key,
            temperature=0.5,
            max_new_tokens=1024
        )
        llm=ChatHuggingFace(llm=llm)

        # ---------------- Chain ---------------- #

        chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",
            prompt=prompt
        )

        # ---------------- Generate Summary ---------------- #

        with st.spinner("Generating Summary..."):

            result = chain.invoke(
                {"input_documents": docs}
            )

        st.success("Summary Generated Successfully!")

        st.subheader("Summary")

        st.write(result["output_text"])

    except Exception as e:
        st.error(f"Error: {str(e)}")