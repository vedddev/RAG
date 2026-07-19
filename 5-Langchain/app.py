import streamlit as st
from dotenv import load_dotenv
from langsmith import Client
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun,
)
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)

load_dotenv()

st.set_page_config(page_title="LangChain Search Agent")
st.title("🔎 AI Search Agent")


# Sidebar

groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)


# Tools

search = DuckDuckGoSearchRun(name='Search')

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=300
    )
)

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=300
    )
)

tools = [search, wiki, arxiv]


# Chat History

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# User Input

prompt = st.chat_input("Ask me anything...")

if prompt:

    if not groq_api_key:
        st.warning("Please enter your Groq API Key.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-20b",
        temperature=0
    )

    # Pull standard ReAct prompt
    client=Client()
    # react_prompt = client.("hwchase17/react")

    # Create agent
    agent = create_agent(
    model=llm,
    tools=tools,
    # system_prompt="You are a helpful AI assistant that can use tools to answer user questions."
)

    # Agent Executor
    # agent_executor = AgentExecutor(
    #     agent=agent,
    #     tools=tools,
    #     verbose=True,
    #     handle_parsing_errors=True
    # )

    with st.spinner("Thinking..."):

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

        answer = response["messages"][-1].content

        st.write(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )