import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain,LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

## Set up the Streamlit app
st.set_page_config(page_title="Math Solver",page_icon="🧮")
st.title("Text to math Problem Solver using Groq model")

groq_api_key=st.sidebar.text_input(label="Groq_API_Key",type="password")

if not groq_api_key:
    st.info("Please add your Groq APPI key to continue")
    st.stop()
    
llm=ChatGroq(api_key=groq_api_key,
            model="llama-3.3-70b-versatile")


Wikipedia_Wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=Wikipedia_Wrapper.run,
    description="Useful for answering factual questions using Wikipedia."
)

## initializeing the math tools

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathmatical expression needs to be provided"
)

prompt="""
Your a agent tasked for solving users mathemetical question.Logically arrive at the solution and provide a detailed explanation and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=['question'],
    template=prompt
)

## Combine all the tools into chain
reasoning_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)

reasoning_tool=Tool(
    name="Reasoning",
    func=reasoning_chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

##initialize the agents

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi,I'm a math chatbot who cna answer all your maths questions."}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
    
## func genrate response
def genrate_reponse(question):
    response=assistant_agent.invoke({'input':question})
    return response

# Let's start interaction
question=st.text_area("Enter your question:","I have 5 bananas and 7 grapes.I eat 2 bananas")

if st.button("find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response = assistant_agent.invoke({"input": question},callbacks=[st_cb])
            response = response["output"]
            st.session_state.messages.append({"role":'assistant',"content":response})
            st.success(response)
        
    else:
        st.warning("Please enter the question")