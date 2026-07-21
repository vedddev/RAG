from crewai import Agent,LLM
from tools import yt_tool

import os
from dotenv import load_dotenv
load_dotenv()

llm = LLM(
    model="ollama/llama3.2:1b",
    base_url="http://localhost:11434"
)



blog_reasearch=Agent(
    role='Blog Reasearcher from youtube videos',
    goal='get the relevant video content for the {topic} from Yt channle',
    llm=llm,
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data Science,Machine Learning and GEN AI and providing suggestions"
    ),
    tools=[yt_tool],
    allow_delegation=True
)

## create Writer agent with YT tool

blog_writer=Agent(
    role='Blog writer',
    goal='Narrate compelling tech stories about the video {topic} from YT video',
    llm=llm,
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    allow_delegation=True
)