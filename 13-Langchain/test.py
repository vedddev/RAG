from crewai import Agent, Task, Crew, LLM
import os
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

agent = Agent(
    role="Assistant",
    goal="Answer questions",
    backstory="Helpful AI",
    llm=llm,
    verbose=True,
)

task = Task(
    description="Say hello.",
    expected_output="A greeting",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    planning=False,
    memory=False,
    cache=False,
)

print(crew.kickoff())