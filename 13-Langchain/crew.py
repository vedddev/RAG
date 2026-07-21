from crewai import Crew
from agents import blog_reasearch,blog_writer
from tasks import research_task,write_task


crew = Crew(
    agents=[blog_reasearch, blog_writer],
    tasks=[research_task, write_task],
    verbose=True,
    planning=False,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

result=crew.kickoff(inputs={
        "topic": "Lookism Manhwa"
    })
print(result)