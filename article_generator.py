import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
import streamlit as st
from pydantic import BaseModel
from dotenv import load_dotenv
import re
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent
from crewai_tools import BrowserbaseLoadTool


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

# browser_api_key = os.getenv("BROWSERBASE_API_KEY")
# browser_proj_id = os.getenv("BROWSERBASE_PROJECT_ID")

# duckduckgo_tool = BrowserbaseLoadTool(browser_api_key,browser_proj_id)

llm = ChatOpenAI(
    model="gpt-4o-mini",  
    temperature=0.1,
    api_key=api_key
)

researcher_backstory = """
You are a distinguished research analyst with expertise in emerging technologies 
and market trends. With a Ph.D. in Technology Foresight and years of experience at leading 
think tanks, you excel at identifying breakthrough innovations and their potential impact 
on industries and society.
"""

writer_backstory = """
You are an award-winning technology journalist known for making complex subjects 
accessible to everyone. With a background in both science communication and 
creative writing, you've mastered the art of transforming technical insights into 
compelling narratives. Your articles have been featured in leading tech publications, 
and you have a knack for finding the human angle in every story.
"""

News_Researcher = Agent(
    role="Senior Research Analyst",
    goal="Conduct comprehensive analysis of emerging trends and technologies in {topic}",
    verbose=True,
    memory=True,
    backstory=researcher_backstory,
    # tools=[duckduckgo_tool],
    llm=llm,
    allow_delegation=True
)

News_Writer = Agent(
    role='Technology Journalist',
    goal='Create engaging and informative articles about {topic} for a general audience',
    verbose=True,
    memory=True,
    backstory=writer_backstory,
    # tools=[duckduckgo_tool],
    llm=llm,
    allow_delegations=False
)

Research_task = Task(
    description="""
    Perform an in-depth analysis of {topic} by addressing the following key areas:
    1. Overview of the current landscape and recent breakthroughs
    2. Identification of key players and innovations driving progress
    3. Analysis of market impact and emerging opportunities
    4. Exploration of the main technical challenges and potential solutions
    5. Future predictions and expected developments

    Ensure the research includes detailed examples, relevant statistics (where applicable), 
    and insights from credible sources.
    """,
    expected_output="A detailed research report covering all major aspects of the topic, with specific examples and data points",
    agent=News_Researcher,
    llm=llm
)

Write_task = Task(
    description="""
    Using the research findings, craft an engaging article that:
    1. Starts with an attention-grabbing introduction
    2. Simplifies complex ideas using relatable analogies
    3. Incorporates practical examples and real-world use cases
    4. Explores the potential implications for various industries
    5. Concludes with a forward-looking perspective

    The article should have a balanced tone, being optimistic but realistic about the challenges. 
    Present the content in clear, well-organized markdown, with appropriate headings and sections for easy reading.
    """,
    expected_output="A well-structured markdown article of 12-14 paragraphs with clear sections and engaging content",
    agent=News_Writer,
    async_execution=False,
    output_file='blog.md',
    llm=llm
)

st.set_page_config(page_title="CrewAI Article Generator", page_icon="📝", layout="wide")
st.title("📝 AI Article Generator")
st.markdown("""
    This application uses CrewAI with OpenAI to generate comprehensive articles on any topic. 
    The system employs two AI agents:
    - A research analyst who gathers and analyzes information
    - A technology journalist who crafts the final article
    """)

topic = st.text_input("Enter a topic:", placeholder="e.g., Quantum Computing, AI, Web3")

if st.button("Generate Article"):
    if not topic:
        st.warning("Please enter a topic before generating the article.")
    else:
        progress_bar = st.progress(0)
        
        crew = Crew(
            agents=[News_Researcher, News_Writer],
            tasks=[Research_task, Write_task],
            process=Process.sequential,
        )

        with st.spinner(f"Researching and writing about '{topic}'..."):
            progress_bar.progress(50)
            result = crew.kickoff(inputs={'topic': topic})
        
        progress_bar.progress(100)
        
        st.subheader("Generated Article:")
        st.markdown(result)

        st.download_button(
            label="Download Article",
            data=str(result),
            file_name=f"{topic.replace(' ', '_').lower()}_article.md",
            mime="text/plain"
        )