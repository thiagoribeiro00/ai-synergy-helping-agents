import streamlit as st
import pandas as pd
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq

#Langchain, Groq, CrewAI, Pandas, Streamlit

def main():
    # Configurações da barra lateral
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )

    # Inicialização do modelo LLM
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name=model
    )

    # Interface do usuário com Streamlit
    st.title('Content Enhancement Web Application')
    multiline_text = """
    This application allows users to input URLs, scrape website content, enhance it using AI, and transfer the enhanced content for review. 
    Each agent has a specific role in this process to ensure a smooth workflow.
    """
    st.markdown(multiline_text, unsafe_allow_html=True)

    # Exibição do logo Groq
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image(r'../public/groqcloud_darkmode.png')

    # Definição dos agentes
    Scraping_Agent = Agent(
        role='Content_Scraping_Agent',
        goal="Scrape content from the user-provided URLs and extract relevant SEO keywords using Semrush API.",
        backstory="You are an expert in web scraping and SEO. Your task is to gather content from the specified URLs and extract SEO keywords for further processing.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Enhancement_Agent = Agent(
        role='Content_Enhancement_Agent',
        goal="Rewrite the scraped content using Claude AI, incorporating SEO keywords and user-defined prompts.",
        backstory="You are an AI content enhancer. Your role is to take the scraped content and enhance it using AI, ensuring it aligns with SEO best practices and user specifications.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Transfer_Agent = Agent(
        role='Data_Transfer_Agent',
        goal="Transfer the enhanced content to GatherContent for review and approval.",
        backstory="You are responsible for transferring the enhanced content into GatherContent. Ensure that the data is formatted correctly for seamless integration.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # Entrada de dados do usuário
    urls_input = st.text_area("Enter the URLs to scrape, one per line:")
    
    if urls_input:
        urls = urls_input.strip().split("\n")

        task_scrape_content = Task(
            description=f"""
            Scrape content from the following URLs and extract relevant SEO keywords using Semrush API:
            {urls}
            """,
            agent=Scraping_Agent,
            expected_output="Scraped content and SEO keywords extracted from the provided URLs."
        )

        task_enhance_content = Task(
            description=f"""
            Utilize Claude AI to rewrite the scraped content, incorporating SEO keywords
            and any user-defined prompts.
            Here is the scraped content from the URLs:

            {urls}
            """,
            agent=Enhancement_Agent,
            expected_output="Enhanced content with SEO keywords and user-defined prompts."
        )

        task_transfer_content = Task(
            description="Transfer the enhanced content into GatherContent for review and approval.",
            agent=Transfer_Agent,
            expected_output="Enhanced content ready for transfer."
        )

        crew = Crew(
            agents=[Scraping_Agent, Enhancement_Agent, Transfer_Agent],
            tasks=[task_scrape_content, task_enhance_content, task_transfer_content],
            verbose=2
        )

        result = crew.kickoff()

        st.write(result)

if __name__ == "__main__":
    main()
