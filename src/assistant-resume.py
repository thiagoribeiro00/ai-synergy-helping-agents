import streamlit as st
import pandas as pd
from crewai import Agent, Task, Crew
import PyPDF2 as pdf
from langchain_groq import ChatGroq

def main():
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )

    llm = ChatGroq(
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name=model
    )

    # Interface do usuário com Streamlit
    st.title('Resume Generator Web Application')
    multiline_text = """
    This Application helps you improve your resume based on job descriptions.
    """
    st.markdown(multiline_text, unsafe_allow_html=True)

    # Exibição do logo Groq
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image(r'../public/groqcloud_darkmode.png')

    # Definição dos agentes
    Scraping_Agent = Agent(
        role='Job_Analysis_Agent',
        goal="Analyze the job description to extract key responsibilities, requirements, desired skills, and knowledge.",
        backstory="You are an expert in job analysis. Your task is to analyze the job description and extract the main responsibilities, requirements, and desired skills.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Enhancement_Agent = Agent(
        role='Curriculum_Structuring_Agent',
        goal="Structure the resume based on the extracted information from the job description, including a summary/objective statement.",
        backstory="You are a resume structuring expert. Your role is to take the extracted information and structure the resume accordingly, ensuring a clear summary/objective statement.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Transfer_Agent = Agent(
        role='Improvement_Suggestions_Agent',
        goal="Provide improvement suggestions for the resume based on the job requirements, apply these suggestions to enhance match probability with the job description, and incorporate relevant keywords.",
        backstory="You are a resume improvement expert. Your task is to provide suggestions to improve the resume based on the job requirements, apply these suggestions to the resume, and incorporate key words from the job description to increase the match probability and measure the experiences.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # Entrada de dados do usuário
    job_description = st.text_area("Job description:")
    uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload your resume")

    submit = st.button("Submit")

    if submit:
        if uploaded_file is not None and job_description:
            # Extrair texto do arquivo PDF fornecido
            def input_pdf_text(uploaded_file):
                reader = pdf.PdfReader(uploaded_file)
                text = ""
                for page in range(len(reader.pages)):
                    page = reader.pages[page]
                    text += str(page.extract_text())
                return text

            resume_text = input_pdf_text(uploaded_file)

            # Tarefa de Análise da Vaga
            task_scrape_content = Task(
                description=f"""
                Analyze the following job description to extract key responsibilities, requirements, desired skills, and knowledge:
                {job_description}
                """,
                agent=Scraping_Agent,
                expected_output="Extracted key responsibilities, requirements, desired skills, and knowledge from the job description."
            )

            # Tarefa de Estruturação do Currículo
            task_enhance_content = Task(
                description=f"""
                Structure the resume based on the extracted information from the job description, including a summary/objective statement.
                Here is the extracted information:

                {task_scrape_content.expected_output}

                And here is the resume text:
                {resume_text}
                """,
                agent=Enhancement_Agent,
                expected_output="Structured resume with a tailored experience statement."
            )

            # Tarefa de Sugestões de Melhoria e Aplicação
            task_transfer_content = Task(
                description="""Provide improvement suggestions for the resume based on the job requirements, apply these suggestions to enhance match probability with the job description, and incorporate relevant keywords.
                Here is the structured resume:

                {task_enhance_content.expected_output}

                Please focus on the following aspects:
                - Improve all experiences experience descriptions to align with the job requirements.
                - maintain a standard of at least 4 well-detailed and explained bullet points aligned with the job description in each experience.
                - Enhance all projects descriptions to highlight relevant skills and achievements.
                - Use keywords from the job description to increase the match probability.

                """,
                agent=Transfer_Agent,
                expected_output="Improved resume with applied suggestions, aligned experience and project descriptions, and relevant keywords."
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