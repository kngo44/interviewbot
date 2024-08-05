from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

def generate_questions(resume_chunks, job_description_chunks, additional_context, api_key):
    llm = OpenAI(api_key=api_key, model="gpt-4o-mini")
    prompt_template = PromptTemplate(
        input_variables=["resume", "job_description", "context"],
        template="Generate interview questions based on the following resume, job description, and additional context:\n\nResume:\n{resume}\n\nJob Description:\n{job_description}\n\nContext:\n{context}"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    resume_text = "\n".join(resume_chunks)
    job_description_text = "\n".join(job_description_chunks)
    
    response = chain.run({"resume": resume_text, "job_description": job_description_text, "context": additional_context})
    questions = response.split('\n')
    
    return questions
