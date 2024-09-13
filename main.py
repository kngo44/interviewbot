from dotenv import load_dotenv
import os
from pdf_parser import parse_pdf
from url_parser import parse_url
from question_generator import generate_questions
from conversation_agent import conduct_interview
from feedback_agent import provide_feedback
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def main():
     resume_path = input("Enter the path to your resume (PDF format): ")
     job_description_url = input("Enter the URL of the job description: ")
     additional_context = input("Enter any additional context or preferences for the interview (e.g., preferred job roles, industries): ")
         
     resume_data = parse_pdf(resume_path, OPENAI_API_KEY)
     job_description_data = parse_url(job_description_url, OPENAI_API_KEY)

     questions = generate_questions(resume_data, job_description_data, additional_context, OPENAI_API_KEY)
     memory = ConversationBufferMemory(memory_key="chat_history")
     user_responses = conduct_interview(questions, memory, OPENAI_API_KEY)
     provide_feedback(user_responses, OPENAI_API_KEY)

if __name__ == "__main__":
    main()