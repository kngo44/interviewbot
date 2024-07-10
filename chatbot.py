from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

import streamlit as st

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
import os

load_dotenv()

OPEN_API_KEY = os.getenv('OPENAI_API_KEY', 'OPENAI_API_KEY') 

def load_LLM(open_api_key):
    llm = ChatOpenAI(temperature=0.7, openai_api_key=OPEN_API_KEY)

# function to get info from urls
def website_info(url):
    try:
        response = requests.get(url)
    except:
        print ("URL does not work")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')

    text = soup.get_text()

    text = md(text)

    return text

# getting urls from input
website_data = ""
urls = ["https://jobs.mdanderson.org/search/jobdetails/associate--data-engineer-lab-medicine/d53e1f57-49e3-44ed-aae1-28e3871e93f8"]

for url in urls:
    text = website_info(url)

    website_data += text

# function to get info from PDFs
def pdf_info(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text = ""
        for doc in documents:
            text += doc.page_content
        return text
    except Exception as e:
        print(f"Failed to load PDF: {e}")
        return ""

# loading pdfs
pdf_path = "Ngo, Khuong.pdf"
pdf_data = pdf_info(pdf_path)

# print(website_data[2000:5000])
# print(pdf_data[:400])

user_information = website_data + pdf_data

text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=4000)
docs = text_splitter.create_documents([user_information])

map_prompt = """You are an expert recruiter that aids a user in practicing interviews.
Below is information about a person named {persons_name} and the place they are getting interviewed at called {job_name}
Information will include a resume about {persons_name} and a job description about {job_name}
Your goal is to generate interview questions that {job_name} might ask {persons_name}
Use specifics from the resume and job description when possible

% START OF INFORMATION ABOUT {persons_name} and {job_name}:
{text}
$ END OF INFORMATION ABOUT {persons_name} and {job_name}:

Please respond with list of a few interview questions based on the topics above

YOUR RESPONSE:"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "persons_name", "job_name"])

combine_prompt = """
You are an expert recruiter that aids a user in practicing interviews.
You will be given a list of potential interview questions that we can ask {persons_name}.

Please consolidate the questions and return a list

% INTERVIEW QUESTIONS
{text}

% YOUR RESPONSE:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "persons_name"])

llm = ChatOpenAI(temperature=0.25, model_name='gpt-3.5-turbo')

chain = load_summarize_chain(llm,
                             chain_type="map_reduce",
                             map_prompt=map_prompt_template,
                             combine_prompt=combine_prompt_template,
                             verbose=True
                             )