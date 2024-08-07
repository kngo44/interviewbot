import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from langchain.text_splitter import RecursiveCharacterTextSplitter
from preprocessing import preprocess_text

def parse_url(url):
    try:
        response = requests.get(url)
    except Exception as e:
        print(f"URL does not work: {e}")
        return ""
    
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    text = md(text)
    
    return text

def parse_job_description(url, api_key):
    text = parse_url(url)
    processed_text = preprocess_text(text)
    text_splitter = RecursiveCharacterTextSplitter(max_length=4000)
    chunks = text_splitter.split_text(processed_text)
    return chunks