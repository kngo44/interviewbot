from markdownify import markdownify as md
from bs4 import BeautifulSoup
import requests

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