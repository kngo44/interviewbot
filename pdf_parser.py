from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from preprocessing import preprocess_text

def parse_pdf(pdf_path):
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
    
def parse_resume(file_path, api_key):
    text = parse_pdf(file_path)
    processed_text = preprocess_text(text)
    text_splitter = RecursiveCharacterTextSplitter(max_length=4000)
    chunks = text_splitter.split_text(processed_text)
    return chunks