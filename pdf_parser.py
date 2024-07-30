from langchain_community.document_loaders import PyPDFLoader

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