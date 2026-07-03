import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Explicit path loading to ensure the hidden .env parameters are read cleanly
current_directory = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_directory, '.env')
load_dotenv(dotenv_path=env_path)

def run_local_faiss_ingestion(pdf_file_path):
    """Utility bridge used if you want an isolated script to handle page extractions."""
    print(f"📄 Local ingestion initialization triggered for target: {pdf_file_path}")
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"Target PDF asset path not found: {pdf_file_path}")
        
    loader = PyMuPDFLoader(pdf_file_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(documents)
    
    print(f"✅ Extracted {len(documents)} pages and compiled {len(chunks)} text chunks.")
    return chunks