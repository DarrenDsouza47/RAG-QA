import os
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
deployment_name = os.getenv("DEPLOYMENT_NAME")

def initialize_llm():
    return AzureChatOpenAI(api_key=openai_api_key,
                            azure_endpoint=azure_endpoint,
                            api_version='2024-05-01-preview',
                            model='gpt-4o-mini',
                            deployment_name='gpt-4o-mini')

def process_pdf(uploaded_pdf):
    pdf_reader = PdfReader(uploaded_pdf)
    raw_text = ""
    
    for page_num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[page_num].extract_text()
        if page_text:
            raw_text += page_text
        else:
            return None, f"Warning: Page {page_num + 1} contains no extractable text."
    
    if not raw_text.strip():
        return None, "Error: The PDF contains no extractable text."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    
    return [Document(page_content=chunk) for chunk in chunks], None

def create_embeddings(documents):
    return AzureOpenAIEmbeddings(openai_api_key=openai_api_key, chunk_size=1000, azure_endpoint=azure_endpoint)

def create_vectorstore(documents, embeddings):
    return Chroma.from_documents(documents, embeddings, persist_directory='./embeddings')
