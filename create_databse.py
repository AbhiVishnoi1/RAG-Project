## load the pdf 
## split it into chunks 
## create the embedding 
## strore into chroma 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

data=PyPDFLoader("document loaders/Machine Learning.pdf")
docs=data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunk=splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstores = Chroma.from_documents(
    embedding=embeddings,
    documents=chunk,
    persist_directory="chroma_db"
)   