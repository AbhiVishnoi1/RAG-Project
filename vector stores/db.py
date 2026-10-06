from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document

docs=[
    Document(page_content="Python is widely used in AI",metadata={"source":"Book1"}),
    Document(page_content="Pandas is widely used in data analysis",metadata={"source":"Book2"}),
    Document(page_content="Neural Network is widely used in AI",metadata={"source":"Book3"})
]

embedding_model = HuggingFaceEmbeddings()

vectorstores=chroma.from_documents(
    embedding=embedding_model,
    documents=docs,
    persist_directory="chroma-db"
)