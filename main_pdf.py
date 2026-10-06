from dotenv import load_dotenv
load_dotenv()


from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(
    model ="mistral-small-2603"
)

from langchain_community.document_loaders import PyPDFLoader
data = PyPDFLoader("document loaders/Lost.pdf")
docs=data.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunk = splitter.split_documents(docs)

from langchain_core.prompts import ChatPromptTemplate
template=ChatPromptTemplate.from_messages(
    [("system","Hey as your are a AI so please give me the summary"),
     ("human","{data}")]
)

prompt = template.format_messages(data=docs[0].page_content)

response = model.invoke(prompt)
print(response.content)