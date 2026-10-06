from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(
    model ="mistral-small-2603"
)

from langchain_community.document_loaders import TextLoader

data = TextLoader("document loaders/notes.txt")
docs= data.load()

from langchain_core.prompts import ChatPromptTemplate
template=ChatPromptTemplate.from_messages(
    [("system",
      "you are a AI that summarize the text"
      ),
      ("human",
       "{data}"
       )]
)

prompt=template.format_messages(data=docs[0].page_content)
    
response = model.invoke(prompt)
print(response.content)