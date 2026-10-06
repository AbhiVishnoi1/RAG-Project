from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriver=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)
model = ChatMistralAI(
    model = "mistral-small-2506"
)
## creating the template 
prompt = ChatPromptTemplate(
    [
        ("system",
        """
        You are a AI and you are going to give em the answer of the question and if tbhe answer is now there
        then show "I could not find the Answer"
        """),
        ("human",
         """
        context:{context}
        Question : {question}       
         """)
    ]
)
print("RAG System")
print("Press 0 for exit")
while True:
    querry = input("You :")
    if querry == "0":
        break
    docs=retriver.invoke(querry)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        "context": context,
        "question":querry
    }
    )
    response = model.invoke(final_prompt)
    print(response.content)