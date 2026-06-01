from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data=PyPDFLoader(
    "document_loaders/GRU.pdf"
)
docs=data.load()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant that summarizes the given text."),
        ("human", "{data}")
    ]
)

prompt = template.format_messages(
    #data=docs[0].page_content
    data=docs
)

model = ChatMistralAI(model="mistral-small-2603")

result = model.invoke(prompt)

print(result.content)