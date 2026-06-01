
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API keys from .env file
load_dotenv()

# Load PDF document
data = PyPDFLoader(
    "document_loaders/deeplearning.pdf"
)

# Convert PDF pages into Document objects
docs = data.load()

# Split large text into smaller chunks for LLM processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # max chars per chunk
    chunk_overlap=200     # preserve context between chunks
)

# Create chunks
chunks = text_splitter.split_documents(docs)

print(len(chunks))

# Prompt template for summarization task
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant that summarizes the given text."),
        ("human", "{data}")
    ]
)

# Inject document content into prompt
prompt = template.format_messages(
    data=docs
)

# Initialize Mistral LLM
model = ChatMistralAI(model="mistral-small-2603")

# Send prompt to model
result = model.invoke(prompt)

# Display generated summary
print(result.content)