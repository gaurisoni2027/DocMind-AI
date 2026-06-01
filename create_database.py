# Load PDF
# Split PDF into chunks
# Generate embeddings for each chunk
# Store embeddings in ChromaDB

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Load PDF as LangChain Documents
loader = PyPDFLoader(
    "document_loaders/deeplearning.pdf"
)
docs = loader.load()

print(f"Pages Loaded: {len(docs)}")

# Split large documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Max characters per chunk
    chunk_overlap=200    # Preserve context between chunks
)

chunks = splitter.split_documents(docs)

print(f"Chunks Created: {len(chunks)}")

# Convert text chunks into vector embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Using Hugging Face Embeddings")

# Store chunk embeddings in ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,         # Store chunks, not original docs
    embedding=embeddings_model,
    persist_directory="chroma-db"
)

print("Vector Store Created Successfully!")