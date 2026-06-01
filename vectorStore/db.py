from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Sample documents to store in Chroma DB
docs = [
    Document(
        page_content="""
Machine Learning enables computers to learn patterns from data and make predictions.
Common algorithms include Linear Regression, Decision Trees, and Random Forests.
Applications include recommendation systems, fraud detection, and spam filtering.
""",
        metadata={
            "topic": "Machine Learning",
            "difficulty": "Beginner",
            "source": "ML Fundamentals"
        }
    ),

    Document(
        page_content="""
Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers.
Popular architectures include CNNs for images, RNNs for sequences, and Transformers for NLP.
""",
        metadata={
            "topic": "Deep Learning",
            "difficulty": "Intermediate",
            "source": "DL Handbook"
        }
    ),

    Document(
        page_content="""
Generative AI creates new content such as text, images, audio, and code.
Examples include GPT, Gemini, Claude, and Mistral. It is widely used in chatbots,
content generation, and document summarization.
""",
        metadata={
            "topic": "Generative AI",
            "difficulty": "Advanced",
            "source": "GenAI Guide"
        }
    )
]

# Convert text into embeddings (vectors)
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Using Hugging Face Embeddings")

# Create and store documents in Chroma DB
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    persist_directory="chroma-db"
)

print("Vector Store Created Successfully!")

# Similarity Search
results = vector_store.similarity_search(
    "What are neural networks?",
    k=1
)

print("\nSimilarity Search Result:\n")

for r in results:
    print(r.page_content)
    print(r.metadata)

# Convert vector store into a retriever
retriever = vector_store.as_retriever()

# Retrieve relevant documents
retrieved_docs = retriever.invoke(
    "Explain deep learning"
)

print("\nRetriever Result:\n")

for doc in retrieved_docs:
    print(doc.page_content)