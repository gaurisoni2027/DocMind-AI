from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (API keys, secrets, etc.)
load_dotenv()

# Create embedding model to convert text into vector representations
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the already created Chroma vector database
vector_store = Chroma(
    persist_directory="chroma-db",
    embedding_function=embeddings_model
)

# Create a retriever to fetch relevant documents
retriever = vector_store.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance for diverse results
    search_kwargs={
        "k": 4,              # Return top 4 documents
        "fetch_k": 10,       # Initially fetch top 10 documents
        "lambda_mult": 0.5   # Balance relevance and diversity
    }
)

# Initialize the Mistral language model
llm = ChatMistralAI(model="mistral-small-2603")

# Define prompt template for the RAG system
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.
            Use ONLY the provided context to answer the question.
            If the answer is not found in the context, say:
            "I couldn't find the answer in the document."
            """
        ),
        (
            "human",
            """
            Context: {context}

            Question: {question}
            """
        )
    ]
)

print("RAG system created successfully")
print("Press 0 to exit")

# Start chat loop
while True:
    query = input("You: ")

    # Exit the application if user enters 0
    if query == "0":
        break

    # Retrieve relevant documents from vector database
    docs = retriever.invoke(query)

    # Combine retrieved document contents into a single context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Fill the prompt template with context and user question
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    # Generate response using the LLM
    response = llm.invoke(final_prompt)

    # Display the final answer
    print(f"\nAI: {response.content}")