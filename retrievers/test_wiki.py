from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=2
)

docs = retriever.invoke("Large Language Models")

for doc in docs:
    print(doc.page_content[:500])