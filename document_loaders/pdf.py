from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter


data=PyPDFLoader(
    "document_loaders/GRU.pdf"
)
docs=data.load()
print(len(docs))

text_splitter = TokenTextSplitter(

    chunk_size=1000,
    chunk_overlap=10
)

chunks=text_splitter.split_documents(docs)
print(len(chunks))
print("first chunk: --->",chunks[0].page_content)

#ek pdf me jitne pages hote hai unka combined ek doc bnega isliye GRU me 15pages the isliye doc ki len is 15
