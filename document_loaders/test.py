from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter=CharacterTextSplitter(
    separator="",
    chunk_size=1000,
    chunk_overlap=1

)
print("Loading file...")

loader = TextLoader(
    "document_loaders/notes.txt",
    encoding="utf-8"
)

docs = loader.load()


print("Loaded successfully!")
print(type(docs))
print(len(docs))
#print(docs[0].page_content[:200])

chunks=splitter.split_documents(docs)
print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()