from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyMuPDFLoader("data/sample.pdf")
docs = loader.load()

# Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

print(f"\nTotal chunks created: {len(chunks)}")

print("\nFirst chunk:\n")
print(chunks[0].page_content)

print("\n" + "="*80)

print("\nSecond chunk:\n")
print(chunks[1].page_content)