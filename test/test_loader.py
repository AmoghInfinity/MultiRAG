from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("data/sample.pdf")

docs = loader.load()

print(f"Total pages loaded: {len(docs)}")

print("\nFirst page preview:\n")
print(docs[0].page_content[:1000])