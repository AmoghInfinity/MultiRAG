import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Load API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0
)

# Test query
response = llm.invoke(
    "What is Retrieval Augmented Generation?"
)

print("\nGroq Response:\n")

print(response.content)