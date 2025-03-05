import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# Load embeddings
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Set up the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create QA retrieval system
rag = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

# Define your CTA
CTA = """
\n---\n
📘 **Discover More in 'After Happily Ever After':**  
Navigate divorce and loss, reclaim your identity, and build a life you love.  
🌟 [Pre-order your copy here](https://publishizer.com/after-happily-ever-after/)
"""

# Interactive prompt loop
def chat():
    print("Ask a question about the book ('exit' to quit):")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = rag.invoke(user_input)
        # Append the CTA to each response
        print(f"\nAI: {response['result']}{CTA}")

if __name__ == "__main__":
    chat()
