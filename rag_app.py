import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

# Load environment variables (API key)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# Load FAISS index with your book embeddings
vector_store = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Streamlit UI
st.title("Ask Jason Crossman - *After Happily Ever After*")
st.write("Got questions about healing after divorce? Ask below!")

# User input
user_input = st.text_input("Type your question here and press Enter:")

if user_input:
    # Retrieve relevant book content
    retrieved_docs = vector_store.similarity_search(user_input, k=5)  # Increased context retrieval
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Custom prompt to ensure the AI speaks in your tone
    prompt_template = f"""You are Jason Crossman, an Emmy Award-winning director, author, and storyteller.
    You have a fun, engaging, and often humorous way of writing. You connect deeply with your audience and use a conversational tone.
    Your job is to answer the user's question using information from the following retrieved context from your book.
    If the answer is not in the book, do not make something up—just say you don't know.

    Context:
    {context}

    User Question:
    {user_input}

    Your Response:
    """
    
    # Generate response
    response = llm(prompt_template)

    # Append CTA to every response
    response_text = response + "\n\n💡 *Want to dive deeper?* Grab my book *After Happily Ever After* here: [Pre-Order Now](https://publishizer.com/after-happily-ever-after/)!"

    # Display response
    st.write(response_text)
