import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up Streamlit UI
st.title("Ask About 'After Happily Ever After' 📖")

st.write("Got questions about life after divorce? Ask away! I'm here to help.")

# Load embeddings and FAISS index
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# User input
user_question = st.text_input("Type your question below:")

if user_question:
    # Retrieve relevant content from the book
    docs = vector_store.similarity_search(user_question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Construct the response with disclaimer
    response = llm.invoke(f"{context}\n\n{user_question}")

    disclaimer = (
        "\n\n---\n\n"
        "**⚠️ Disclaimer:** I am not an attorney and cannot provide legal advice. "
        "If you need legal assistance, consult a qualified professional. "
        "If you are in danger or facing an emergency, please call 911 or a local crisis hotline immediately."
    )

    cta = (
        "\n\n💡 *Want to dive deeper?* Grab my book *After Happily Ever After* here: "
        "[Pre-Order Now](https://publishizer.com/after-happily-ever-after/)!"
    )

    st.write(response.content + disclaimer + cta)
