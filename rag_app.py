import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("API Key not found! Please check your environment variables.")
    st.stop()

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Load FAISS vector store
try:
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

# Initialize Chat Model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Streamlit UI
st.title("📚 After Happily Ever After - Chat with Jason's Book")

st.write("👋 Ask anything about the book, and I'll answer in my own words!")

# User input
user_input = st.text_input("💬 What would you like to ask?", "")

if user_input:
    # Retrieve relevant context from the vector store
    docs = vector_store.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate response
    prompt = f"""
    You are Jason Crossman, the author of *After Happily Ever After*. 
    Answer the user's question in **your own voice**, with humor, empathy, and encouragement.

    **Context from your book:**  
    {context}

    **User Question:** {user_input}

    **Your Response:**
    """
    
    response = llm.invoke(prompt)
    
    # Display response
    if hasattr(response, "content"):
        st.write(response.content)
    else:
        st.write(response)

    # Call-to-action message
    st.markdown(
        "\n\n💡 *Want to dive deeper?* Grab my book **After Happily Ever After** here: "
        "[Pre-Order Now](https://publishizer.com/after-happily-ever-after/)!"
    )
