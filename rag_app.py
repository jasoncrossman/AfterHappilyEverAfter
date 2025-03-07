import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Define FAISS index path
index_path = "faiss_index"

# Function to create FAISS index if missing
def create_faiss_index():
    st.warning("FAISS index not found. Rebuilding... ⏳")
    
    docs_folder = "./book_docs/"
    all_docs = []

    for file in os.listdir(docs_folder):
        if file.endswith('.txt'):
            loader = TextLoader(os.path.join(docs_folder, file), encoding='utf-8')
            docs = loader.load()
            all_docs.extend(docs)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(all_docs)

    # Create FAISS index
    vector_store = FAISS.from_documents(doc_chunks, embeddings)
    vector_store.save_local(index_path)
    
    st.success("✅ FAISS index rebuilt successfully!")

# Load or recreate FAISS index
if os.path.exists(f"{index_path}/index.faiss"):
    st.info("✅ Loading FAISS index...")
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
else:
    create_faiss_index()
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Set up LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")

st.title("📖 Ask My Book - After Happily Ever After")
st.write("Type a question about my book, and I'll fetch the most relevant answer!")

# User Input
user_input = st.text_input("What would you like to ask?")

if user_input:
    docs = vector_store.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    response = llm.predict(f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {user_input}")

    st.subheader("Answer:")
    st.write(response)

    # Call-to-action (CTA)
    st.markdown("### 📢 Want to read the full book?")
    st.markdown("[Click here to pre-order now!](https://publishizer.com/after-happily-ever-after/)")
