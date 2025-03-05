import os
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

# Set OpenAI API Key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")

# Load book documents from the 'book_docs' folder
def load_book_docs():
    loader = DirectoryLoader("book_docs", glob="*.txt")
    return loader.load()

# Process the book content
docs = load_book_docs()

# Split the content for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# Create an embedding model and a FAISS vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(split_docs, embeddings)

# Define a prompt template to structure responses
template = """You are an AI assistant trained on the book *After Happily Ever After*. 
Answer the user's question using the book’s themes, insights, and guidance. 
Provide supportive, practical, and empathetic advice.

User Question: {question}

Book-Based Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create the RAG pipeline
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Function to initialize and return the pipeline
def setup_pipeline():
    return qa_chain, retriever
