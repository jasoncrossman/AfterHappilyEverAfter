import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Load documents
docs_folder = "./book_docs/"
all_docs = []

for file in os.listdir(docs_folder):
    if file.endswith('.txt'):
        loader = TextLoader(os.path.join(docs_folder, file), encoding='utf-8')
        docs = loader.load()
        all_docs.extend(docs)

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_chunks = text_splitter.split_documents(all_docs)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store from the chunks
vector_store = FAISS.from_documents(doc_chunks, embeddings)

# Save embeddings locally
vector_store.save_local("faiss_index")

print("✅ Embeddings successfully created and saved!")

