import os
import openai
import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

# Load text files from a folder
def load_text_files(folder):
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append(Document(page_content=text))
    return docs

# Setup the RAG pipeline
def setup_pipeline():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment variables.")

    os.makedirs("book_docs", exist_ok=True)

    # Load and split documents
    raw_docs = load_text_files("book_docs")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(raw_docs)

    # Create embeddings and vector database
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    # Setup conversational RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4o"),
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain, retriever

# Generate a structured 60-day book marketing plan
def generate_marketing_plan(qa_chain):
    query = (
        "Create a detailed 60-day book marketing plan to sell 300 copies of my book. "
        "Include specific daily tasks for social media, email marketing, ads, and partnerships. "
        "Focus on practical step-by-step actions leading from zero to success. "
        "Format it in a structured way for easy tracking and execution."
    )
    result = qa_chain.invoke({"query": query})
    return result.get("answer", "No response generated.")

# Format the marketing plan output for better readability
def format_60_day_plan(text):
    sections = text.split("\n\n")
    formatted_plan = "\n📅 **60-Day Book Marketing Plan**\n"
    for section in sections:
        wrapped_text = textwrap.fill(section, width=100)
        formatted_plan += f"\n📌 {wrapped_text}\n"
    return formatted_plan

if __name__ == "__main__":
    qa_chain, retriever = setup_pipeline()
    marketing_plan_text = generate_marketing_plan(qa_chain)
    formatted_plan = format_60_day_plan(marketing_plan_text)

    # Save results for later use
    with open("60_day_marketing_plan.txt", "w") as f:
        f.write(formatted_plan)
    
    print("✅ 60-Day Marketing Plan saved as `60_day_marketing_plan.txt`!")
