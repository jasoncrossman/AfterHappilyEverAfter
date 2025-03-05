import streamlit as st
from basic_rag_pipeline import setup_pipeline  # Import your RAG setup function

# Initialize the RAG pipeline
qa_chain, retriever = setup_pipeline()

# Streamlit UI
st.set_page_config(page_title="📖 After Happily Ever After AI Assistant", layout="wide")

st.title("📖 After Happily Ever After AI Assistant")
st.write("### Welcome, and Thanks for Stopping By! 👋")
st.write(
    "Got a question about the book or about me? I'm all ears! Drop your question below, and let's start a conversation. ⬇️"
)

user_query = st.text_area("💡 How Can I Help:")

if st.button("🔍 Get Answer"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    elif not qa_chain:
        st.error("Error: RAG pipeline is not initialized.")
    else:
        response = qa_chain.invoke({"question": user_query, "chat_history": []})
        answer = response.get("answer", "No response generated.")

        # Append CTA message and link after every response
        cta_message = """
        
        ---

        ### 🔥 **Want to go deeper?**  
        **As an Emmy Award-winning filmmaker and former Content Producer for The Dave Ramsey Show,**  
        I have over twenty years of experience crafting messages that resonate deeply and drive real change.  

        But beyond my professional background, I wrote *After Happily Ever After* from a place of lived experience.  
        Having endured a painful divorce, the loss of family members, and challenges that nearly cost me my daughter,  
        I understand the depths of pain men face during these crises.  

        Through my own journey of healing, I’ve developed a clear, actionable roadmap that can help millions  
        not only survive but emerge stronger from the darkest chapters of their lives.  

        This book is my way of reaching beyond the hundreds of men I’ve coached across the U.S. and worldwide.  
        It’s a unique resource for men, addressing a profound need with honesty, empathy, and practical guidance.  

        I’m convinced that this message of resilience and hope has the power to transform lives.  

        **📖 Support the book & join the journey:**  
        👉 [Click here to pre-order & sponsor](https://publishizer.com/after-happily-ever-after/)
        """

        # Combine answer with the CTA message
        full_response = answer + cta_message

        st.success("✅ Response:")
        st.write(full_response)
