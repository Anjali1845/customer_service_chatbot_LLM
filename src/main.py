import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖")

st.title("🤖 CUSTOMER SERVICE CHATBOT")

# Create Knowledge Base
if st.button("📚 Create Knowledgebase"):
    with st.spinner("Creating knowledge base..."):
        create_vector_db()
    st.success("Knowledge base created successfully! ✅")

st.divider()

question = st.text_input("Ask your question")

if question:
    chain = get_qa_chain()

    with st.spinner("Generating answer..."):
        response = chain.invoke({"query": question})

    st.subheader("Answer")
    st.write(response["result"])