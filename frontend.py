# Think Tank AI Document Chatbot (Frontend) 

# This Streamlit app provides a chat interface for two modes:
# 1. Chat with indexed documents (Azure Cognitive Search)
# 2. Chat with newly uploaded documents (Azure Document Intelligence + GPT)

import streamlit as st
import requests  # Used to communicate with Flask backend

# Set up basic page configuration
st.set_page_config(page_title="Think Tank AI Chatbot", page_icon="🤖", layout="wide")

# Sidebar with branding and usage instructions
with st.sidebar:
    st.image("tt.png", width=180)  # Project logo
    st.markdown("### About AI Chatbot")
    st.markdown("This chatbot helps retrieve business insights and answer general queries.")
    st.markdown("### How to Use")
    st.markdown("- Upload a document and ask questions")
    st.markdown("- Or query indexed documents directly")

# Page title and welcome message
st.title("Think Tank AI Enterprise Chatbot")
st.markdown("**Welcome! Upload documents or chat with indexed content.**")

# Mode selector: lets user choose between indexed docs or upload chat
st.markdown("---")
mode = st.radio("Choose mode", ["Chat with Indexed Docs", "Ask Question About Uploaded Doc"])

# Initialize session variables to store message history and document text
if "messages" not in st.session_state:
    st.session_state.messages = []  # For indexed document chat
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""  # Stores uploaded document's extracted content
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Chat history for uploaded document

# MODE 1: Chat with Indexed Documents (Azure Cognitive Search)
if mode == "Chat with Indexed Docs":
    st.markdown("###  Chat History")
    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for user to ask questions
    if prompt := st.chat_input("Ask about indexed documents..."):
        # Save user's message to session
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send message to backend and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                response = requests.post("http://127.0.0.1:5000/chat", json={"message": prompt})
                ai_response = response.json()["response"]
                message_placeholder.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                message_placeholder.markdown(f"Error: {str(e)}")

# === MODE 2: Upload Document and Chat (Document Intelligence + GPT) ===
elif mode == "Ask Question About Uploaded Doc":
    st.markdown("### 📄 Upload a document")
    # Upload field for PDFs or TXT files
    uploaded_doc = st.file_uploader("Upload your document (PDF Format)", type=["pdf", "txt"])

    # If a file is uploaded and button is clicked, extract text from backend
    if uploaded_doc:
        if st.button("Extract & Load Document"):
            with st.spinner("Uploading and analyzing..."):
                try:
                    res = requests.post("http://127.0.0.1:5000/extract_text", files={"file": uploaded_doc})
                    result = res.json()
                    # Store the extracted text in session
                    st.session_state.doc_text = result["text"]
                    st.session_state.chat_history = []  # Reset history
                    st.success(" Document loaded. You can now ask questions.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Once text is loaded, allow chat with memory
    if st.session_state.doc_text:
        st.markdown("###  Chat History")
        # Display previous chat messages with this document
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input field for user to ask a question about the uploaded doc
        user_question = st.chat_input("Ask something about the uploaded document...")
        if user_question:
            # Store user question
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Send user question and full history to backend
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    res = requests.post(
                        "http://127.0.0.1:5000/followup_chat",
                        json={
                            "doc": st.session_state.doc_text,
                            "history": st.session_state.chat_history
                        }
                    )
                    result = res.json()
                    ai_response = result["answer"]
                    message_placeholder.markdown(ai_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                except Exception as e:
                    message_placeholder.markdown(f"Error: {str(e)}")
