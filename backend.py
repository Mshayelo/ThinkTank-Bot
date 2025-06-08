# Think Tank AI Chatbot Backend 
# This backend powers the document assistant. It handles file upload, text extraction, and generating AI responses
# using Azure OpenAI, Azure Blob Storage, Azure Cognitive Search, and Azure Document Intelligence (Form Recognizer).

import os  # For environment variables
import uuid  # For unique filenames
from datetime import datetime, timedelta  # For SAS token expiry
from dotenv import load_dotenv  # To load environment variables from .env file
from flask import Flask, request, jsonify  # Flask app and request handling

# Azure SDKs
from openai import AzureOpenAI  # Azure OpenAI client
from azure.search.documents import SearchClient  # Cognitive Search client
from azure.core.credentials import AzureKeyCredential  # Authentication for Azure services
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions  # For file upload & secure access
from azure.ai.formrecognizer import DocumentAnalysisClient  # Document Intelligence (Form Recognizer)
from azure.core.credentials import AzureKeyCredential as FormRecognizerKey  # Auth for Form Recognizer

#Load environment variables
load_dotenv()

# Azure Configuration
# These values are stored in .env for security and used to authenticate Azure services
AZURE_OAI_ENDPOINT = os.getenv("AZURE_OAI_ENDPOINT")
AZURE_OAI_KEY = os.getenv("AZURE_OAI_KEY")
AZURE_OAI_DEPLOYMENT = os.getenv("AZURE_OAI_DEPLOYMENT")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")
AZURE_BLOB_KEY = os.getenv("AZURE_BLOB_KEY")

AZURE_DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
AZURE_DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")

#Azure Client Initialization

# General OpenAI (for uploaded docs)
client = AzureOpenAI(
    api_key=AZURE_OAI_KEY,
    api_version="2023-09-01-preview",
    azure_endpoint=AZURE_OAI_ENDPOINT
)

# OpenAI with extensions (for indexed docs via Cognitive Search)
client_with_extensions = AzureOpenAI(
    base_url=f"{AZURE_OAI_ENDPOINT}/openai/deployments/{AZURE_OAI_DEPLOYMENT}/extensions",
    api_key=AZURE_OAI_KEY,
    api_version="2023-09-01-preview"
)

# Azure Search client
search_client = SearchClient(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, AzureKeyCredential(AZURE_SEARCH_KEY))

# Azure Blob client for file uploads
blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER)

# Azure Document Intelligence (Form Recognizer) client
doc_client = DocumentAnalysisClient(endpoint=AZURE_DOC_INTELLIGENCE_ENDPOINT, credential=FormRecognizerKey(AZURE_DOC_INTELLIGENCE_KEY))

# Flask App Initialization
app = Flask(__name__)

# Basic check route
@app.route("/")
def home():
    return "Flask chatbot API is running!"

# Chat with Indexed Documents (Cognitive Search)
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")  # Get user query
    response_text = chatbot_query(user_input)  # Generate AI response
    return jsonify({"response": response_text})  # Return JSON

def chatbot_query(user_input):
    # Configuration for Azure Cognitive Search as a data source
    extension_config = {
        "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": AZURE_SEARCH_ENDPOINT,
                    "key": AZURE_SEARCH_KEY,
                    "indexName": AZURE_SEARCH_INDEX,
                }
            }
        ]
    }

    # Prompt given to OpenAI
    prompt = f"""
      You are an AI assistant helping users engage with business documents.
      User query: {user_input}

    Always provide specific, relevant information      
      First, give a general summary(Include scope & purpose). Then extract , focus on these:
      - Technology Specs
      - Problem being addressed
        -User numbers and deployment scale
        -Current systems and pain points
        - Action Items for proposal team
        - Required integrations
        - If suggesting BOM items, explain why they match the requirements
        If any section is missing, respond with: "This document does not contain information on [Missing Section]."

        Respond clearly and conversationally.
    """

    try:
        # Generate AI response with extensions (Cognitive Search)
        response = client_with_extensions.chat.completions.create(
            model=AZURE_OAI_DEPLOYMENT,
            temperature=0.5,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": "You answer using Azure Cognitive Search."},
                {"role": "user", "content": prompt}
            ],
            extra_body=extension_config
        )
        return response.choices[0].message.content
    except Exception as ex:
        return f"Error retrieving insights: {str(ex)}"

# Upload + Ask Once (One-time file + question)
@app.route("/upload_and_ask", methods=["POST"])
def upload_and_ask():
    try:
        # Get file and user question from form data
        file = request.files["file"]
        user_question = request.form["question"]

        # Upload document to Blob Storage with unique name
        filename = f"{uuid.uuid4()}_{file.filename}"
        blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_CONTAINER, blob=filename)
        blob_client.upload_blob(file, overwrite=True)

        # Generate secure link (SAS token) for document
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=AZURE_BLOB_CONTAINER,
            blob_name=filename,
            account_key=AZURE_BLOB_KEY,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(minutes=10)
        )
        sas_url = f"{blob_client.url}?{sas_token}"

        # Analyze file using Document Intelligence
        poller = doc_client.begin_analyze_document_from_url("prebuilt-document", sas_url)
        result = poller.result()
        full_text = " ".join([p.content for p in result.paragraphs]) if result.paragraphs else "No content found"

        # Build AI prompt with content + question
        prompt = f"""
        The user uploaded this document:
        {full_text}

        They asked:
        {user_question}

        Please answer using only the document's contents.
        """

        # Generate answer from OpenAI
        response = client.chat.completions.create(
            model=AZURE_OAI_DEPLOYMENT,
            temperature=0.5,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": "You are an AI document assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return jsonify({"answer": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/extract_text", methods=["POST"])
def extract_text():
    try:
        file = request.files["file"]
        filename = f"{uuid.uuid4()}_{file.filename}"
        blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_CONTAINER, blob=filename)
        blob_client.upload_blob(file, overwrite=True)

        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=AZURE_BLOB_CONTAINER,
            blob_name=filename,
            account_key=AZURE_BLOB_KEY,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(minutes=10)
        )
        sas_url = f"{blob_client.url}?{sas_token}"

        # Debugging: Print raw Azure Document Intelligence response
        poller = doc_client.begin_analyze_document_from_url("prebuilt-document", sas_url)
        result = poller.result()
        print(f"🔍 Raw API Response: {result}")  #  Add this line

        # Check if paragraphs exist
        if not result.paragraphs:
            return jsonify({"error": "No text extracted from document"}), 400

        full_text = " ".join([p.content for p in result.paragraphs])

        return jsonify({"text": full_text})
    
    except Exception as e:
        print(f" Error extracting text: {str(e)}")  #  Log exact error
        return jsonify({"error": str(e)}), 500

# Ongoing Chat for Uploaded Document (with memory)
@app.route("/followup_chat", methods=["POST"])
def followup_chat():
    try:
        data = request.get_json()
        doc_text = data["doc"]  # Full text from uploaded document
        chat_history = data["history"]  # Previous messages

        # Instructional prompt that guides AI response
        prompt_intro = """

        You are an AI assistant analyzing uploaded business documents.
        First, summarize key content(include problem trying to solve/address). Then extract:
        - Technology Specs
        -User numbers and deployment scale
        -Current systems and pain points
        - Action Items for proposal team
        - Required integrations
        - If suggesting BOM items, explain why they match the requirements
        - Compliance requirements
        -conclusive summary
        If any section is missing, respond with: "This document does not contain information on [Missing Section]."
        Respond clearly and conversationally.

        """

        # Combine system instructions, document, and previous Q&A
        messages = [
            {"role": "system", "content": prompt_intro},
            {"role": "system", "content": doc_text}
        ] + chat_history

        # Generate response using OpenAI with context
        response = client.chat.completions.create(
            model=AZURE_OAI_DEPLOYMENT,
            temperature=0.5,
            max_tokens=1000,
            messages=messages
        )

        return jsonify({"answer": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Launch Flask Server ===
if __name__ == "__main__":
    app.run(debug=True)
