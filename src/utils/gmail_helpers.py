"""
Gmail Auto-Reply System - Helper Classes
Contains GmailAutoReplyBot and GmailService classes
"""

from __future__ import print_function
import os.path
import base64
import re
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class GmailAutoReplyBot:
    """Gmail Auto-Reply Bot using LangChain and FAISS"""
    
    def __init__(self, csv_path, openai_model="gpt-4", embedding_model="text-embedding-3-small"):
        """Initialize the bot with CSV data and LangChain components"""
        self.csv_path = csv_path
        self.openai_model = openai_model
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.retriever = None
        self.chain = None
        self._setup_langchain()
    
    def _setup_langchain(self):
        """Setup LangChain components: embeddings, vectorstore, and chain"""
        print("📚 Loading CSV and building vector store...")
        
        # Load CSV and prepare data
        df = pd.read_csv(self.csv_path)
        df["text"] = df.apply(
            lambda x: f"Category: {x['Category']}\nQuery: {x['Customer_Email']}\nReply: {x['Support_Reply']}", 
            axis=1
        )
        
        # Load documents
        loader = DataFrameLoader(df, page_content_column="text")
        documents = loader.load()
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 1}
        )
        
        # Setup LLM
        llm = ChatOpenAI(model=self.openai_model, temperature=0.3)
        
        # Create prompt template
        prompt = PromptTemplate.from_template("""
You are a helpful and polite customer support assistant for a beauty brand.

Customer email:
{customer_email}

Similar previous query and official reply:
{similar_text}

Based on the similar query above, generate a professional and personalized reply to the customer.
Adapt the reply to match the customer's specific situation while maintaining a polite and helpful tone.

Final email reply:
""")
        
        # Create chain using LCEL (LangChain Expression Language)
        def format_docs(docs):
            return docs[0].page_content if docs else "No similar query found."
        
        self.chain = (
            {
                "customer_email": RunnablePassthrough(),
                "similar_text": self.retriever | format_docs
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("✅ LangChain setup complete!")
    
    def generate_reply(self, customer_email):
        """Generate reply for customer email using LangChain"""
        try:
            response = self.chain.invoke(customer_email)
            return response.strip()
        except Exception as e:
            print(f"❌ Error generating reply: {e}")
            # Fallback: try manual retrieval
            try:
                print("🔄 Attempting fallback retrieval method...")
                docs = self.retriever.invoke(customer_email)
                if docs:
                    similar_text = docs[0].page_content
                    fallback_prompt = ChatPromptTemplate.from_template("""
You are a helpful and polite customer support assistant for a beauty brand.

Customer email:
{customer_email}

Similar previous query and official reply:
{similar_text}

Based on the similar query above, generate a professional and personalized reply to the customer.Adapt the reply to match the customer's specific situation while maintaining a polite and helpful tone.

Final email reply:
""")
                    llm = ChatOpenAI(model=self.openai_model, temperature=0.5)
                    response = (fallback_prompt | llm | StrOutputParser()).invoke({
                        "customer_email": customer_email,
                        "similar_text": similar_text
                    })
                    return response.strip()
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
            return None


class GmailService:
    """Gmail API service handler"""
    
    def __init__(self, client_secret_path, token_file="token.json", scopes=None):
        """Initialize Gmail service"""
        self.client_secret_path = client_secret_path
        self.token_file = token_file
        self.scopes = scopes or ["https://www.googleapis.com/auth/gmail.modify"]
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate and build Gmail service"""
        creds = None
        
        # Load credentials from token file if it exists
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
        
        # If credentials don't exist or are invalid, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secret_path,
                    self.scopes,
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials for future use
            with open(self.token_file, "w") as token:
                token.write(creds.to_json())
        
        # Build Gmail service
        self.service = build("gmail", "v1", credentials=creds)
        print("✅ Gmail authentication successful!")
    
    def get_unread_messages(self):
        """Fetch unread messages from inbox"""
        try:
            results = self.service.users().messages().list(
                userId="me", 
                labelIds=["INBOX"], 
                q="is:unread"
            ).execute()
            
            messages = results.get("messages", [])
            return messages
        except HttpError as error:
            print(f"❌ Error fetching messages: {error}")
            return []
    
    def get_message_details(self, msg_id):
        """Get full message details"""
        try:
            message = self.service.users().messages().get(
                userId="me", 
                id=msg_id, 
                format="full"
            ).execute()
            return message
        except HttpError as error:
            print(f"❌ Error getting message details: {error}")
            return None
    
    def extract_email_info(self, message):
        """Extract sender, subject, body, and thread_id from message"""
        headers = message["payload"]["headers"]
        
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "")
        thread_id = message["threadId"]
        body = self._get_email_body(message["payload"])
        
        # Clean body (remove quoted text)
        clean_body = re.sub(r"On .* wrote:.*", "", body, flags=re.DOTALL).strip()
        
        return {
            "sender": sender,
            "subject": subject,
            "body": clean_body,
            "thread_id": thread_id
        }
    
    def _get_email_body(self, payload):
        """Extract plain text email body from Gmail API message payload"""
        if "parts" in payload:
            for part in payload["parts"]:
                if part["mimeType"] == "text/plain":
                    data = part["body"].get("data")
                    if data:
                        return base64.urlsafe_b64decode(data).decode("utf-8")
        else:
            data = payload["body"].get("data")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8")
        return ""
    
    def send_reply(self, to, subject, message_text, thread_id):
        """Create and send a reply email"""
        try:
            # Create reply message
            message = MIMEText(message_text)
            message["to"] = to
            message["subject"] = "Re: " + subject
            
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            body = {"raw": raw_message, "threadId": thread_id}
            
            # Send the message
            sent = self.service.users().messages().send(
                userId="me", 
                body=body
            ).execute()
            
            return sent
        except HttpError as error:
            print(f"❌ Error sending reply: {error}")
            return None
    
    def mark_as_read(self, msg_id):
        """Mark message as read"""
        try:
            self.service.users().messages().modify(
                userId="me", 
                id=msg_id, 
                body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            return True
        except HttpError as error:
            print(f"❌ Error marking as read: {error}")
            return False