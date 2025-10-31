"""
Gmail Auto-Reply System - Configuration
Loads configuration from environment variables
"""

import os

# Gmail API Scopes
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# File Paths (from environment variables)
CSV_PATH = os.getenv("CSV_PATH", "beauty_support_dataset.csv")
CLIENT_SECRET_PATH = os.getenv("CLIENT_SECRET_PATH", "client_secret.json")
TOKEN_FILE = os.getenv("TOKEN_FILE", "token.json")

# OpenAI Configuration (from environment variables with defaults)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")