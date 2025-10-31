# 🤖 Gmail Auto-Reply System with LangGraph

An intelligent, AI-powered email automation system that handles both simple queries and complex multi-turn conversations using LangChain, LangGraph, and OpenAI GPT-4.

## 🌟 Overview

This system automatically processes incoming Gmail messages and generates contextually appropriate responses by:
- Using **LangChain** for straightforward, single-turn queries
- Employing **LangGraph** for complex, multi-turn conversations that require state management
- Leveraging **FAISS vector store** for semantic search of similar past interactions
- Intelligently routing queries based on complexity and follow-up likelihood

**Perfect for:** Customer support teams, help desks, e-commerce businesses, and any organization needing automated email response capabilities with intelligent conversation handling.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GMAIL AUTO-REPLY SYSTEM                          │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │  Gmail API   │
                              │   (Unread    │
                              │   Emails)    │
                              └──────┬───────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │  Email Ingestion       │
                        │  • Extract sender      │
                        │  • Extract subject     │
                        │  • Extract body        │
                        │  • Get thread_id       │
                        └────────┬───────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │  Query Complexity Analysis  │
                    │  • Keyword detection        │
                    │  • Length assessment        │
                    │  • Context evaluation       │
                    └────────┬──────────┬─────────┘
                             │          │
                   ┌─────────┘          └──────────┐
                   │                               │
         [Simple Query]                  [Multi-turn Query]
                   │                               │
                   ▼                               ▼
    ┌──────────────────────────┐    ┌─────────────────────────────┐
    │   LANGCHAIN PIPELINE     │    │   LANGGRAPH STATE MACHINE   │
    │                          │    │                             │
    │  ┌────────────────────┐  │    │  ┌───────────────────────┐ │
    │  │  Vector Retrieval  │  │    │  │   State Definition    │ │
    │  │  (FAISS + OpenAI   │  │    │  │   • messages          │ │
    │  │   Embeddings)      │  │    │  │   • turn_number       │ │
    │  └─────────┬──────────┘  │    │  │   • conversation_     │ │
    │            │              │    │  │     history           │ │
    │            ▼              │    │  │   • needs_followup    │ │
    │  ┌────────────────────┐  │    │  └──────────┬────────────┘ │
    │  │  Prompt Template   │  │    │             │              │
    │  │  + LLM (GPT-4)     │  │    │             ▼              │
    │  └─────────┬──────────┘  │    │  ┌───────────────────────┐ │
    │            │              │    │  │   Graph Workflow      │ │
    │            ▼              │    │  │                       │ │
    │  ┌────────────────────┐  │    │  │  1. Retrieve Context  │ │
    │  │  Response Output   │  │    │  │     ↓                 │ │
    │  └────────────────────┘  │    │  │  2. Generate Response │ │
    │                          │    │  │     ↓                 │ │
    └────────────┬─────────────┘    │  │  3. Check Follow-up   │ │
                 │                  │  │     ↓                 │ │
                 │                  │  │  4. Route Decision    │ │
                 │                  │  │     ├─→ Continue      │ │
                 │                  │  │     └─→ End           │ │
                 │                  │  └───────────┬───────────┘ │
                 │                  │              │             │
                 │                  │    ┌─────────▼──────────┐  │
                 │                  │    │  Memory Saver      │  │
                 │                  │    │  (Checkpointing)   │  │
                 │                  │    └─────────┬──────────┘  │
                 │                  │              │             │
                 └──────────────────┴──────────────┘             │
                                    │                            │
                                    ▼                            │
                      ┌──────────────────────────┐               │
                      │   Response Beautifier    │               │
                      │   • Format paragraphs    │               │
                      │   • Add spacing          │               │
                      │   • Insert signature     │               │
                      └──────────┬───────────────┘               │
                                 │                               │
                                 ▼                               │
                      ┌──────────────────────────┐               │
                      │   Gmail API              │               │
                      │   • Send reply           │               │
                      │   • Mark as read         │               │
                      │   • Maintain thread      │               │
                      └──────────────────────────┘               │
                                                                 │
┌────────────────────────────────────────────────────────────────┘
│
│  VECTOR STORE (FAISS)
│  ┌─────────────────────────────────────────────┐
│  │  Beauty Support Dataset (CSV)               │
│  │  • Customer queries + replies               │
│  │  • Multi-turn conversations                 │
│  │  • Embedded using OpenAI text-embedding     │
│  │  • Similarity search for context retrieval  │
│  └─────────────────────────────────────────────┘
│
└─────────────────────────────────────────────────
```

## ✨ Key Features

### 🎯 Intelligent Query Routing
- **Automatic Detection**: Analyzes incoming queries to determine if they require simple or multi-turn handling
- **Context-Aware**: Considers query length, keywords, and complexity indicators
- **Product Question Recognition**: Identifies straightforward product inquiries and routes them to the fast simple chain

### 🔄 Multi-Turn Conversation Support
- **State Management**: Maintains conversation context across multiple email exchanges
- **LangGraph Integration**: Uses stateful graph-based workflows for complex dialogues
- **Memory Persistence**: Checkpointing ensures conversation continuity
- **Turn Limiting**: Automatically caps conversations at 3 turns to prevent loops

### 🧠 Smart Context Retrieval
- **Vector Similarity Search**: FAISS-powered semantic search finds relevant past interactions
- **Multi-Turn Training Data**: Learns from complete conversation histories, not just single exchanges
- **Top-K Retrieval**: Fetches 3 most similar cases for comprehensive context

### 📧 Professional Email Formatting
- **Auto-Beautification**: Adds proper spacing, paragraphs, and professional signatures
- **Signature Normalization**: Removes duplicate or malformed signatures
- **Readability Enhancement**: Formats sentences for optimal reading experience

### 🔐 Gmail Integration
- **OAuth 2.0 Authentication**: Secure Gmail API access
- **Thread Management**: Maintains email conversation threads
- **Auto-Marking**: Marks processed emails as read
- **Reply Association**: Properly links replies to original emails

## 📁 Project Structure

```
email-bot/
├── config/
│   ├── __init__.py
│   └── config.py                     # Configuration loader (reads from .env)
│
├── src/
│   ├── __init__.py
│   ├── code/
│   │   ├── __init__.py
│   │   ├── email_bot.py              # Main bot logic & orchestration
│   │   └── main.py                   # Entry point with pre-flight checks
│   ├── data/
│   │   └── beauty_support_dataset.csv # Training dataset (place here)
│   └── utils/
│       ├── __init__.py
│       └── gmail_helpers.py          # Gmail API & LangChain utilities
│
├── .env                              # Environment variables (CREATE THIS!)
├── .env.example                      # Template for environment variables
├── .gitignore                        # Protects secrets from git
├── client_secret_*.json              # Google OAuth credentials
├── token.json                        # Generated after first auth (gitignored)
├── LICENSE                           # MIT License
├── PROJECT_STRUCTURE.md              # Detailed structure documentation
├── README.md                         # This file
├── SETUP_GUIDE.md                   # Step-by-step setup instructions
├── SUMMARY.md                        # Project summary and changes
└── requirements.txt                  # Python dependencies
```

### Directory Descriptions

- **`config/`**: Centralized configuration management
  - Loads all settings from `.env` file
  - Validates environment variables
  - Provides default values

- **`src/code/`**: Core application logic
  - `main.py`: Entry point with pre-flight checks
  - `email_bot.py`: Email processing orchestration

- **`src/utils/`**: Utility modules
  - `gmail_helpers.py`: Gmail API wrapper and bot classes

- **`src/data/`**: Dataset storage
  - Place your CSV training data here

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Gmail account with API access enabled
- OpenAI API key
- Google Cloud OAuth 2.0 credentials (`client_secret_*.json`)
- CSV dataset with customer support conversations

### 1. Clone or Setup Project Structure

Create the directory structure as shown above or clone the repository.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your actual values
nano .env  # or use your preferred editor
```

**Required environment variables:**

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small

# File Paths
CSV_PATH=src/data/beauty_support_dataset.csv
CLIENT_SECRET_PATH=client_secret_xxxxx.json
TOKEN_FILE=token.json

# Gmail API Scopes (default provided in config)
```

**Alternative: Export directly**

```bash
export OPENAI_API_KEY='your-openai-api-key'
export CSV_PATH='src/data/beauty_support_dataset.csv'
export CLIENT_SECRET_PATH='client_secret_xxxxx.json'
```

### 4. Add Your Dataset

Place your CSV dataset in `src/data/` directory:

```bash
cp /path/to/your/beauty_support_dataset.csv src/data/
```

### 5. Add Google OAuth Credentials

Download your `client_secret_*.json` from Google Cloud Console and place it in the project root (or update `CLIENT_SECRET_PATH` in `.env`).

### 6. Run the Bot

```bash
# From project root
python src/code/main.py

# Or using module syntax
python -m src.code.main
```

**On first run:**
- Browser will open for Gmail authentication
- Authorize the application
- `token.json` will be created automatically
- Subsequent runs won't require re-authentication

### 7. Monitor Execution

The bot will:
1. ✅ Validate all prerequisites
2. 📚 Load and index your CSV dataset
3. 🔐 Authenticate with Gmail
4. 📬 Check for unread emails
5. 🤖 Generate and send AI replies
6. ✉️ Mark emails as read

```
🚀 Starting Gmail Auto-Reply Bot with LangGraph...

📚 Loading CSV and building vector store...
✅ Vector store setup complete!
✅ Simple chain setup complete!
✅ LangGraph setup complete!
✅ Gmail authentication successful!

📬 Checking for unread emails...
📨 Found 3 unread email(s)

============================================================
Processing email 1/3
============================================================
...
```

## 📊 CSV Dataset Format

Your training dataset should include multi-turn conversation examples:

| Column | Description | Required |
|--------|-------------|----------|
| `Category` | Support category (e.g., "Product Info", "Order Issue") | Yes |
| `Customer_Email` | Initial customer query | Yes |
| `Support_Reply` | First support response | Yes |
| `Customer_Reply_2` | Follow-up query (if any) | Optional |
| `Support_Response_2` | Second support response | Optional |
| `Customer_Reply_3` | Third query (if any) | Optional |
| `Support_Response_3` | Third support response | Optional |

**Example:**
```csv
Category,Customer_Email,Support_Reply,Customer_Reply_2,Support_Response_2
"Product Info","Is your moisturizer suitable for sensitive skin?","Yes, our moisturizer is dermatologically tested...","What ingredients does it contain?","The key ingredients are hyaluronic acid, ceramides..."
```

## 🔧 How It Works

### Step-by-Step Process

1. **Email Fetching**
   - Bot connects to Gmail API
   - Retrieves all unread emails from inbox
   - Extracts sender, subject, body, and thread ID

2. **Query Analysis**
   - Analyzes query characteristics:
     - Strong indicators: "order", "refund", "track", "problem"
     - Weak indicators: "help", "issue", "change"
     - Query length and specificity
     - Product question patterns
   - Routes to appropriate pipeline

3. **Simple Query Handling** (LangChain)
   - Retrieves 3 most similar past queries from FAISS
   - Constructs prompt with context
   - Generates response via GPT-4
   - Returns formatted reply

4. **Multi-Turn Handling** (LangGraph)
   - **State Initialization**: Creates conversation state with empty history
   - **Retrieve Node**: Searches vector store with current query + history
   - **Generate Node**: Creates response using LLM with full context
   - **Check Node**: Analyzes if follow-up is likely needed
   - **Route Node**: Decides to continue loop or end
   - **Memory Saver**: Persists state using thread_id as checkpoint key

5. **Response Beautification**
   - Removes duplicate signatures
   - Adds proper line breaks and spacing
   - Formats paragraphs for readability
   - Appends professional signature

6. **Email Sending**
   - Sends reply via Gmail API
   - Maintains thread association
   - Marks original email as read
   - Logs success/failure

## 📚 Key Components

### `config/config.py`
**Purpose**: Centralized configuration management

**Features**:
- Loads all settings from `.env` file
- Validates required environment variables
- Provides sensible defaults
- Raises clear errors for missing configuration

**Usage**:
```python
from config.config import (
    OPENAI_API_KEY,
    CSV_PATH,
    CLIENT_SECRET_PATH
)
```

### `src/utils/gmail_helpers.py`
**Purpose**: Core bot logic and Gmail API wrapper

**Contains two main classes**:

#### 1. `GmailAutoReplyBot`
Handles AI-powered reply generation using LangChain and LangGraph.

**Key Methods**:
- `__init__()`: Initializes vector store, chains, and graph
- `generate_reply()`: Routes queries to appropriate pipeline
- `is_likely_multiturn()`: Determines if query needs multi-turn handling
- `beautify_email()`: Formats responses professionally
- `continue_conversation()`: Handles follow-up queries in threads

**Components**:
- **Vector Store**: FAISS with OpenAI embeddings
- **Simple Chain**: LangChain LCEL pipeline for straightforward queries
- **LangGraph Workflow**: Stateful graph for complex conversations

#### 2. `GmailService`
Manages all Gmail API operations.

**Key Methods**:
- `_authenticate()`: OAuth 2.0 authentication
- `get_unread_messages()`: Fetches unread emails
- `get_message_details()`: Retrieves full message content
- `extract_email_info()`: Parses sender, subject, body, thread_id
- `send_reply()`: Sends formatted reply in thread
- `mark_as_read()`: Updates email status
- `get_thread_messages()`: Fetches conversation history

### `src/code/email_bot.py`
**Purpose**: Main bot orchestration logic (if using modular structure)

Coordinates between `GmailService` and `GmailAutoReplyBot` to process emails in a structured workflow.

### `src/code/main.py`
**Purpose**: Application entry point with pre-flight checks

**Pre-flight Checks**:
- ✅ Verifies CSV dataset exists
- ✅ Validates client secret file
- ✅ Confirms OpenAI API key is set
- ✅ Checks Python version compatibility
- ❌ Exits gracefully with helpful error messages

**Execution Flow**:
```python
def main():
    # 1. Validate environment
    check_prerequisites()
    
    # 2. Initialize components
    bot = GmailAutoReplyBot(CSV_PATH)
    gmail = GmailService(CLIENT_SECRET_PATH)
    
    # 3. Process emails
    process_unread_emails(bot, gmail)
    
    # 4. Report results
    print_summary()
```

### Multi-Turn Conversation Flow

```
Thread ID: "abc123"

Turn 1:
├─ Customer: "I haven't received my order"
├─ State: {turn_number: 0, needs_followup: true}
├─ Bot: "I apologize for the delay. Could you please provide your order number?"
└─ State saved with thread_id

Turn 2:
├─ Customer: "Order #12345"
├─ State: {turn_number: 1, conversation_history: "...", needs_followup: true}
├─ Bot: "Thank you. I've located your order. It's currently in transit..."
└─ State updated

Turn 3:
├─ Customer: "When will it arrive?"
├─ State: {turn_number: 2, conversation_history: "...", needs_followup: false}
├─ Bot: "Based on tracking, it should arrive by Friday."
└─ Conversation ends (turn limit reached)
```

## ⚙️ Configuration Guide

### Configuration Architecture

The project uses a centralized configuration system in `config/config.py` that loads all settings from environment variables.

### Advanced Configuration

#### Custom Gmail Scopes

```python
# In config/config.py
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",  # Read-only
    "https://www.googleapis.com/auth/gmail.send",      # Send only
]
```

## 🐛 Troubleshooting

### Common Issues & Solutions

**1. "OPENAI_API_KEY not set"**
```bash
# Check if .env file exists
ls -la .env

# Verify contents
cat .env | grep OPENAI_API_KEY

# If using terminal export, reload:
source .env

# Or set directly:
export OPENAI_API_KEY='sk-your-key-here'
```

**2. "CSV file not found"**
```bash
# Check if file exists at specified path
ls -la src/data/beauty_support_dataset.csv

# Verify CSV_PATH in .env matches actual location
cat .env | grep CSV_PATH

# Update .env if needed:
CSV_PATH=src/data/your_actual_file.csv
```

**3. "Client secret file not found"**
```bash
# Verify file exists
ls -la client_secret_*.json

# Check CLIENT_SECRET_PATH in .env
cat .env | grep CLIENT_SECRET_PATH

# Download from Google Cloud Console if missing
# https://console.cloud.google.com/apis/credentials
```

**4. "Gmail authentication failed"**
```bash
# Delete existing token and re-authenticate
rm token.json
python src/code/main.py

# If still failing, check:
# 1. Gmail API is enabled in Google Cloud Console
# 2. OAuth consent screen is configured
# 3. Client secret file is valid and not expired
```

**5. "No unread emails found"**
- Verify emails are in **Inbox** (not Spam or other labels)
- Check emails are marked as **unread**
- Confirm you're authenticated with the correct Gmail account
- Try sending a test email to yourself

**6. "ModuleNotFoundError: No module named 'X'"**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# If using virtual environment:
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

**7. LangGraph/LangChain errors**
```bash
# Upgrade to latest versions
pip install --upgrade langgraph langchain langchain-community langchain-openai

# Check compatibility
pip show langgraph langchain
```

**8. "FAISS index build failed"**
```bash
# Check CSV file encoding
file src/data/beauty_support_dataset.csv

# Try different encoding in code:
df = pd.read_csv(CSV_PATH, encoding='utf-8')  # or 'latin1', 'cp1252'

# Verify CSV has required columns:
# - Category
# - Customer_Email
# - Support_Reply
```

**9. Gmail API Quota Exceeded**
- **Limit**: 250 quota units/second/user
- **Solution**: Implement rate limiting or reduce frequency
- **Check quota**: Google Cloud Console → APIs & Services → Quotas

**10. OpenAI Rate Limit Errors**
```python
# Add retry logic or reduce API calls
import time
from openai import RateLimitError

try:
    response = llm.invoke(prompt)
except RateLimitError:
    time.sleep(20)  # Wait and retry
    response = llm.invoke(prompt)
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
# Add to main.py or email_bot.py
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Environment Validation Script

Create `validate_env.py`:

```python
import os
from pathlib import Path

def validate_environment():
    """Validate all required environment variables and files"""
    
    required_vars = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'CSV_PATH': os.getenv('CSV_PATH'),
        'CLIENT_SECRET_PATH': os.getenv('CLIENT_SECRET_PATH')
    }
    
    print("🔍 Validating environment...\n")
    
    all_valid = True
    
    # Check environment variables
    for var, value in required_vars.items():
        if not value:
            print(f"❌ {var} not set")
            all_valid = False
        else:
            print(f"✅ {var} is set")
    
    # Check files exist
    if required_vars['CSV_PATH']:
        if Path(required_vars['CSV_PATH']).exists():
            print(f"✅ CSV file exists: {required_vars['CSV_PATH']}")
        else:
            print(f"❌ CSV file not found: {required_vars['CSV_PATH']}")
            all_valid = False
    
    if required_vars['CLIENT_SECRET_PATH']:
        if Path(required_vars['CLIENT_SECRET_PATH']).exists():
            print(f"✅ Client secret exists: {required_vars['CLIENT_SECRET_PATH']}")
        else:
            print(f"❌ Client secret not found: {required_vars['CLIENT_SECRET_PATH']}")
            all_valid = False
    
    print()
    if all_valid:
        print("✅ All checks passed! You're ready to run the bot.")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
    
    return all_valid

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    validate_environment()
```

Run validation:
```bash
python validate_env.py
```

## 🚀 Usage Patterns

### Basic Usage

```bash
# Standard execution
python src/code/main.py

# With specific environment
ENV=production python src/code/main.py

# With logging
python src/code/main.py --log-level DEBUG

# Dry run (don't send emails)
python src/code/main.py --dry-run
```

## 📄 License

MIT License - Feel free to use and modify for your projects.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions
- Review the troubleshooting section above

## 🙏 Acknowledgments

- **LangChain**: Framework for LLM applications
- **LangGraph**: Stateful multi-agent workflows
- **OpenAI**: GPT-4 and embeddings API
- **Google**: Gmail API
- **FAISS**: Efficient similarity search

---

**Built with ❤️ using LangChain, LangGraph, and OpenAI GPT-4**

*Last updated: October 2025*