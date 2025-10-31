# Quick Setup Guide

## ✅ You're absolutely right!

Sensitive information like API keys and file paths should be in `.env` files, not hardcoded. I've updated the project to follow security best practices.

```python
# config.py - READS FROM .env (GOOD!)
CSV_PATH = os.getenv("CSV_PATH", "beauty_support_dataset.csv")
CLIENT_SECRET_PATH = os.getenv("CLIENT_SECRET_PATH", "client_secret.json")
```

## 🚀 Quick Start

### Step 1: Copy `.env.example` to `.env`
```bash
cp .env.example .env
```

### Step 2: Edit `.env` with your actual values
```env
# Required
OPENAI_API_KEY=sk-your-actual-openai-key-here
CSV_PATH=C:\Users\Admin\Downloads\beauty_support_dataset.csv
CLIENT_SECRET_PATH=C:\Users\Admin\Downloads\client_secret_597965791909-xxxxx.json

# Optional (defaults provided)
TOKEN_FILE=token.json
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small
```

### Step 3: Run the bot
```bash
python main.py
```

## 🔒 Security Benefits

1. **No secrets in code** - All sensitive data in `.env`
2. **`.gitignore` included** - Prevents accidental commits
3. **Template provided** - `.env.example` shows structure without exposing secrets
4. **Easy to share** - Share code safely without exposing credentials

## 📁 Updated Files

- **`.env.example`** - Template for your environment variables
- **`.gitignore`** - Protects sensitive files from git
- **`config.py`** - Now reads from environment variables
- **`README.md`** - Updated setup instructions

## ⚠️ Important

**Never commit these files:**
- `.env` (your actual credentials)
- `client_secret_*.json` (Google OAuth credentials)
- `token.json` (Gmail auth token)
- `*.csv` (if contains sensitive data)

The `.gitignore` file is already configured to protect these files!
