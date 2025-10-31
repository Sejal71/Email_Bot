"""
Gmail Auto-Reply System - Main Execution
Updated for latest LangChain versions
"""

import os
from dotenv import load_dotenv
from src.gmail_helpers import GmailAutoReplyBot, GmailService
from config.config import (
    SCOPES, 
    CSV_PATH, 
    CLIENT_SECRET_PATH, 
    OPENAI_MODEL, 
    EMBEDDING_MODEL, 
    TOKEN_FILE
)


def main():
    """Main function to run the Gmail auto-reply bot"""
    
    print("🚀 Starting Gmail Auto-Reply Bot...\n")
    
    # Check if required files exist
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV file not found: {CSV_PATH}")
        return
    
    if not os.path.exists(CLIENT_SECRET_PATH):
        print(f"❌ Client secret file not found: {CLIENT_SECRET_PATH}")
        return
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Set it using: export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        # Initialize bot and Gmail service
        bot = GmailAutoReplyBot(CSV_PATH, OPENAI_MODEL, EMBEDDING_MODEL)
        gmail = GmailService(CLIENT_SECRET_PATH, TOKEN_FILE, SCOPES)
        
        # Fetch unread messages
        print("\n📬 Checking for unread emails...")
        messages = gmail.get_unread_messages()
        
        if not messages:
            print("✅ No new unread emails found.")
            return
        
        print(f"📨 Found {len(messages)} unread email(s)\n")
        
        # Process each message
        for idx, msg in enumerate(messages, 1):
            msg_id = msg["id"]
            
            print(f"{'='*60}")
            print(f"Processing email {idx}/{len(messages)}")
            print(f"{'='*60}")
            
            # Get message details
            message = gmail.get_message_details(msg_id)
            if not message:
                continue
            
            # Extract email info
            email_info = gmail.extract_email_info(message)
            
            print(f"\n📩 From: {email_info['sender']}")
            print(f"📌 Subject: {email_info['subject']}")
            print(f"📄 Body preview: {email_info['body'][:200]}...\n")
            
            # Generate AI reply
            print("🤖 Generating AI reply...")
            reply_text = bot.generate_reply(email_info['body'])
            
            if not reply_text:
                print("❌ Failed to generate reply. Skipping...\n")
                continue
            
            print(f"\n💬 Generated Reply:\n{'-'*60}")
            print(reply_text)
            print(f"{'-'*60}\n")
            
            # Send reply
            print("📤 Sending reply...")
            sent = gmail.send_reply(
                email_info['sender'],
                email_info['subject'],
                reply_text,
                email_info['thread_id']
            )
            
            if sent:
                # Mark as read
                gmail.mark_as_read(msg_id)
                print("✅ Reply sent and marked as read!\n")
            else:
                print("❌ Failed to send reply\n")
        
        print(f"\n{'='*60}")
        print(f"✅ Processed all {len(messages)} email(s) successfully!")
        print(f"{'='*60}\n")
        
    except Exception as error:
        print(f"\n❌ An unexpected error occurred: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    load_dotenv()
    main()