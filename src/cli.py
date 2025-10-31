"""
CLI Interface for Gmail Auto-Reply System
-----------------------------------------
Run commands like:
    python cli.py --check       # Check Gmail connection
    python cli.py --run         # Run the full auto-reply workflow
    python cli.py --status      # Show unread email count
"""

import argparse
import os
from dotenv import load_dotenv
from src.gmail_helpers import GmailAutoReplyBot, GmailService
from config.config import (
    SCOPES,
    CSV_PATH,
    CLIENT_SECRET_PATH,
    OPENAI_MODEL,
    EMBEDDING_MODEL,
    TOKEN_FILE,
)


def check_requirements():
    """Validate required files and environment variables."""
    print("🔍 Checking project configuration...\n")

    missing = False

    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV file not found: {CSV_PATH}")
        missing = True
    else:
        print(f"✅ CSV found: {CSV_PATH}")

    if not os.path.exists(CLIENT_SECRET_PATH):
        print(f"❌ Client secret file not found: {CLIENT_SECRET_PATH}")
        missing = True
    else:
        print(f"✅ Client secret found: {CLIENT_SECRET_PATH}")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment!")
        missing = True
    else:
        print("✅ OPENAI_API_KEY found")

    if missing:
        print("\n⚠️  Fix the above issues before running the bot.\n")
        return False

    print("\n✅ All checks passed.\n")
    return True


def run_bot():
    """Run the full Gmail Auto-Reply process."""
    print("🚀 Starting Gmail Auto-Reply Bot...\n")

    bot = GmailAutoReplyBot(CSV_PATH, OPENAI_MODEL, EMBEDDING_MODEL)
    gmail = GmailService(CLIENT_SECRET_PATH, TOKEN_FILE, SCOPES)

    print("📬 Checking for unread emails...")
    messages = gmail.get_unread_messages()

    if not messages:
        print("✅ No new unread emails found.")
        return

    print(f"📨 Found {len(messages)} unread email(s)\n")

    for idx, msg in enumerate(messages, 1):
        print(f"{'=' * 60}")
        print(f"Processing email {idx}/{len(messages)}")
        print(f"{'=' * 60}")

        msg_id = msg["id"]
        message = gmail.get_message_details(msg_id)
        if not message:
            continue

        email_info = gmail.extract_email_info(message)
        print(f"\n📩 From: {email_info['sender']}")
        print(f"📌 Subject: {email_info['subject']}")
        print(f"📄 Body preview: {email_info['body'][:150]}...\n")

        print("🤖 Generating AI reply...")
        reply_text = bot.generate_reply(email_info['body'])
        if not reply_text:
            print("❌ Failed to generate reply. Skipping...\n")
            continue

        print(f"\n💬 Generated Reply:\n{'-' * 60}")
        print(reply_text)
        print(f"{'-' * 60}\n")

        print("📤 Sending reply...")
        sent = gmail.send_reply(
            email_info["sender"],
            email_info["subject"],
            reply_text,
            email_info["thread_id"],
        )

        if sent:
            gmail.mark_as_read(msg_id)
            print("✅ Reply sent and email marked as read!\n")
        else:
            print("❌ Failed to send reply\n")

    print(f"\n{'=' * 60}")
    print("🎉 All unread emails processed successfully!")
    print(f"{'=' * 60}\n")


def show_status():
    """Display how many unread messages exist."""
    gmail = GmailService(CLIENT_SECRET_PATH, TOKEN_FILE, SCOPES)
    messages = gmail.get_unread_messages()
    print(f"📨 You currently have {len(messages)} unread email(s).")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="CLI tool for running Gmail Auto-Reply Bot"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check environment and file setup"
    )
    parser.add_argument("--run", action="store_true", help="Run the full email bot")
    parser.add_argument(
        "--status", action="store_true", help="Show unread email count"
    )

    args = parser.parse_args()

    if args.check:
        check_requirements()
    elif args.status:
        show_status()
    elif args.run:
        if check_requirements():
            run_bot()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
