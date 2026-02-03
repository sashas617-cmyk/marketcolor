#!/usr/bin/env python3
"""
Daily Market Pulse Bot - Cloud Version
Uses GPT-5.2 Pro to generate daily market analysis.
"""

import os
import requests
from openai import OpenAI
from datetime import datetime

# Get secrets from environment variables
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_market_analysis():
    """Get market analysis from GPT-5.2 Pro."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = """Tell me 10 things I shouldn't miss in this market today relevant for investment decisions.

Cover:
- US and global stocks
- Bonds and interest rates
- Geopolitical developments
- Key economic data releases
- Earnings reports
- Sector movements
- Commodities (oil, gold, etc.)
- Currency movements
- Market sentiment indicators
- Any breaking news that could impact markets

Be specific with numbers, percentages, and company names where relevant. Keep each point concise but informative."""

    response = client.chat.completions.create(
        model="gpt-5.2-pro",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )

    return response.choices[0].message.content

def send_telegram_message(message):
    """Send message to Telegram group."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    # Telegram has a 4096 character limit, so we may need to split
    if len(message) > 4000:
        chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
        for chunk in chunks:
            payload = {
                "chat_id": CHAT_ID,
                "text": chunk,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, json=payload, timeout=30)
        return response.json()
    else:
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload, timeout=30)
        return response.json()

def main():
    if not BOT_TOKEN or not CHAT_ID:
        print("Error: Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return False

    if not OPENAI_API_KEY:
        print("Error: Missing OPENAI_API_KEY")
        return False

    print("Getting market analysis from GPT-5.2 Pro...")
    analysis = get_market_analysis()

    today = datetime.now().strftime("%B %d, %Y")

    message = f"ð *Daily Market Pulse*\n"
    message += f"ð {today}\n\n"
    message += analysis
    message += "\n\n_Powered by GPT-5.2 Pro via GitHub Actions_"

    print("Sending to Telegram...")
    result = send_telegram_message(message)

    if result.get("ok"):
        print("Message sent successfully!")
        return True
    else:
        print(f"Error: {result}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
