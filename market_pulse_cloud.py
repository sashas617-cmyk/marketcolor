#!/usr/bin/env python3
"""
Daily Market Pulse Bot - Cloud Version
Uses GPT-5.2 Pro with web search for real-time market analysis.
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
    """Get market analysis from GPT-5.2 Pro with real-time web search."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = """Provide a market briefing with this structure:

FIRST: Start with a 2-3 sentence executive summary capturing the overall market mood and the most important themes driving markets right now.

THEN: Tell me 10 things/stories I shouldn't miss right now that may affect financial markets - including but not limited to Macro events, US and global equities, global bonds, commodities, geopolitics, earnings, social media sentiment, political events.

Make it impactful yet engaging. Keep each point concise with specific numbers and names where relevant."""

    # Use Responses API with GPT-5.2 Pro and web search tool
    response = client.responses.create(
        model="gpt-5.2-pro",
        tools=[{"type": "web_search"}],
        input=prompt
    )

    return response.output_text

def send_telegram_message(message):
    """Send message to Telegram group, splitting at newlines if needed."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    max_len = 4000

    if len(message) > max_len:
        # Split at newlines to avoid breaking mid-sentence
        chunks = []
        current_chunk = ""

        for line in message.split("\n"):
            if len(current_chunk) + len(line) + 1 > max_len:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk = current_chunk + "\n" + line if current_chunk else line

        if current_chunk:
            chunks.append(current_chunk)

        result = None
        for chunk in chunks:
            payload = {
                "chat_id": CHAT_ID,
                "text": chunk
            }
            result = requests.post(url, json=payload, timeout=30)
        return result.json() if result else {"ok": False}
    else:
        payload = {
            "chat_id": CHAT_ID,
            "text": message
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

    print("Getting market analysis from GPT-5.2 Pro with web search...")
    analysis = get_market_analysis()

    today = datetime.now().strftime("%B %d, %Y %H:%M UTC")

    message = f"Daily Market Pulse - {today}\n\n"
    message += analysis

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
