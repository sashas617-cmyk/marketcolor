#!/usr/bin/env python3
"""
Daily Market Pulse Bot - Cloud Version
Runs on GitHub Actions for automated market analysis.
"""

import os
import requests
from datetime import datetime

# Get secrets from environment variables
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def get_market_data():
    """Fetch current market data from Yahoo Finance API."""
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC",
        "10Y Treasury": "^TNX",
        "VIX": "^VIX",
        "Gold": "GC=F",
        "Oil (WTI)": "CL=F"
    }

    market_info = []

    for name, symbol in indices.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()

            result = data.get("chart", {}).get("result", [{}])[0]
            meta = result.get("meta", {})
            price = meta.get("regularMarketPrice", 0)
            prev_close = meta.get("previousClose", price)

            if prev_close and prev_close != 0:
                change_pct = ((price - prev_close) / prev_close) * 100
                direction = "up" if change_pct >= 0 else "down"
                emoji = "\U0001F7E2" if change_pct >= 0 else "\U0001F534"
                market_info.append(f"{emoji} {name}: {price:.2f} ({change_pct:+.2f}%)")
        except Exception as e:
            market_info.append(f"\u26A0\uFE0F {name}: Data unavailable")

    return market_info

def generate_analysis(market_data):
    """Generate market analysis summary."""
    today = datetime.now().strftime("%B %d, %Y")

    analysis = f"\U0001F4CA *Daily Market Pulse*\n"
    analysis += f"\U0001F4C5 {today}\n\n"
    analysis += "*Market Overview:*\n"
    analysis += "\n".join(market_data)
    analysis += "\n\n_Automated daily update via GitHub Actions_"

    return analysis

def send_telegram_message(message):
    """Send message to Telegram group."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
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

    print("Fetching market data...")
    market_data = get_market_data()

    print("Generating analysis...")
    analysis = generate_analysis(market_data)

    print("Sending to Telegram...")
    result = send_telegram_message(analysis)

    if result.get("ok"):
        print("Message sent successfully!")
        return True
    else:
        print(f"Error: {result}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
