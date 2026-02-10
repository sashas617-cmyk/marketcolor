#!/usr/bin/env python3
"""
Daily Market Pulse Bot - Enhanced Version
Uses Tavily for deep multi-angle search + GPT-5.2 Pro for reasoning/synthesis.
No GPT web search - all searching outsourced to Tavily for better control.
"""

import os
import json
import requests
from datetime import datetime
from openai import OpenAI
from tavily import TavilyClient

# Configuration
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


def init_clients():
    """Initialize API clients."""
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return openai_client, tavily_client


# ---------------------------------------------------------------------------
# STAGE 0: GPT reasons about what to search for right now
# ---------------------------------------------------------------------------
def stage0_generate_queries(openai_client):
    """GPT-5.2 Pro generates targeted search queries based on current context."""
    now = datetime.utcnow()
    today = now.strftime("%A, %B %d, %Y %H:%M UTC")
    hour = now.hour

    response = openai_client.responses.create(
        model="gpt-5.2-pro",
        input=f"""Today is {today}. Current UTC hour: {hour}.

You are a financial markets research director planning the morning search sweep.
Generate exactly 3 highly specific, timely search queries that would surface the most
market-moving stories RIGHT NOW that a generic search would miss.

Think about:
- Day of week context (Monday = weekend catch-up, Friday = positioning/flows)
- Earnings season timing - any major companies reporting this week?
- Central bank meeting schedule (Fed, ECB, BOJ, BOE)
- Known geopolitical hotspots and trade negotiations
- Which markets are currently open based on the UTC hour
- Seasonal factors (end of quarter rebalancing, options expiry, etc.)

Return ONLY a JSON array of 3 query strings. No explanation. Example:
["Fed FOMC meeting decision interest rate impact", "NVIDIA earnings Q4 results guidance", "China trade tariffs semiconductor restriction"]"""
    )

    try:
        queries = json.loads(response.output_text)
        if isinstance(queries, list):
            return queries[:3]
    except Exception:
        pass

    return [
        "most important financial market news today",
        "major earnings reports economic data releases today",
        "geopolitical events affecting global markets today",
    ]


# ---------------------------------------------------------------------------
# STAGE 1: Parallel Tavily searches across multiple angles
# ---------------------------------------------------------------------------
def stage1_tavily_searches(tavily_client, bonus_queries):
    """Run targeted Tavily searches across multiple angles."""
    all_results = {}

    fixed_searches = [
        {
            "name": "market_news",
            "query": "most important market moving financial news today stocks bonds",
            "topic": "finance",
            "search_depth": "advanced",
            "max_results": 8,
            "time_range": "day",
        },
        {
            "name": "macro_geopolitics",
            "query": "Federal Reserve central bank economic data inflation trade policy tariffs",
            "topic": "news",
            "search_depth": "advanced",
            "max_results": 6,
            "days": 1,
        },
        {
            "name": "commodities_crypto",
            "query": "oil gold commodities bitcoin crypto major price moves",
            "topic": "finance",
            "search_depth": "basic",
            "max_results": 5,
            "time_range": "day",
        },
        {
            "name": "social_twitter",
            "query": "stock market financial markets trending takes analysis",
            "topic": "news",
            "search_depth": "advanced",
            "include_domains": ["x.com"],
            "max_results": 5,
            "days": 1,
        },
        {
            "name": "social_reddit",
            "query": "stock market investing trading sentiment today",
            "search_depth": "basic",
            "include_domains": ["reddit.com"],
            "max_results": 5,
        },
    ]

    def run_search(config):
        """Execute a single Tavily search with error handling."""
        name = config.pop("name")
        query = config.pop("query")
        try:
            response = tavily_client.search(query=query, **config)
            results = response.get("results", [])
            print(f"  {name}: {len(results)} results")
            return name, results
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            return name, []

    for search_config in fixed_searches:
        name, results = run_search(dict(search_config))
        all_results[name] = results

    for i, query in enumerate(bonus_queries):
        config = {
            "name": f"gpt_query_{i}",
            "query": query,
            "topic": "news",
            "search_depth": "advanced",
            "max_results": 5,
            "days": 1,
        }
        name, results = run_search(config)
        all_results[name] = results

    return all_results


# ---------------------------------------------------------------------------
# STAGE 2: GPT first-pass analysis - rank, flag, identify dig-deeper targets
# ---------------------------------------------------------------------------
def stage2_first_pass(openai_client, search_results):
    """GPT-5.2 Pro analyzes all Tavily results and produces structured analysis."""

    context_parts = []
    for category, items in search_results.items():
        is_social = "social" in category or "twitter" in category or "reddit" in category
        source_tag = " [SOCIAL MEDIA]" if is_social else ""
        context_parts.append(f"\n--- {category.upper()}{source_tag} ---")
        for item in items:
            entry = f"Title: {item.get('title', 'N/A')}"
            entry += f"\nURL: {item.get('url', '')}"
            entry += f"\nSnippet: {item.get('content', '')[:400]}"
            if item.get("published_date"):
                entry += f"\nPublished: {item['published_date']}"
            context_parts.append(entry)

    context = "\n".join(context_parts)

    response = openai_client.responses.create(
        model="gpt-5.2-pro",
        input=f"""=== TAVILY SEARCH RESULTS ===
{context}

You are a senior financial analyst. Analyze these search results and produce a structured assessment.

TASKS:
1. RANK: Identify the 12-15 most market-relevant stories. Score by potential impact on equities, bonds, FX, commodities. Deduplicate similar stories.
2. DIG DEEPER: Pick 3-4 stories that deserve deeper investigation - could be bigger than they appear, or where more context would significantly improve the briefing.
3. FACT CHECK: Flag ANY stories sourced from social media (x.com, reddit.com) that make specific factual claims about earnings, data releases, deals, or market events. These MUST be verified before inclusion.

Return ONLY valid JSON:
{{
  "top_stories": [
    {{"title": "...", "summary": "one sentence", "source_url": "...", "impact": "high/medium/low", "is_social": false}}
  ],
  "dig_deeper": [
    {{"topic": "...", "search_query": "specific Tavily search query", "why": "..."}}
  ],
  "fact_check": [
    {{"claim": "the specific claim to verify", "original_source": "x.com or reddit.com", "verify_query": "search query targeting credible sources"}}
  ]
}}"""
    )

    try:
        return json.loads(response.output_text)
    except Exception:
        text = response.output_text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
        print("  WARNING: Could not parse Stage 2 JSON, using empty analysis")
        return {"top_stories": [], "dig_deeper": [], "fact_check": []}


# ---------------------------------------------------------------------------
# STAGE 3: Tavily verification + deep dives
# ---------------------------------------------------------------------------
CREDIBLE_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "cnbc.com",
    "ft.com",
    "wsj.com",
    "apnews.com",
    "bbc.com",
    "marketwatch.com",
    "finance.yahoo.com",
]


def stage3_verify_and_deepen(tavily_client, analysis):
    """Run verification for social media claims and deep dives on key stories."""
    additional = {}

    for i, item in enumerate(analysis.get("dig_deeper", [])[:4]):
        query = item.get("search_query", "")
        if not query:
            continue
        try:
            response = tavily_client.search(
                query=query,
                topic="news",
                search_depth="advanced",
                max_results=4,
                days=2,
            )
            results = response.get("results", [])
            additional[f"deep_{i}"] = {
                "topic": item.get("topic", ""),
                "why": item.get("why", ""),
                "results": results,
            }
            print(f"  Deep dive '{item.get('topic', '')[:40]}': {len(results)} results")
        except Exception as e:
            print(f"  Deep dive {i} error: {e}")

    for i, item in enumerate(analysis.get("fact_check", [])[:3]):
        query = item.get("verify_query", "")
        if not query:
            continue
        try:
            response = tavily_client.search(
                query=query,
                topic="news",
                search_depth="basic",
                max_results=3,
                include_domains=CREDIBLE_DOMAINS,
                days=2,
            )
            sources = response.get("results", [])
            verified = len(sources) > 0
            additional[f"verify_{i}"] = {
                "claim": item.get("claim", ""),
                "original_source": item.get("original_source", ""),
                "verified": verified,
                "credible_sources": sources,
            }
            tag = "CONFIRMED" if verified else "UNVERIFIED"
            print(f"  Fact check: {tag} - '{item.get('claim', '')[:50]}'")
        except Exception as e:
            print(f"  Fact check {i} error: {e}")

    return additional


# ---------------------------------------------------------------------------
# STAGE 4: GPT final synthesis into polished briefing
# ---------------------------------------------------------------------------
def stage4_final_synthesis(openai_client, analysis, additional):
    """GPT-5.2 Pro writes the final market briefing from all gathered intelligence."""
    now = datetime.utcnow().strftime("%A, %B %d, %Y %H:%M UTC")

    parts = [f"Date: {now}\n"]

    parts.append("=== RANKED STORIES ===")
    for story in analysis.get("top_stories", []):
        social_tag = " [FROM SOCIAL MEDIA]" if story.get("is_social") else ""
        parts.append(
            f"- [{story.get('impact', 'medium').upper()}]{social_tag} "
            f"{story.get('title', '')}: {story.get('summary', '')}"
        )

    parts.append("\n=== DEEP DIVE FINDINGS ===")
    for key, data in additional.items():
        if not key.startswith("deep_"):
            continue
        parts.append(f"\nTopic: {data.get('topic', '')} (Why: {data.get('why', '')})")
        for r in data.get("results", []):
            parts.append(f"  - {r.get('title', '')}: {r.get('content', '')[:250]}")

    parts.append("\n=== SOCIAL MEDIA VERIFICATION RESULTS ===")
    has_verifications = False
    for key, data in additional.items():
        if not key.startswith("verify_"):
            continue
        has_verifications = True
        status = "CONFIRMED by credible sources" if data.get("verified") else "NOT CONFIRMED by major outlets"
        parts.append(f"- Claim: \"{data.get('claim', '')}\" -> {status}")
        for s in data.get("credible_sources", []):
            parts.append(f"  Confirming source: {s.get('title', '')} ({s.get('url', '')})")
    if not has_verifications:
        parts.append("No social media claims required verification.")

    context = "\n".join(parts)

    response = openai_client.responses.create(
        model="gpt-5.2-pro",
        input=f"""{context}

Write the Daily Market Pulse briefing based on ALL the intelligence above.

FORMAT:
1. Executive summary: 2-3 sentences capturing overall market mood and the single biggest theme.
2. 10 numbered stories using emoji numbers (1\u{fe0f}\u{20e3} through \u{1f51f}).
3. Each story: 2-3 punchy sentences with specific names, numbers, percentages.
4. For verified social media stories, note "confirmed by [source name]".
5. For unverified social buzz that's still interesting, label it "Social Buzz (unverified)" so readers know the credibility level.
6. Include at least 1 social/sentiment story if anything noteworthy was found.
7. End with one "Key Watch:" line for what to monitor next.

TONE: Like a sharp morning briefing from a senior analyst who also checks Twitter.
Professional but engaging. No filler. Every sentence earns its place."""
    )

    return response.output_text


# ---------------------------------------------------------------------------
# Telegram delivery
# ---------------------------------------------------------------------------
def send_telegram_message(message):
    """Send message to Telegram, splitting at newlines if over 4000 chars."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    max_len = 4000

    now = datetime.utcnow().strftime("%B %d, %Y %H:%M UTC")
    full_message = f"Daily Market Pulse \u2014 {now}\n\n{message}"

    if len(full_message) <= max_len:
        resp = requests.post(url, json={"chat_id": CHAT_ID, "text": full_message}, timeout=30)
        return resp.json()

    chunks, current = [], ""
    for line in full_message.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            if current:
                chunks.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current)

    result = None
    for chunk in chunks:
        result = requests.post(url, json={"chat_id": CHAT_ID, "text": chunk}, timeout=30)
    return result.json() if result else {"ok": False}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Daily Market Pulse Bot (Enhanced)")
    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    missing = []
    for var in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "OPENAI_API_KEY", "TAVILY_API_KEY"]:
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        return False

    openai_client, tavily_client = init_clients()

    print("\nStage 0: Generating targeted search queries...")
    bonus_queries = stage0_generate_queries(openai_client)
    for q in bonus_queries:
        print(f"   -> {q}")

    print("\nStage 1: Running Tavily searches...")
    search_results = stage1_tavily_searches(tavily_client, bonus_queries)
    total = sum(len(v) for v in search_results.values())
    print(f"   Total results gathered: {total}")

    print("\nStage 2: GPT analyzing and ranking stories...")
    analysis = stage2_first_pass(openai_client, search_results)
    print(f"   Top stories: {len(analysis.get('top_stories', []))}")
    print(f"   Dig deeper targets: {len(analysis.get('dig_deeper', []))}")
    print(f"   Fact checks needed: {len(analysis.get('fact_check', []))}")

    print("\nStage 3: Deep dives and fact-checking...")
    additional = stage3_verify_and_deepen(tavily_client, analysis)

    print("\nStage 4: Generating final briefing...")
    briefing = stage4_final_synthesis(openai_client, analysis, additional)

    print("\nSending to Telegram...")
    result = send_telegram_message(briefing)

    if result.get("ok"):
        print("Message sent successfully!")
        return True
    else:
        print(f"Send error: {result}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
