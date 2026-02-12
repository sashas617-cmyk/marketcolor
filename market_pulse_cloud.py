#!/usr/bin/env python3
"""
Daily Market Pulse Bot - Enhanced Version v4
Uses Tavily for deep multi-angle search + Benzinga for professional financial news
+ GPT-5.2 Pro for reasoning/synthesis.

v4 changes:
- Added StockTwits/fintwit aggregator searches for Twitter trending
- Dedicated trade policy/tariffs search for better geopolitics coverage
- Alpha stories must be EVENTS not how-to guides
- Tightened dedup + diversity rules
"""

import os
import json
import re
import requests
from datetime import datetime
from openai import OpenAI
from tavily import TavilyClient
from pathlib import Path

# Configuration
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
BENZINGA_API_KEY = os.environ.get("BENZINGA_API_KEY", "")

HISTORY_FILE = Path(__file__).parent / "history.json"


def init_clients():
    """Initialize API clients."""
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return openai_client, tavily_client


def load_history():
    """Load previously covered stories to avoid repetition across runs."""
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        stories = data.get("stories", [])
        tickers = data.get("tickers", [])
        print(f"  Loaded history: {len(stories)} stories, {len(tickers)} tickers from {data.get('last_run', 'unknown')}")
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        print("  No history file found, starting fresh")
        return {"stories": [], "tickers": [], "last_run": None}


def save_history(briefing_text):
    """Extract story headlines and tickers from briefing, save for next run."""
    stories = []
    for line in briefing_text.split("\n"):
        if "**" in line and line.count("**") >= 2:
            start = line.index("**") + 2
            end = line.index("**", start)
            headline = line[start:end].strip()
            if len(headline) > 15 and ":" in headline:
                stories.append(headline)
    tickers = []
    for word in briefing_text.split():
        clean = word.lstrip("$").rstrip(".,;:!()")
        if word.startswith("$") and clean.isalpha() and 1 <= len(clean) <= 5:
            if clean.upper() not in tickers:
                tickers.append(clean.upper())
    tickers.sort()
    data = {
        "last_run": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stories": stories,
        "tickers": tickers,
    }
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved history: {len(stories)} stories, {len(tickers)} tickers")


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

You are a global financial markets research director. Your job: figure out the 12 most
important things happening RIGHT NOW that could affect financial markets, politics, and geopolitics.

Think globally. What happened overnight in Asia? What is Europe doing? Any US pre-market movers?
Earnings surprises? Central bank signals? Geopolitical developments? Commodity moves?
Unusual flows or positioning? Emerging market events? Trade policy shifts?

DO NOT default to generic US-centric queries like "Federal Reserve" or "CPI" unless those are
genuinely the biggest story right now. Think about what a sophisticated global macro trader
sitting in Dubai at this hour would need to know.

Generate exactly 12 search queries designed to surface the freshest, most market-relevant
news across ALL regions and asset classes. Each query should be specific and timely.
Include "today" or "latest" or "February 2026" in queries to bias toward fresh results.

Return ONLY a JSON object. No explanation. Example:
{{"queries": ["Japan yen surge Nikkei record today", "China CPI PPI inflation data latest", "Ukraine Russia overnight attack energy infrastructure", "US pre-market movers earnings surprises today", "oil price IEA demand forecast latest", "Middle East Iran negotiations latest", "European markets DAX economic data today", "semiconductor earnings AI infrastructure today", "central bank rate decision emerging markets latest", "unusual options activity large block trades today", "insider buying SEC Form 4 filings notable today", "trade tariffs policy announcement latest today"]}}"""
    )

    try:
        data = json.loads(response.output_text)
        if isinstance(data, dict):
            queries = data.get("queries", [])[:12]
            if queries:
                return queries[:8], queries[8:]
            return data.get("mainstream", [])[:3], data.get("alpha", [])[:2]
    except Exception:
        pass

    # Fallback queries
    return (
        [
            "most important financial market news today global",
            "Asia markets overnight Nikkei Hang Seng moves today",
            "Europe markets DAX economic data today",
            "US pre-market movers earnings surprises today",
            "geopolitical developments affecting markets today",
            "oil gold commodities major price moves today",
            "central bank rate decisions inflation data latest",
            "trade policy tariffs sanctions latest today",
        ],
        [
            "unusual options activity large block trades today",
            "insider buying selling SEC Form 4 filings notable today",
            "emerging markets currency moves developments today",
            "AI semiconductor infrastructure earnings latest today",
        ],
    )




# ---------------------------------------------------------------------------
# BENZINGA: Fetch professional financial news
# ---------------------------------------------------------------------------
def _clean_html(text):
    """Remove HTML tags from text."""
    if not text:
        return ""
    return re.sub(r'<[^>]+>', '', text).strip()


def fetch_benzinga_news(limit=200):
    """Fetch latest news from Benzinga API and normalize to Tavily result format."""
    if not BENZINGA_API_KEY:
        print("  benzinga: SKIPPED (no API key)")
        return []

    url = "https://api.massive.com/benzinga/v2/news"
    params = {
        "apiKey": BENZINGA_API_KEY,
        "limit": limit,
        "sort": "published.desc",
    }

    try:
        resp = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=15)
        if resp.status_code != 200:
            print(f"  benzinga: API error {resp.status_code}")
            return []

        data = resp.json()
        articles = data.get("results", []) if isinstance(data, dict) else data

        # Filter out routine SEC filings, prospectus supplements, and minor offerings
        SKIP_KEYWORDS = [
            "prospectus supplement", "filed with sec", "shelf registration",
            "intends to offer", "prices offering", "announces offering",
            "announces pricing", "closes offering", "files prospectus",
            "supplemental listing", "form 8-k", "form 10-k", "form 10-q",
        ]
        filtered = []
        for a in articles:
            title_lower = (a.get("title", "") or "").lower()
            teaser_lower = (a.get("teaser", "") or "").lower()
            if any(kw in title_lower + " " + teaser_lower for kw in SKIP_KEYWORDS):
                continue
            filtered.append(a)
        print(f"  benzinga: filtered {len(articles) - len(filtered)} routine filings")
        articles = filtered

        normalized = []
        for a in articles:
            title = _clean_html(a.get("title", ""))
            teaser = _clean_html(a.get("teaser", ""))
            body_snippet = _clean_html(a.get("body", ""))[:400] if a.get("body") else ""
            article_url = a.get("url", "")
            author = a.get("author", "Benzinga")
            created = a.get("created", a.get("updated", ""))

            stocks = a.get("stocks", [])
            ticker_names = [s.get("name", "") for s in stocks if isinstance(s, dict)]
            ticker_str = ", ".join(ticker_names[:5])

            content_parts = []
            if ticker_str:
                content_parts.append(f"Tickers: {ticker_str}")
            if teaser:
                content_parts.append(teaser)
            elif body_snippet:
                content_parts.append(body_snippet)
            if author:
                content_parts.append(f"Source: {author}")

            normalized.append({
                "title": title,
                "url": article_url,
                "content": " | ".join(content_parts),
                "published_date": created,
            })

        print(f"  benzinga: {len(normalized)} articles")
        return normalized

    except Exception as e:
        print(f"  benzinga: ERROR - {e}")
        return []


# ---------------------------------------------------------------------------
# STAGE 1: Parallel Tavily searches across multiple angles
# ---------------------------------------------------------------------------
def stage1_tavily_searches(tavily_client, mainstream_queries, alpha_queries):
    """Run targeted Tavily searches across multiple angles including social/alpha."""
    all_results = {}

    # --- SAFETY-NET SEARCHES (broad; GPT queries do the real work) ---
    mainstream_searches = [
        {
            "name": "global_markets",
            "query": "most important global financial market news today Asia Europe US",
            "topic": "finance",
            "search_depth": "advanced",
            "max_results": 7,
            "time_range": "day",
        },
        {
            "name": "geopolitics_macro",
            "query": "geopolitics trade policy central banks commodities major developments today",
            "topic": "news",
            "search_depth": "advanced",
            "max_results": 5,
            "days": 1,
        },
    ]

    # --- SOCIAL (kept lean) ---
    social_searches = [
        {
            "name": "fintwit_movers",
            "query": "stock market unusual move breaking catalyst today",
            "search_depth": "basic",
            "max_results": 5,
            "time_range": "day",
        },
    ]

    # --- ALPHA / EDGE (kept lean) ---
    alpha_searches = [
        {
            "name": "unusual_options",
            "query": "unusual options activity large block trade sweep today",
            "topic": "finance",
            "search_depth": "advanced",
            "max_results": 5,
            "days": 1,
        },
        {
            "name": "insider_trades",
            "query": "SEC insider buying selling notable Form 4 filing today",
            "topic": "finance",
            "search_depth": "advanced",
            "max_results": 5,
            "days": 2,
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

    # Run all search categories
    print("  [Mainstream]")
    for search_config in mainstream_searches:
        name, results = run_search(dict(search_config))
        all_results[name] = results

    print("  [Social/Fintwit]")
    for search_config in social_searches:
        name, results = run_search(dict(search_config))
        all_results[name] = results

    print("  [Alpha/Edge]")
    for search_config in alpha_searches:
        name, results = run_search(dict(search_config))
        all_results[name] = results

    # Run GPT-generated mainstream queries
    print("  [GPT Mainstream Queries]")
    for i, query in enumerate(mainstream_queries):
        config = {
            "name": f"gpt_mainstream_{i}",
            "query": query,
            "topic": "news",
            "search_depth": "advanced",
            "max_results": 5,
            "days": 1,
        }
        name, results = run_search(config)
        all_results[name] = results

    # Run GPT-generated alpha queries
    print("  [GPT Alpha Queries]")
    for i, query in enumerate(alpha_queries):
        config = {
            "name": f"gpt_alpha_{i}",
            "query": query,
            "topic": "finance",
            "search_depth": "advanced",
            "max_results": 5,
            "days": 1,
        }
        name, results = run_search(config)
        all_results[name] = results

    # Benzinga professional financial news feed
    print("  [Benzinga Pro]")
    benzinga_results = fetch_benzinga_news(limit=200)
    if benzinga_results:
        all_results["benzinga_pro"] = benzinga_results

    return all_results


# ---------------------------------------------------------------------------
# STAGE 2: GPT first-pass analysis - rank, flag, identify dig-deeper targets
# ---------------------------------------------------------------------------
def stage2_first_pass(openai_client, search_results):
    """GPT-5.2 Pro analyzes all Tavily results and produces structured analysis."""

    # Format all results into context with source categorization
    context_parts = []
    for category, items in search_results.items():
        is_social = any(tag in category for tag in ["social", "twitter", "fintwit", "reddit", "wsb"])
        is_alpha = any(tag in category for tag in ["alpha", "options", "insider", "short"])
        is_benzinga = "benzinga" in category
        if is_social:
            source_tag = " [SOCIAL/FINTWIT]"
        elif is_benzinga:
            source_tag = " [BENZINGA PRO]"
        elif is_alpha:
            source_tag = " [ALPHA/EDGE]"
        else:
            source_tag = " [MAINSTREAM]"
        context_parts.append(f"\n--- {category.upper()}{source_tag} ---")
        for item in items:
            entry = f"Title: {item.get('title', 'N/A')}"
            entry += f"\nURL: {item.get('url', '')}"
            entry += f"\nSnippet: {item.get('content', '')[:500]}"
            if item.get("published_date"):
                entry += f"\nPublished: {item['published_date']}"
            context_parts.append(entry)

    context = "\n".join(context_parts)

    response = openai_client.responses.create(
        model="gpt-5.2-pro",
        input=f"""=== TAVILY SEARCH RESULTS ===
{context}

You are a senior financial analyst who also monitors fintwit, WSB, unusual flow data, and Benzinga pro news.
Analyze these search results and produce a structured assessment.
Sources include Tavily web search, Benzinga professional financial news [BENZINGA PRO], social media [SOCIAL/FINTWIT], and alpha/edge [ALPHA/EDGE]. Benzinga articles are professional-grade Ã¢ÂÂ treat as high-credibility, no fact-checking needed.

CRITICAL FRESHNESS RULE: This briefing runs 3x daily. Stories must be FRESH - published within the last 8 hours ideally, 12 hours max. REJECT stale news.

CRITICAL SOURCE DIVERSITY RULE: You MUST include stories from [SOCIAL/FINTWIT] and [ALPHA/EDGE] categories, not just mainstream news. The whole point of this tool is to surface things readers WON'T see on CNBC. Also prioritize any Twitter/fintwit trending tickers or StockTwits buzz - this is high-value social signal.

CRITICAL DEDUPLICATION RULE: NO TWO STORIES can cover the same underlying data point or event. If retail sales data was released, that is ONE story - not "retail sales flat" + "bonds rally on retail data" + "consumer demand fragile." The bond reaction and demand narrative are PART OF the retail sales story, not separate stories. Aggressively merge related angles into a single richer story. The 10 final stories must cover 10 DIFFERENT topics.

CRITICAL TOPIC DIVERSITY: The 10 stories MUST span at least 6 of these categories:
- Macro data / economic releases
- Central banks / Fed / monetary policy
- Geopolitics / wars / military / sanctions
- Trade policy / tariffs / export controls (PRIORITY Ã¢ÂÂ always include if tariff or trade news exists)
- Earnings / individual company news
- Commodities / energy / crypto
- Unusual options / insider activity / flow data
- Social media buzz / fintwit / Twitter trending / WSB
If you find yourself with 3+ stories on the same macro release, you are doing it wrong. Geopolitics and trade policy are SEPARATE categories Ã¢ÂÂ a tariff story is not the same as a war story.

TASKS:
1. RANK: Identify 15-18 most IMPORTANT and MARKET-MOVING stories. Importance = stories a professional trader NEEDS to know today. If Bloomberg would not flash it as breaking news, skip it. For each, you MUST include the source_url from the search results. AGGRESSIVELY deduplicate - merge all angles of the same event into ONE story. Categories:
   - MAINSTREAM (8-10): Major market news from credible outlets
   - ALPHA (3-5): Unusual options, insider trades, short squeezes, flow data. MUST cite specific tickers, dollar amounts, or strike prices. MUST describe an EVENT that happened (a trade, a filing, a spike, a move) - NEVER a "how to" guide, tutorial, or advice on how to scan/use a tool. If it reads like instruction rather than news, it is NOT alpha.
   - SOCIAL BUZZ (1-2): Only include if genuinely viral or market-moving. Skip if stale or low-impact
2. DIG DEEPER: Pick 3-4 stories (prefer ALPHA/SOCIAL) that deserve deeper investigation.
3. FACT CHECK: Flag social media claims that make specific factual assertions. These MUST be verified.

Return ONLY valid JSON:
{{
  "top_stories": [
    {{"title": "...", "summary": "one sentence", "source_url": "the actual URL from search results", "impact": "high/medium/low", "category": "mainstream/alpha/social"}}
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

    # Deep dive searches
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

    # Fact-check social media claims against credible sources
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
def stage4_final_synthesis(openai_client, analysis, additional, history=None):
    """GPT-5.2 Pro writes the final market briefing from all gathered intelligence."""
    now = datetime.utcnow().strftime("%A, %B %d, %Y %H:%M UTC")

    # Compile all intelligence WITH source URLs
    parts = [f"Date: {now}\n"]

    parts.append("=== RANKED STORIES WITH SOURCES ===")
    for story in analysis.get("top_stories", []):
        cat_tag = f"[{story.get('category', 'mainstream').upper()}]"
        parts.append(
            f"- [{story.get('impact', 'medium').upper()}] {cat_tag} "
            f"{story.get('title', '')}: {story.get('summary', '')}"
            f"\n  Source: {story.get('source_url', 'N/A')}"
        )

    parts.append("\n=== DEEP DIVE FINDINGS ===")
    for key, data in additional.items():
        if not key.startswith("deep_"):
            continue
        parts.append(f"\nTopic: {data.get('topic', '')} (Why: {data.get('why', '')})")
        for r in data.get("results", []):
            parts.append(f"  - {r.get('title', '')}: {r.get('content', '')[:300]}")
            parts.append(f"    URL: {r.get('url', '')}")

    parts.append("\n=== SOCIAL MEDIA VERIFICATION RESULTS ===")
    has_verifications = False
    for key, data in additional.items():
        if not key.startswith("verify_"):
            continue
        has_verifications = True
        status = "CONFIRMED by credible sources" if data.get("verified") else "NOT CONFIRMED by major outlets"
        parts.append(f"- Claim: '{data.get('claim', '')}' -> {status}")
        for s in data.get("credible_sources", []):
            parts.append(f"  Confirming source: {s.get('title', '')} ({s.get('url', '')})")
    if not has_verifications:
        parts.append("No social media claims required verification.")

    context = "\n".join(parts)

    # Build history context for deduplication across runs
    history_text = ""
    if history and history.get("stories"):
        history_text = "\n\nPREVIOUSLY COVERED (headlines from most recent briefing):\n"
        for s_item in history["stories"]:
            history_text += f"- {s_item}\n"
        if history.get("tickers"):
            history_text += f"Tickers mentioned: {', '.join(history['tickers'])}\n"
        history_text += "\nDEDUPLICATION GUIDANCE (soft nudge, NOT a hard block):\n"
        history_text += "- If a story above is STILL among the day\u2019s most important narratives, KEEP IT but write a fresh angle with updated data or new development\n"
        history_text += "- Only SKIP stories that were marginal filler last time and have no new update\n"
        history_text += "- Major themes (Fed, tariffs, geopolitical crises) SHOULD recur when genuinely driving markets\n"
        history_text += "- The goal is fresh WRITING and angles, not exclusion of important topics\n"


    response = openai_client.responses.create(
        model="gpt-5.2-pro",
        input=f"""{context}

Write the Daily Market Pulse briefing based on ALL the intelligence above.

CRITICAL: This briefing is generated {now} and readers expect REAL-TIME freshness. Every story must reflect what is happening RIGHT NOW or within the last few hours.

CRITICAL SELECTION PRINCIPLE: You are curating the 10 MOST IMPORTANT stories of the day for professional traders and investors. Every story must pass this test: "Is this something a portfolio manager NEEDS to know before the next session?" If a story is routine corporate housekeeping (offerings, filings, prospectus supplements) or niche noise with no broad market impact, it does NOT belong. Prioritize by MARKET IMPACT and BROAD RELEVANCE over specificity. FRESHNESS MATTERS: if a story has been widely covered for days and has no new development, deprioritize it in favor of something newer.

STORY MIX REQUIREMENTS:
- Stories 1-7: Major market-moving mainstream news. MUST cover DIVERSE topics AND REGIONS: macro data, earnings, geopolitics/politics, central banks, commodities, FX, etc. Think globally — if Asia or Europe had significant moves, they MUST be represented. Do NOT fill all 7 slots with US news when important things happened in other regions. If a macro data release happened, it gets ONE story that includes the market reaction — not 3 separate stories about the data, the bond move, and the demand narrative.
- Stories 8-9: Alpha/edge stories with SPECIFIC actionable intelligence that is MARKET-MOVING and of BROAD interest. These MUST name specific tickers, dollar amounts, strike prices, dates, or insider names. They MUST describe something that IS HAPPENING or HAS HAPPENED Ã¢ÂÂ an event, a trade, a filing, a spike. NEVER write "how to use" a tool, "how to scan for" X, "monitor this dashboard," or any instructional content. NEVER use routine corporate filings as alpha stories. BAD examples: prospectus supplements, shelf registrations, convertible note offerings, equity offering program updates, SEC form filings. These are corporate housekeeping, NOT alpha intelligence. The reader wants unusual TRADES, FLOW, INSIDER MOVES, or MARKET-MOVING events - not paperwork.
BAD: "Use Fintel to monitor borrow fees." BAD: "Scan for unusual options activity." GOOD: "$TSLA saw $45M in call sweeps at $280 strike, Feb 14 expiry - 3x normal volume." GOOD: "CEO of XYZ bought $2.1M in shares on the open market, largest insider buy in 3 years." If the data doesn't have specific alpha events, pick the most newsworthy non-mainstream story available.
- Story 10: Wildcard \u2014 the single most important remaining story you haven\u2019t covered yet, regardless of category. Could be social buzz, additional alpha, a developing geopolitical angle, or anything else. Pick purely by importance and freshness. If it\u2019s from social media, label as \u2018(unverified social buzz).\u2019

ABSOLUTE RULE: Each of the 10 stories must cover a DIFFERENT topic. No two stories should be about the same data release, the same company, or the same market theme. If you catch yourself writing two stories about the same thing, MERGE them into one and find a new topic for the freed-up slot. Look for geopolitics, trade policy/tariffs (these are SEPARATE topics), sector-specific moves, individual earnings, crypto, commodities, politics - the world is big. ALWAYS include a trade policy/tariff story if any tariff or trade news exists in the data.

FORMAT:
1. Executive summary: 2-3 sentences capturing overall market mood and the single biggest theme RIGHT NOW.
2. 10 numbered stories using emoji numbers (1 through 10).
3. Each story: 2-3 punchy sentences with specific names, numbers, percentages.
4. MANDATORY: After each story, include the source link on a new line formatted as: Link: [url]
   This is CRITICAL - every story MUST have its source link. Use the Source URLs provided in the data above.
5. For stories 8-9, prefix with a lightning bolt emoji to signal alpha content.
6. For story 10, prefix with a star emoji to signal wildcard/best remaining story.
7. For verified social claims, note "confirmed by [source]". For unverified, label "(unverified social buzz)".
8. End with one "Key Watch:" line for what to monitor next.

TONE: Like a sharp morning briefing from a senior analyst who also monitors fintwit and unusual flow.
Professional but engaging. Stories 8-9 should feel like insider intel you can't get from mainstream news. Story 10 should surprise the reader with the most important thing they might have missed.
Every sentence earns its place.\n{history_text} No filler, no padding. The reader should feel smarter after reading all 10 stories, each one teaching them something new about a DIFFERENT corner of the market."""
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
    full_message = f"Daily Market Pulse - {now}\n\n{message}"

    if len(full_message) <= max_len:
        resp = requests.post(url, json={"chat_id": CHAT_ID, "text": full_message, "disable_web_page_preview": True}, timeout=30)
        return resp.json()

    # Split on newlines
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
        result = requests.post(url, json={"chat_id": CHAT_ID, "text": chunk, "disable_web_page_preview": True}, timeout=30)
    return result.json() if result else {"ok": False}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Daily Market Pulse Bot v4 (Twitter + Geopolitics + Alpha Fix)")
    print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    # Validate environment
    missing = []
    for var in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "OPENAI_API_KEY", "TAVILY_API_KEY"]:
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        return False

    if not os.environ.get("BENZINGA_API_KEY"):
        print("Note: BENZINGA_API_KEY not set - Benzinga source will be skipped")

    openai_client, tavily_client = init_clients()

    # Stage 0 - GPT generates smart search queries (mainstream + alpha)
    print("\nStage 0: Generating targeted search queries...")
    mainstream_queries, alpha_queries = stage0_generate_queries(openai_client)
    print("  Mainstream queries:")
    for q in mainstream_queries:
        print(f"    -> {q}")
    print("  Alpha queries:")
    for q in alpha_queries:
        print(f"    -> {q}")

    # Stage 1 - Multi-angle Tavily searches (mainstream + social + alpha)
    print("\nStage 1: Running Tavily searches...")
    search_results = stage1_tavily_searches(tavily_client, mainstream_queries, alpha_queries)
    total = sum(len(v) for v in search_results.values())
    print(f"   Total results gathered: {total}")

    # Stage 2 - GPT first-pass analysis
    print("\nStage 2: GPT analyzing and ranking stories...")
    analysis = stage2_first_pass(openai_client, search_results)
    top = analysis.get("top_stories", [])
    cats = {}
    for s in top:
        c = s.get("category", "mainstream")
        cats[c] = cats.get(c, 0) + 1
    print(f"   Top stories: {len(top)} ({', '.join(f'{k}={v}' for k,v in cats.items())})")
    print(f"   Dig deeper targets: {len(analysis.get('dig_deeper', []))}")
    print(f"   Fact checks needed: {len(analysis.get('fact_check', []))}")

    # Stage 3 - Verification + deep dives
    print("\nStage 3: Deep dives and fact-checking...")
    additional = stage3_verify_and_deepen(tavily_client, analysis)

    # Load history of previously covered stories
    history = load_history()

    # Stage 4 \u2014 Final synthesis
    print("\nStage 4: Generating final briefing...")
    briefing = stage4_final_synthesis(openai_client, analysis, additional, history)

    # Save history for next run
    save_history(briefing)

    # Deliver
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
