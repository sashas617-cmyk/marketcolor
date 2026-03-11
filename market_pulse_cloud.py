#!/usr/bin/env python3
"""
Daily Market Pulse Bot - Enhanced Version v4.3 (Venice AI)
Uses Tavily for deep multi-angle search + Benzinga for professional financial news
+ Venice AI GPT-5.4 for reasoning/synthesis.

v4.1 changes:
- Migrated from OpenAI GPT-5.2 Pro to Venice AI GPT-5.4
- Uses Venice AI OpenAI-compatible API endpoint
- Cost savings: Venice AI pricing vs OpenAI direct

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
import time
from datetime import datetime
from openai import OpenAI
from pathlib import Path

# Configuration
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
VENICE_API_KEY = os.environ.get("VENICE_API_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
BENZINGA_API_KEY = os.environ.get("BENZINGA_API_KEY", "")
HISTORY_FILE = Path(__file__).parent / "history.json"

# Venice AI model
VENICE_MODEL = "openai-gpt-52"
VENICE_FALLBACK_MODEL = "grok-41-fast"
VENICE_BASE_URL = "https://api.venice.ai/api/v1"


def init_clients():
    """Initialize API clients."""
    openai_client = OpenAI(
        api_key=VENICE_API_KEY,
        base_url=VENICE_BASE_URL,
    )
    return openai_client


def _venice_chat(client, prompt, max_tokens=4096, retries=3):
    """Helper: call Venice AI chat completions and return text content.
    v4.5: Switch to stable GPT-5.2 with Grok 4.1 Fast fallback.
    """
    models_to_try = [VENICE_MODEL, VENICE_FALLBACK_MODEL]
    for model in models_to_try:
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    extra_body={
                        "venice_parameters": {
                            "include_venice_system_prompt": False,
                            "strip_thinking_response": True,
                        }
                    },
                )
                result = response.choices[0].message.content
                if result and result.strip():
                    if model != VENICE_MODEL:
                        print(f"  INFO: Got response from fallback model {model}")
                    return result
                print(f"  WARNING: Empty response from {model} (attempt {attempt+1}/{retries}). finish_reason={response.choices[0].finish_reason}")
            except Exception as e:
                print(f"  ERROR: {model} API call failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
        print(f"  WARNING: All {retries} attempts failed for {model}, trying next model...")
    print("  CRITICAL: All models failed to return content")
    return ""
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
    alpha_tickers = []
    in_alpha_section = False

    for line in briefing_text.split("\n"):
        if "**" in line and line.count("**") >= 2:
            start = line.index("**") + 2
            end = line.index("**", start)
            headline = line[start:end].strip()
            if len(headline) > 15 and ":" in headline:
                stories.append(headline)

        # Detect alpha stories (8-9) marked with lightning bolt
        if "\u26a1" in line or "Alpha" in line or "alpha" in line:
            in_alpha_section = True

        # Extract tickers from alpha section
        if in_alpha_section:
            for word in line.split():
                clean = word.lstrip("$").rstrip(".,;:!()")
                if word.startswith("$") and clean.isalpha() and 1 <= len(clean) <= 5:
                    if clean.upper() not in alpha_tickers:
                        alpha_tickers.append(clean.upper())

        # Reset on story divider
        if in_alpha_section and line.strip() == "---":
            in_alpha_section = False

    tickers = []
    for word in briefing_text.split():
        clean = word.lstrip("$").rstrip(".,;:!()")
        if word.startswith("$") and clean.isalpha() and 1 <= len(clean) <= 5:
            if clean.upper() not in tickers:
                tickers.append(clean.upper())
    tickers.sort()
    alpha_tickers.sort()

    data = {
        "last_run": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stories": stories,
        "tickers": tickers,
        "alpha_tickers": alpha_tickers,
    }

    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved history: {len(stories)} stories, {len(tickers)} tickers, {len(alpha_tickers)} alpha tickers")



# ---------------------------------------------------------------------------
# BRAVE SEARCH: Web search via Brave Search API
# ---------------------------------------------------------------------------

def brave_search(query, count=5, freshness="pd"):
    """Search using Brave Search API. Returns list of {title, url, content, published_date}."""
    if not BRAVE_API_KEY:
        print(f"  brave: SKIPPED (no API key)")
        return []
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "count": min(count, 20),
        "text_decorations": False,
        "search_lang": "en",
    }
    if freshness:
        params["freshness"] = freshness
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"  brave: API error {resp.status_code}")
            return []
        data = resp.json()
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("description", ""),
                "published_date": item.get("page_age", item.get("age", "")),
            })
        return results
    except Exception as e:
        print(f"  brave: ERROR - {e}")
        return []

# ---------------------------------------------------------------------------
# STAGE 0: GPT reasons about what to search for right now
# ---------------------------------------------------------------------------
def stage0_generate_queries(openai_client):
    """Venice AI GPT-5.4 generates targeted search queries based on current context."""
    now = datetime.utcnow()
    today = now.strftime("%A, %B %d, %Y %H:%M UTC")
    hour = now.hour

    prompt = f"""Today is {today}. Current UTC hour: {hour}.

You are a global financial markets research director. Your job: figure out the 12 most important things happening RIGHT NOW that could affect financial markets, politics, and geopolitics.

RECENCY IS CRITICAL. At least 4-6 of your 12 queries MUST target events from the LAST 2-4 HOURS. Think: what just happened? What broke in the last session? What moved in the last few hours? Use phrases like "breaking", "just now", "last hour", "today session" in those queries.

The remaining queries can cover important ongoing themes from earlier today.

Think globally. What happened overnight in Asia? What is Europe doing? Any US pre-market movers? Earnings surprises? Central bank signals? Geopolitical developments? Commodity moves? Unusual flows or positioning? Emerging market events? Trade policy shifts?

DO NOT default to generic US-centric queries like "Federal Reserve" or "CPI" unless those are genuinely the biggest story right now. Think about what a sophisticated global macro trader sitting in Dubai at this hour would need to know.

Generate exactly 12 search queries designed to surface the freshest, most market-relevant news across ALL regions and asset classes. Each query should be specific and timely. Include "today" or "latest" or "breaking" or "March 2026" in queries to bias toward fresh results.

Return ONLY a JSON object. No explanation.

Example:
{{"queries": ["Japan yen surge Nikkei record today", "China CPI PPI inflation data latest", "Ukraine Russia overnight attack energy infrastructure", "US pre-market movers earnings surprises today", "oil price IEA demand forecast latest", "Middle East Iran negotiations latest", "European markets DAX economic data today", "semiconductor earnings AI infrastructure today", "central bank rate decision emerging markets latest", "unusual options activity large block trades today", "insider buying SEC Form 4 filings notable today", "trade tariffs policy announcement latest today"]}}"""

    text = _venice_chat(openai_client, prompt, max_tokens=2048)

    try:
        data = json.loads(text)
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


def fetch_benzinga_news(limit=100):
    """Fetch latest news from Benzinga API with retry logic."""
    if not BENZINGA_API_KEY:
        print("  benzinga: SKIPPED (no API key)")
        return []

    url = "https://api.massive.com/benzinga/v2/news"
    params = {
        "apiKey": BENZINGA_API_KEY,
        "limit": limit,
        "sort": "published.desc",
    }

    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=30)
            if resp.status_code == 200:
                break
            print(f"  benzinga: HTTP {resp.status_code} (attempt {attempt+1}/3)")
        except Exception as e:
            print(f"  benzinga: timeout/error (attempt {attempt+1}/3) - {e}")
        if attempt < 2:
            time.sleep(3)

    if resp is None or resp.status_code != 200:
        print("  benzinga: FAILED after 3 attempts")
        return []

    try:
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
        print(f"  benzinga: ERROR parsing response - {e}")
        return []

# ---------------------------------------------------------------------------
# STAGE 1: Parallel Tavily searches across multiple angles
# ---------------------------------------------------------------------------
def stage1_brave_searches(mainstream_queries, alpha_queries):
    """Run targeted Brave searches across multiple angles including social/alpha."""
    all_results = {}

    # --- SAFETY-NET SEARCHES (broad) ---
    mainstream_searches = [
        {"name": "breaking_now", "query": "breaking financial market news stock market moves latest today", "count": 7, "freshness": "pd"},
        {"name": "global_markets", "query": "most important global financial market news today Asia Europe US", "count": 5, "freshness": "pd"},
        {"name": "geopolitics_macro", "query": "geopolitics trade policy central banks commodities major developments today", "count": 5, "freshness": "pd"},
    ]

    # --- SOCIAL ---
    social_searches = [
        {"name": "fintwit_movers", "query": "stock market unusual move breaking catalyst today fintwit", "count": 5, "freshness": "pd"},
    ]

    # --- ALPHA ---
    alpha_searches = [
        {"name": "unusual_options", "query": "unusual options activity large block trade sweep today", "count": 5, "freshness": "pd"},
        {"name": "insider_trades", "query": "SEC insider buying selling notable Form 4 filing today", "count": 5, "freshness": "pw"},
    ]

    def run_search(config):
        name = config["name"]
        query = config["query"]
        count = config.get("count", 5)
        freshness = config.get("freshness", "pd")
        results = brave_search(query, count=count, freshness=freshness)
        print(f"    {name}: {len(results)} results")
        return name, results

    print("  [Mainstream]")
    for sc in mainstream_searches:
        name, results = run_search(sc)
        all_results[name] = results

    print("  [Social/Fintwit]")
    for sc in social_searches:
        name, results = run_search(sc)
        all_results[name] = results

    print("  [Alpha/Edge]")
    for sc in alpha_searches:
        name, results = run_search(sc)
        all_results[name] = results

    # Run GPT-generated mainstream queries
    print("  [GPT Mainstream Queries]")
    for i, query in enumerate(mainstream_queries):
        results = brave_search(query, count=5, freshness="pd")
        name = f"gpt_mainstream_{i}"
        print(f"    {name}: {len(results)} results")
        all_results[name] = results

    # Run GPT-generated alpha queries
    print("  [GPT Alpha Queries]")
    for i, query in enumerate(alpha_queries):
        results = brave_search(query, count=5, freshness="pd")
        name = f"gpt_alpha_{i}"
        print(f"    {name}: {len(results)} results")
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
    """Venice AI GPT-5.4 analyzes all Tavily results and produces structured analysis."""

    # Format all results into context with source categorization
    # v4.3: Cap context to ~60K chars to stay within model context window.
    # 288 results at 500 chars each = 144K which overwhelms the model (returns empty).
    MAX_CONTEXT_CHARS = 25000
    MAX_SNIPPET_CHARS = 300  # shortened from 500

    context_parts = []
    total_chars = 0
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

        header = f"\n--- {category.upper()}{source_tag} ---"
        context_parts.append(header)
        total_chars += len(header)

        # For Benzinga, limit to top 40 (most recent) to avoid flooding
        max_items = 40 if is_benzinga else len(items)
        for item in items[:max_items]:
            entry = f"Title: {item.get('title', 'N/A')}"
            entry += f"\nURL: {item.get('url', '')}"
            entry += f"\nSnippet: {item.get('content', '')[:MAX_SNIPPET_CHARS]}"
            if item.get("published_date"):
                entry += f"\nPublished: {item['published_date']}"
            total_chars += len(entry)
            if total_chars > MAX_CONTEXT_CHARS:
                context_parts.append("... (remaining results truncated for context window)")
                break
            context_parts.append(entry)
        if total_chars > MAX_CONTEXT_CHARS:
            break
    context = "\n".join(context_parts)
    print(f"  Stage 2 context size: {len(context)} chars ({total_chars} raw)")

    prompt = f"""=== TAVILY SEARCH RESULTS ===

{context}

You are a senior financial analyst who also monitors fintwit, WSB, unusual flow data, and Benzinga pro news. Analyze these search results and produce a structured assessment.

Sources include Tavily web search, Benzinga professional financial news [BENZINGA PRO], social media [SOCIAL/FINTWIT], and alpha/edge [ALPHA/EDGE]. Benzinga articles are professional-grade — treat as high-credibility, no fact-checking needed.

CRITICAL FRESHNESS RULE: This briefing runs 3x daily. Stories must be FRESH - published within the last 8 hours ideally, 12 hours max. REJECT stale news.

CRITICAL SOURCE DIVERSITY RULE: You MUST include stories from [SOCIAL/FINTWIT] and [ALPHA/EDGE] categories, not just mainstream news. The whole point of this tool is to surface things readers WON'T see on CNBC. Also prioritize any Twitter/fintwit trending tickers or StockTwits buzz - this is high-value social signal.

CRITICAL DEDUPLICATION RULE: NO TWO STORIES can cover the same underlying data point or event. If retail sales data was released, that is ONE story - not "retail sales flat" + "bonds rally on retail data" + "consumer demand fragile." The bond reaction and demand narrative are PART OF the retail sales story, not separate stories. Aggressively merge related angles into a single richer story. The 10 final stories must cover 10 DIFFERENT topics.

CRITICAL TOPIC DIVERSITY: The 10 stories MUST span at least 6 of these categories:
- Macro data / economic releases
- Central banks / Fed / monetary policy
- Geopolitics / wars / military / sanctions
- Trade policy / tariffs / export controls (PRIORITY — always include if tariff or trade news exists)
- Earnings / individual company news
- Commodities / energy / crypto
- Unusual options / insider activity / flow data
- Social media buzz / fintwit / Twitter trending / WSB

If you find yourself with 3+ stories on the same macro release, you are doing it wrong. Geopolitics and trade policy are SEPARATE categories — a tariff story is not the same as a war story.

TASKS:

1. RANK: Identify 15-18 most IMPORTANT and MARKET-MOVING stories. Importance = stories a professional trader NEEDS to know today. If Bloomberg would not flash it as breaking news, skip it. For each, you MUST include the source_url from the search results.

   AGGRESSIVELY deduplicate - merge all angles of the same event into ONE story.

   Categories:
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

    text = _venice_chat(openai_client, prompt, max_tokens=12288)

    # v4.3: If response is empty, retry once with a shorter context
    if not text or not text.strip():
        print(f"  WARNING: Empty response from Venice AI ({len(context)} char context). Retrying with shorter context...")
        # Retry with halved context
        short_context = context[:MAX_CONTEXT_CHARS // 2]
        short_prompt = prompt.replace(context, short_context)
        text = _venice_chat(openai_client, short_prompt, max_tokens=8192)
        if not text or not text.strip():
            print(f"  WARNING: Still empty after retry. Response length: {len(text or '')} chars")
            return {"top_stories": [], "dig_deeper": [], "fact_check": []}

    # v4.2: Robust JSON extraction — handle thinking tags, markdown, etc.
    cleaned = text

    # Strip <think>...</think> reasoning tags (in case strip_thinking_response didn't work)
    think_end = cleaned.find("</think>")
    if think_end >= 0:
        cleaned = cleaned[think_end + 8:].strip()

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts[1:]:  # skip text before first fence
            inner = part.strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            if inner.startswith("{"):
                cleaned = inner
                break

    try:
        return json.loads(cleaned)
    except Exception as e1:
        # Try extracting the outermost JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except Exception:
                pass

        # v4.6.2: Regex fallback — extract individual story objects from malformed JSON
        # This handles truncated responses, trailing commas, unescaped chars, etc.
        print(f"  WARNING: Could not parse Stage 2 JSON (error: {e1}). Trying regex fallback...")
        print(f"  Response length: {len(text)} chars")
        stories = []
        # Match objects that have at least title and source_url
        pattern = r'\{[^{}]*"title"\s*:\s*"([^"]*)"[^{}]*"source_url"\s*:\s*"([^"]*)"[^{}]*\}'
        for m in re.finditer(pattern, cleaned):
            full_match = m.group(0)
            try:
                obj = json.loads(full_match)
                stories.append(obj)
            except Exception:
                # Build object manually from the match
                title = m.group(1)
                url = m.group(2)
                # Try to extract summary too
                summary_m = re.search(r'"summary"\s*:\s*"([^"]*)"', full_match)
                summary = summary_m.group(1) if summary_m else title
                stories.append({
                    "title": title,
                    "summary": summary,
                    "source_url": url,
                    "impact": "medium",
                    "category": "mainstream"
                })
        if stories:
            print(f"  Regex fallback recovered {len(stories)} stories")
            return {"top_stories": stories, "dig_deeper": [], "fact_check": []}

        # Last resort: try fixing common JSON issues
        try:
            # Remove trailing commas before ] or }
            fixed = re.sub(r',\s*([\]\}])', r'\1', cleaned[start:end] if start >= 0 else cleaned)
            # Try to close truncated JSON
            open_braces = fixed.count('{') - fixed.count('}')
            open_brackets = fixed.count('[') - fixed.count(']')
            fixed += ']' * max(0, open_brackets) + '}' * max(0, open_braces)
            result = json.loads(fixed)
            print(f"  JSON repair recovered {len(result.get('top_stories', []))} stories")
            return result
        except Exception:
            pass

        preview = text[:500] if len(text) > 500 else text
        print(f"  CRITICAL: All JSON parsing methods failed. Response preview:\n{preview}")
        return {"top_stories": [], "dig_deeper": [], "fact_check": []}


# ---------------------------------------------------------------------------
# STAGE 3: Tavily verification + deep dives
# ---------------------------------------------------------------------------
CREDIBLE_DOMAINS = [
    "reuters.com", "bloomberg.com", "cnbc.com", "ft.com",
    "wsj.com", "apnews.com", "bbc.com", "marketwatch.com",
    "finance.yahoo.com",
]


def stage3_verify_and_deepen(analysis):
    """Run verification for social media claims and deep dives on key stories using Brave."""
    additional = {}

    # Deep dive searches
    for i, item in enumerate(analysis.get("dig_deeper", [])[:4]):
        query = item.get("search_query", "")
        if not query:
            continue
        try:
            results = brave_search(query, count=4, freshness="pw")
            additional[f"deep_{i}"] = {
                "topic": item.get("topic", ""),
                "why": item.get("why", ""),
                "results": results,
            }
            print(f"    Deep dive '{item.get('topic', '')[:40]}': {len(results)} results")
        except Exception as e:
            print(f"    Deep dive {i} error: {e}")

    # Fact-check social media claims against credible sources
    for i, item in enumerate(analysis.get("fact_check", [])[:3]):
        query = item.get("verify_query", "")
        if not query:
            continue
        try:
            # Add credible source names to query for better results
            credible_query = f"{query} site:reuters.com OR site:bloomberg.com OR site:cnbc.com OR site:ft.com OR site:wsj.com"
            sources = brave_search(credible_query, count=3, freshness="pw")
            verified = len(sources) > 0
            additional[f"verify_{i}"] = {
                "claim": item.get("claim", ""),
                "original_source": item.get("original_source", ""),
                "verified": verified,
                "credible_sources": sources,
            }
            tag = "CONFIRMED" if verified else "UNVERIFIED"
            print(f"    Fact check: {tag} - '{item.get('claim', '')[:50]}'")
        except Exception as e:
            print(f"    Fact check {i} error: {e}")

    return additional
# ---------------------------------------------------------------------------
# STAGE 4: GPT final synthesis into polished briefing
# ---------------------------------------------------------------------------
def stage4_final_synthesis(openai_client, analysis, additional, history=None):
    """Venice AI GPT-5.4 writes the final market briefing from all gathered intelligence."""
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

        # Hard block on alpha ticker repeats
        if history.get("alpha_tickers"):
            history_text += "\nALPHA TICKER HARD BLOCK (MANDATORY):\n"
            history_text += f"These tickers were used in alpha stories (8-9) last run: {', '.join(history['alpha_tickers'])}\n"
            history_text += "You MUST NOT use ANY of these tickers for stories 8-9 in this run. Find COMPLETELY DIFFERENT alpha.\n"
            history_text += "This is a HARD RULE. If the only unusual options data is on a blocked ticker, pick a different alpha story entirely (insider buy, short squeeze, block trade on a DIFFERENT name).\n"

        history_text += "\nDEDUPLICATION GUIDANCE (mainstream stories 1-7):\n"
        history_text += "- Major themes (Fed, tariffs, geopolitical crises) SHOULD recur when genuinely driving markets \u2014 write a fresh angle with updated data\n"
        history_text += "- Skip stories that were marginal filler last time and have no new update\n"
        history_text += "- The goal is fresh WRITING and angles, not exclusion of important macro topics\n"

    prompt = f"""{context}

Write the Daily Market Pulse briefing based on ALL the intelligence above.

CRITICAL: This briefing is generated {now} and readers expect REAL-TIME freshness. Every story must reflect what is happening RIGHT NOW or within the last few hours.

CRITICAL SELECTION PRINCIPLE: You are curating the 10 MOST IMPORTANT stories of the day for professional traders and investors. Every story must pass this test: "Is this something a portfolio manager NEEDS to know before the next session?" If a story is routine corporate housekeeping (offerings, filings, prospectus supplements) or niche noise with no broad market impact, it does NOT belong. Prioritize by MARKET IMPACT and BROAD RELEVANCE over specificity. FRESHNESS MATTERS: if a story has been widely covered for days and has no new development, deprioritize it in favor of something newer.

RECENCY RULE (CRITICAL): At least 3-5 of the 10 stories MUST be about events from the LAST 2-4 HOURS. If the S&P just dropped 2%, or earnings just printed, or a central bank just spoke, those stories come FIRST. Do NOT fill the briefing with stories from 12+ hours ago when fresh developments exist. The reader needs to know what JUST happened, not a recap of yesterday. Older stories are fine for context, but recency wins when choosing between two stories of similar importance.

STORY MIX REQUIREMENTS:
- Stories 1-7: Major market-moving mainstream news. MUST cover DIVERSE topics AND REGIONS: macro data, earnings, geopolitics/politics, central banks, commodities, FX, etc. Think globally. If a macro data release happened, it gets ONE story that includes the market reaction.
- Stories 8-9: Alpha/edge stories with SPECIFIC actionable intelligence that is MARKET-MOVING and of BROAD interest. These MUST name specific tickers, dollar amounts, strike prices, dates, or insider names. They MUST describe something that IS HAPPENING or HAS HAPPENED. NEVER write "how to use" a tool or any instructional content. NEVER use routine corporate filings as alpha stories.
  BAD: "Use Fintel to monitor borrow fees." BAD: "Scan for unusual options activity."
  GOOD: "$TSLA saw $45M in call sweeps at $280 strike, Feb 14 expiry - 3x normal volume."
  GOOD: "CEO of XYZ bought $2.1M in shares on the open market, largest insider buy in 3 years."
  If the data doesn't have specific alpha events, pick the most newsworthy non-mainstream story available.
- Story 10: Wildcard \u2014 the single most important remaining story you haven't covered yet, regardless of category.

ABSOLUTE RULE: Each of the 10 stories must cover a DIFFERENT topic. No two stories should be about the same data release, the same company, or the same market theme.

FORMAT:
1. Executive summary: 2-3 sentences capturing overall market mood and the single biggest theme RIGHT NOW.
2. 10 numbered stories using emoji numbers (1 through 10).
3. Each story: 2-3 punchy sentences with specific names, numbers, percentages.
4. MANDATORY: After each story, include the source link on a new line formatted as:
   Link: [url]
5. For stories 8-9, prefix with a lightning bolt emoji to signal alpha content.
6. For story 10, prefix with a star emoji to signal wildcard/best remaining story.
7. For verified social claims, note "confirmed by [source]". For unverified, label "(unverified social buzz)".
8. End with one "Key Watch:" line for what to monitor next.

TONE: Like a sharp morning briefing from a senior analyst who also monitors fintwit and unusual flow. Professional but engaging.
{history_text}
No filler, no padding. The reader should feel smarter after reading all 10 stories, each one teaching them something new about a DIFFERENT corner of the market."""

    return _venice_chat(openai_client, prompt, max_tokens=4096)


# ---------------------------------------------------------------------------
# Telegram delivery
# ---------------------------------------------------------------------------
def send_telegram_message(message):
    """Send message to Telegram, splitting at newlines if over 4000 chars. v4.6.3: full error logging."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    max_len = 4000
    now = datetime.utcnow().strftime("%B %d, %Y %H:%M UTC")
    full_message = f"Daily Market Pulse \u2013 {now}\n\n{message}"
    print(f"  Telegram message length: {len(full_message)} chars")

    def _send_chunk(text, label=""):
        """Send a single chunk and return (ok, response_dict)."""
        try:
            resp = requests.post(url, json={"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": True}, timeout=30)
            data = resp.json()
            if not data.get("ok"):
                print(f"  Telegram API error{label}: {data.get('error_code')} - {data.get('description')}")
            return data.get("ok", False), data
        except Exception as e:
            print(f"  Telegram send exception{label}: {e}")
            return False, {"ok": False, "description": str(e)}

    if len(full_message) <= max_len:
        ok, data = _send_chunk(full_message)
        return data

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

    print(f"  Message split into {len(chunks)} chunks: {[len(c) for c in chunks]}")
    all_ok = True
    last_data = {"ok": False}
    for i, chunk in enumerate(chunks):
        ok, data = _send_chunk(chunk, f" [chunk {i+1}/{len(chunks)}]")
        if ok:
            print(f"  Chunk {i+1}/{len(chunks)} sent ({len(chunk)} chars)")
        else:
            all_ok = False
        last_data = data
        time.sleep(0.5)  # Rate limit buffer between chunks

    if all_ok:
        last_data["ok"] = True
    return last_data
# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Daily Market Pulse Bot v4.6.3 (Brave Search + Benzinga + Venice AI)'%Y-%m-%d %H:%M:%S UTC')")
    print("=" * 60)

    # Validate environment
    missing = []
    for var in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "VENICE_API_KEY", "BRAVE_API_KEY"]:
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}")
        return False

    if not os.environ.get("BENZINGA_API_KEY"):
        print("Note: BENZINGA_API_KEY not set - Benzinga source will be skipped")

    openai_client = init_clients()

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
    print("\nStage 1: Running Brave + Benzinga searches...")
    search_results = stage1_brave_searches(mainstream_queries, alpha_queries)
    total = sum(len(v) for v in search_results.values())
    print(f"  Total results gathered: {total}")

    # Stage 2 - GPT first-pass analysis
    print("\nStage 2: GPT analyzing and ranking stories...")
    analysis = stage2_first_pass(openai_client, search_results)
    top = analysis.get("top_stories", [])
    cats = {}
    for s in top:
        c = s.get("category", "mainstream")
        cats[c] = cats.get(c, 0) + 1
    print(f"  Top stories: {len(top)} ({', '.join(f'{k}={v}' for k,v in cats.items())})")
    print(f"  Dig deeper targets: {len(analysis.get('dig_deeper', []))}")
    print(f"  Fact checks needed: {len(analysis.get('fact_check', []))}")

    # Stage 3 - Verification + deep dives
    print("\nStage 3: Deep dives and fact-checking...")
    additional = stage3_verify_and_deepen(analysis)

    # Load history of previously covered stories
    history = load_history()

    # Stage 4 — Final synthesis
    print("\nStage 4: Generating final briefing...")
    briefing = stage4_final_synthesis(openai_client, analysis, additional, history)

    # Save history for next run
    save_history(briefing)

    # Deliver
    print(f"\nSending to Telegram... (briefing length: {len(briefing)} chars)")
    print(f"  CHAT_ID: {CHAT_ID}")
    print(f"  Briefing preview: {briefing[:200]}...")
    result = send_telegram_message(briefing)
    if result.get("ok"):
        print("Message sent successfully!")
    else:
        print(f"  Telegram send FAILED: {result}")
        print("  NOTE: Pipeline completed but Telegram delivery failed.")
        print("  Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
    # Always return True if we got this far — the analysis/briefing succeeded
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
