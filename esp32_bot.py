"""
ESP32-C3 Super Mini — MicroPython Telegram Bot
================================================
Multi-provider AI chat bot (Groq, Gemini, OpenRouter, Cerebras, NVIDIA)
with DuckDNS updater and WiFi retry loop.

Hardware: ESP32-C3 Super Mini (400 KB SRAM, 4 MB flash)
Runtime:  MicroPython 1.20+

ZERO DISK WRITES — all state is in-memory, reboot resets everything.
"""

import network
import urequests
import ujson
import time
import gc
from machine import WDT, reset, freq

# ============================================================================
# CONFIGURATION — Edit these values before flashing
# ============================================================================

# WiFi
WIFI_SSID = "YOUR_WIFI_SSID"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"

# Telegram
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
ALLOWED_USER_IDS = []  # e.g. [123456789, 987654321] — empty = allow all

# DuckDNS (set to "" to disable)
DUCKDNS_TOKEN = "YOUR_DUCKDNS_TOKEN"
DUCKDNS_DOMAIN = "YOUR_SUBDOMAIN"  # just the subdomain, not full URL
DUCKDNS_INTERVAL = 300  # seconds between updates (5 min)

# AI Provider API Keys (set to "" to disable a provider)
GROQ_API_KEY = ""
GEMINI_API_KEY = ""
OPENROUTER_API_KEY = ""
CEREBRAS_API_KEY = ""
NVIDIA_API_KEY = ""

# Brave Search API (set to "" to use DuckDuckGo instead — free, no key needed)
BRAVE_API_KEY = ""
MAX_SEARCH_RESULTS = 3  # Keep low for RAM
MAX_SNIPPET_LEN = 200  # Truncate each snippet to save RAM
SEARCH_ENGINE = "brave"  # "brave" or "duckduckgo" — brave needs API key, ddg is free

# Bot behavior
DEFAULT_PROVIDER = "groq"
MAX_TOKENS = 512
TEMPERATURE = 0.7
MAX_HISTORY = 5  # Keep low for RAM — 5 messages ~2KB
MAX_SESSIONS = 3  # Limit concurrent user sessions to save RAM
MAX_RESPONSE_LEN = 4000  # Truncate AI responses beyond this to protect RAM
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise. No markdown tables."

# ============================================================================
# CLIENT-SIDE SEARCH HEURISTIC — decide BEFORE calling the AI
# Saves ~300 bytes of prompt tokens + an entire API round-trip
# when search is clearly not needed (greetings, coding, math, etc.)
# ============================================================================

# Time-sensitive words → almost always need fresh data
_KW_TIME = (
    'today', 'tonight', 'now', 'latest', 'current', 'recent',
    'yesterday', 'tomorrow', 'this week', 'this month', 'this year',
    'last week', 'last month', 'last year', 'next week', 'next month',
)
# Topic words → usually need real-time data
_KW_TOPIC = (
    'price', 'stock', 'crypto', 'bitcoin', 'btc', 'eth', 'market', 'trading',
    'weather', 'forecast', 'temperature',
    'news', 'headline', 'election', 'vote', 'poll',
    'score', 'game', 'match', 'tournament', 'playoff', 'standings',
    'release', 'released', 'launched', 'announced', 'update',
    'dead', 'died', 'alive', 'arrested', 'fired', 'hired', 'resigned',
    'war', 'earthquake', 'hurricane', 'disaster', 'outbreak',
)
# Action words → coding/creative/math tasks that NEVER need search
_KW_NO_SEARCH = (
    'write', 'code', 'create', 'generate', 'build', 'make', 'implement',
    'explain', 'define', 'describe', 'summarize', 'rewrite', 'translate',
    'fix', 'debug', 'refactor', 'optimize', 'calculate', 'solve', 'convert',
    'hello', 'hi ', 'hey ', 'thanks', 'thank you', 'bye', 'good morning',
    'tell me a joke', 'poem', 'story', 'essay', 'list',
)


def _wants_search(text):
    """Lightweight client-side heuristic: should this message get search capability?

    Returns one of:
      'direct'  — high confidence search is needed, skip AI ask, search immediately
      'maybe'   — inject search prompt, let AI decide
      'no'      — definitely no search needed, save tokens + RAM

    This runs BEFORE any API call, saving an entire round-trip for non-search messages
    and optionally saving TWO round-trips when we can search directly."""
    low = text.lower()
    length = len(low)

    # Very short messages (< 12 chars) are greetings/reactions — never search
    if length < 12:
        return 'no'

    # Coding / creative / math tasks — flag but don't return yet
    # (search-need signals take priority when both are present)
    no_hit = False
    for kw in _KW_NO_SEARCH:
        if kw in low:
            no_hit = True
            break

    # --- Check for strong search signals ---
    time_hit = False
    for kw in _KW_TIME:
        if kw in low:
            time_hit = True
            break

    topic_hit = False
    for kw in _KW_TOPIC:
        if kw in low:
            topic_hit = True
            break

    # Recent year mentioned (current year ± 1)
    t = time.localtime()
    year = t[0]
    year_hit = False
    for y in (str(year - 1), str(year), str(year + 1)):
        if y in low:
            year_hit = True
            break

    # Both time + topic → very high confidence, search directly
    if time_hit and topic_hit:
        return 'direct'

    # Any single strong signal → let AI decide (inject search prompt)
    if time_hit or topic_hit or year_hit:
        return 'maybe'

    # No search signals found → respect no-search keywords
    if no_hit:
        return 'no'

    # Questions with '?' that aren't about coding/explanation might need search
    if '?' in low and length > 20:
        # "who is/was/won" patterns
        if 'who ' in low:
            return 'maybe'
        # "how much" / "how many" often need current data
        if 'how much' in low or 'how many' in low:
            return 'maybe'
        # "where is" can be factual
        if 'where ' in low:
            return 'maybe'

    return 'no'


def _extract_search_query(text):
    """Extract a clean search query from the user message for direct search.
    Strips filler words and caps at 120 chars for a clean API query."""
    # Remove common question prefixes
    low = text.strip()
    for prefix in ('what is the', 'what are the', 'what is', 'what are',
                   'how much is', 'how much does', 'how much',
                   'who is the', 'who is', 'who won the', 'who won',
                   'where is the', 'where is',
                   'when is the', 'when is', 'when does',
                   'tell me about', 'search for', 'look up', 'find'):
        if low.lower().startswith(prefix):
            low = low[len(prefix):].strip()
            break
    # Remove trailing question mark
    low = low.rstrip('?').strip()
    return low[:120] if low else text[:120]


def get_search_prompt():
    """Build search prompt with today's date so the AI knows the current year.
    Kept concise — every byte costs RAM on ESP32."""
    t = time.localtime()
    today = "%04d-%02d-%02d" % (t[0], t[1], t[2])
    return (
        " Today is %s." % today
        + " You can search the web. To search, reply ONLY: SEARCH: <keywords>"
        " Keep queries short (2-5 words). Do not guess outdated facts — search instead."
    )

# ============================================================================
# PROVIDER DEFINITIONS
# Each provider has:
#   url        - chat completions endpoint
#   models_url - GET /models endpoint (for dynamic fetch)
#   can_fetch  - True if model list is small enough to fit in RAM
#   fallback   - hardcoded fallback model list
#   models     - current model list (starts as fallback, updated by fetch)
# ============================================================================

PROVIDERS = {}

if GROQ_API_KEY:
    PROVIDERS["groq"] = {
        "name": "Groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "models_url": "https://api.groq.com/openai/v1/models",
        "can_fetch": True,  # ~10-15 models, small JSON
        "key": GROQ_API_KEY,
        "default_model": "llama-3.3-70b-versatile",
        "fallback": [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "models": [],
    }

if GEMINI_API_KEY:
    PROVIDERS["gemini"] = {
        "name": "Gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "models_url": "https://generativelanguage.googleapis.com/v1beta/openai/models",
        "can_fetch": True,  # ~10-20 models, small JSON
        "key": GEMINI_API_KEY,
        "default_model": "gemini-2.0-flash",
        "fallback": [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        "models": [],
    }

if OPENROUTER_API_KEY:
    PROVIDERS["openrouter"] = {
        "name": "OpenRouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "models_url": "",
        "can_fetch": False,  # 200+ models, ~300KB JSON — too large for ESP32 RAM
        "key": OPENROUTER_API_KEY,
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "fallback": [
            "meta-llama/llama-3.3-70b-instruct:free",
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
        ],
        "models": [],
    }

if CEREBRAS_API_KEY:
    PROVIDERS["cerebras"] = {
        "name": "Cerebras",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "models_url": "https://api.cerebras.ai/v1/models",
        "can_fetch": True,  # ~5-10 models, small JSON
        "key": CEREBRAS_API_KEY,
        "default_model": "llama-3.3-70b",
        "fallback": [
            "gpt-oss-120b",
            "llama-3.3-70b",
            "llama3.1-8b",
        ],
        "models": [],
    }

if NVIDIA_API_KEY:
    PROVIDERS["nvidia"] = {
        "name": "NVIDIA",
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "models_url": "",
        "can_fetch": False,  # 183+ models — too large for ESP32 RAM
        "key": NVIDIA_API_KEY,
        "default_model": "openai/gpt-oss-120b",
        "fallback": [
            "openai/gpt-oss-120b",
            "deepseek-ai/deepseek-v3.2",
            "deepseek-ai/deepseek-v3.1-terminus",
            "deepseek-ai/deepseek-v3.1",
            "qwen/qwen3-235b-a22b",
            "qwen/qwen3.5-397b-a17b",
            "qwen/qwen3-coder-480b-a35b-instruct",
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-405b-instruct",
            "minimaxai/minimax-m2",
            "minimaxai/minimax-m2.1",
            "moonshotai/kimi-k2-instruct-0905",
            "moonshotai/kimi-k2.5",
            "z-ai/glm5",
            "z-ai/glm4.7",
        ],
        "models": [],
    }

# ============================================================================
# DYNAMIC MODEL FETCHING
# ============================================================================

def fetch_models(provider_key):
    """Fetch model list from provider's /models endpoint.
    Returns list of model ID strings or None on failure.
    Only works for providers with can_fetch=True."""
    prov = PROVIDERS.get(provider_key)
    if not prov or not prov["can_fetch"] or not prov["models_url"]:
        return None

    headers = {"Authorization": "Bearer " + prov["key"]}
    gc.collect()

    r = None
    try:
        r = urequests.get(prov["models_url"], headers=headers)
        data = r.json()
        r.close()
        r = None
        gc.collect()

        # OpenAI-compatible /models returns {"data": [{"id": "...", ...}, ...]}
        models_data = data.get("data", [])
        model_ids = []
        for m in models_data:
            mid = m.get("id", "")
            if mid:
                model_ids.append(mid)

        # Free the large parsed JSON immediately
        del data
        del models_data
        gc.collect()

        # Sort: models with larger param size first (future-proof ranking)
        model_ids.sort(key=lambda x: _model_sort_key(x))

        if model_ids:
            prov["models"] = model_ids
            print("[Models] %s: fetched %d models" % (prov["name"], len(model_ids)))
            return model_ids
        else:
            print("[Models] %s: fetch returned 0 models, using fallback" % prov["name"])
            return None

    except Exception as e:
        print("[Models] %s fetch error: %s" % (prov["name"], e))
        if r:
            try:
                r.close()
            except Exception:
                pass
        gc.collect()
        return None


def _model_sort_key(model_id):
    """Simple sort key: extract param size (e.g. '70b' -> -70) for ranking.
    Larger models sort first (negative). No-size models sort last."""
    mid = model_id.lower()
    # Find patterns like 70b, 405b, 8b, 1.5b
    i = 0
    while i < len(mid):
        if mid[i].isdigit():
            j = i
            while j < len(mid) and (mid[j].isdigit() or mid[j] == '.'):
                j += 1
            if j < len(mid) and mid[j] == 'b':
                try:
                    size = float(mid[i:j])
                    return -size  # negative so larger models sort first
                except Exception:
                    pass
            i = j
        else:
            i += 1
    return 0  # no size found, sort last among sized models


def refresh_all_models():
    """Fetch models for all providers that support it.
    For non-fetchable providers, use fallback list."""
    for key, prov in PROVIDERS.items():
        if wdt:
            wdt.feed()  # prevent watchdog during multi-provider fetch
        if prov["can_fetch"]:
            result = fetch_models(key)
            if result is None:
                prov["models"] = list(prov["fallback"])
        else:
            prov["models"] = list(prov["fallback"])
        gc.collect()


def get_models(provider_key):
    """Get current model list for a provider. Fetches if empty."""
    prov = PROVIDERS.get(provider_key)
    if not prov:
        return []
    if not prov["models"]:
        if prov["can_fetch"]:
            result = fetch_models(provider_key)
            if result is None:
                prov["models"] = list(prov["fallback"])
        else:
            prov["models"] = list(prov["fallback"])
    return prov["models"]

# ============================================================================
# GLOBAL STATE (in-memory only, never written to disk)
# ============================================================================

# Per-user sessions: { user_id: { "provider": str, "model": str, "history": [] } }
sessions = {}
boot_time = 0
tg_offset = 0
last_duckdns = 0
duckdns_status = "Disabled"  # Tracks DuckDNS state for /status
wdt = None  # Watchdog timer — initialized in main()

# ============================================================================
# WIFI — retry loop with exponential backoff
# ============================================================================

def wifi_connect():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    if wlan.isconnected():
        print("[WiFi] Already connected:", wlan.ifconfig()[0])
        return wlan

    print("[WiFi] Connecting to", WIFI_SSID)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    delay = 2  # start 2s, grow to 60s
    while not wlan.isconnected():
        if wdt:
            wdt.feed()  # prevent watchdog reboot during WiFi retry
        print("[WiFi] Waiting... retry in %ds" % delay)
        time.sleep(delay)
        delay = min(delay * 2, 60)
        # Re-trigger connect in case the AP wasn't up yet
        if not wlan.isconnected():
            try:
                wlan.connect(WIFI_SSID, WIFI_PASSWORD)
            except Exception:
                pass

    print("[WiFi] Connected!", wlan.ifconfig()[0])
    return wlan


def sync_ntp():
    """Sync ESP32 RTC with NTP so we know the real date/time."""
    import ntptime
    # Try multiple NTP servers in case one is blocked or slow
    for server in ("pool.ntp.org", "time.google.com"):
        try:
            ntptime.host = server
            ntptime.settime()  # sets RTC to UTC
            t = time.localtime()
            print("[NTP] Time synced via %s: %04d-%02d-%02d %02d:%02d:%02d UTC" % (
                server, t[0], t[1], t[2], t[3], t[4], t[5]))
            return
        except Exception as e:
            print("[NTP] %s failed: %s" % (server, e))
    print("[NTP] All servers failed (non-critical) — date in search prompt may be wrong")

# ============================================================================
# DUCKDNS UPDATER
# ============================================================================

def update_duckdns():
    global last_duckdns, duckdns_status
    if not DUCKDNS_TOKEN or not DUCKDNS_DOMAIN:
        return
    now = time.time()
    if now - last_duckdns < DUCKDNS_INTERVAL:
        return
    last_duckdns = now
    url = "https://www.duckdns.org/update?domains=%s&token=%s&verbose=true" % (
        DUCKDNS_DOMAIN, DUCKDNS_TOKEN
    )
    r = None
    try:
        r = urequests.get(url)
        result = r.text.strip()
        print("[DuckDNS]", result)
        r.close()
        r = None
        # DuckDNS returns "OK" on first line if successful
        if result.startswith("OK"):
            duckdns_status = "Running"
        else:
            duckdns_status = "Error: %s" % result.split("\n")[0]
    except Exception as e:
        print("[DuckDNS] Error:", e)
        duckdns_status = "Error: %s" % str(e)
        if r:
            try:
                r.close()
            except Exception:
                pass
    gc.collect()

# ============================================================================
# TELEGRAM API
# ============================================================================

TG_BASE = "https://api.telegram.org/bot" + TELEGRAM_TOKEN

def tg_get_updates(offset):
    """Long-poll for new messages (30s timeout)."""
    url = TG_BASE + "/getUpdates?timeout=30&offset=%d" % offset
    r = None
    try:
        r = urequests.get(url)
        data = r.json()
        r.close()
        r = None
        gc.collect()
        if data.get("ok"):
            result = data.get("result", [])
            del data
            return result
        del data
    except OSError as e:
        # Socket timeout / connection reset — common on ESP32
        print("[TG] network error:", e)
        if r:
            try:
                r.close()
            except Exception:
                pass
        gc.collect()
    except Exception as e:
        print("[TG] getUpdates error:", e)
        if r:
            try:
                r.close()
            except Exception:
                pass
        gc.collect()
    return []

def tg_send(chat_id, text):
    """Send message to Telegram, auto-split at 4096 chars."""
    MAX_LEN = 4096
    PREFIX_RESERVE = 10  # room for "[xx/xx]\n" prefix
    url = TG_BASE + "/sendMessage"

    # Split long messages (reserve space for part prefix)
    chunks = []
    split_len = MAX_LEN - PREFIX_RESERVE
    while len(text) > MAX_LEN:
        # Try to split at a newline
        split_at = text.rfind("\n", 0, split_len)
        if split_at < 1:
            split_at = split_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        chunks.append(text)

    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            chunk = "[%d/%d]\n%s" % (i + 1, len(chunks), chunk)
        payload = ujson.dumps({"chat_id": chat_id, "text": chunk})
        r = None
        try:
            r = urequests.post(url, data=payload,
                               headers={"Content-Type": "application/json"})
            r.close()
        except Exception as e:
            print("[TG] send error:", e)
            if r:
                try:
                    r.close()
                except Exception:
                    pass
        del payload
        gc.collect()

def tg_send_action(chat_id):
    """Send 'typing...' indicator."""
    url = TG_BASE + "/sendChatAction"
    payload = ujson.dumps({"chat_id": chat_id, "action": "typing"})
    try:
        r = urequests.post(url, data=payload,
                           headers={"Content-Type": "application/json"})
        r.close()
    except Exception:
        pass
    gc.collect()

# ============================================================================
# AI PROVIDER — unified OpenAI-compatible HTTP POST
# ============================================================================

def ai_chat(provider_key, model, messages, sys_prompt=None):
    """Send chat to any OpenAI-compatible provider. Returns response text."""
    prov = PROVIDERS.get(provider_key)
    if not prov:
        return "Error: provider '%s' not available." % provider_key

    # Build messages with system prompt
    prompt = sys_prompt if sys_prompt else SYSTEM_PROMPT
    chat_msgs = [{"role": "system", "content": prompt}]
    chat_msgs.extend(messages)

    body = {
        "model": model,
        "messages": chat_msgs,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + prov["key"],
    }

    payload = ujson.dumps(body)
    # Free body + chat_msgs immediately — payload string has the data now
    del body
    del chat_msgs
    gc.collect()

    r = None
    try:
        r = urequests.post(prov["url"], data=payload, headers=headers)
        del payload  # free request string
        data = r.json()
        r.close()
        r = None
        gc.collect()

        # Parse OpenAI-compatible response
        choices = data.get("choices")
        if choices and len(choices) > 0:
            msg = choices[0].get("message", {})
            content = msg.get("content", "(empty response)")
            del data
            # Truncate oversized responses to protect RAM
            if len(content) > MAX_RESPONSE_LEN:
                content = content[:MAX_RESPONSE_LEN] + "\n\n[truncated — response too long for ESP32]"
            return content

        # Check for error — API may return error as dict or string
        err = data.get("error")
        del data
        if err:
            if isinstance(err, dict):
                return "API Error: %s" % err.get("message", str(err))
            return "API Error: %s" % str(err)

        return "(no response from %s)" % prov["name"]

    except Exception as e:
        if r:
            try:
                r.close()
            except Exception:
                pass
        gc.collect()
        return "Error calling %s: %s" % (prov["name"], str(e))

# ============================================================================
# WEB SEARCH — Brave API + DuckDuckGo HTML scraping
# ============================================================================

def _url_encode(s):
    """Percent-encoding for query strings on MicroPython (UTF-8 safe)."""
    out = []
    for b in s.encode('utf-8'):
        if (65 <= b <= 90) or (97 <= b <= 122) or (48 <= b <= 57) or b in (45, 46, 95, 126):
            out.append(chr(b))  # A-Z a-z 0-9 - . _ ~
        elif b == 32:
            out.append('+')
        else:
            out.append('%%%02X' % b)
    return ''.join(out)


def _strip_tags(html_str):
    """Remove HTML tags and decode common entities. Lightweight for MicroPython."""
    # Strip tags
    out = []
    in_tag = False
    for c in html_str:
        if c == '<':
            in_tag = True
        elif c == '>':
            in_tag = False
        elif not in_tag:
            out.append(c)
    text = ''.join(out)
    # Decode common HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&#x27;', "'")
    text = text.replace('&nbsp;', ' ')
    return text.strip()


def brave_search(query):
    """Search the web via Brave Search API. Returns list of snippet strings."""
    if not BRAVE_API_KEY:
        return []

    q = _url_encode(query)
    url = "https://api.search.brave.com/res/v1/web/search?q=%s&count=%d" % (q, MAX_SEARCH_RESULTS)
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    gc.collect()

    r = None
    try:
        r = urequests.get(url, headers=headers)
        data = r.json()
        r.close()
        r = None
        gc.collect()

        # Extract snippets from web results
        web = data.get("web", {})
        results = web.get("results", [])
        snippets = []
        for item in results:
            title = item.get("title", "")
            desc = item.get("description", "")
            if desc and len(desc) > MAX_SNIPPET_LEN:
                desc = desc[:MAX_SNIPPET_LEN]
            if title or desc:
                snippets.append("%s: %s" % (title, desc))

        # Free parsed JSON immediately
        del data
        del web
        del results
        gc.collect()

        print("[Search] Brave '%s' -> %d results" % (query, len(snippets)))
        return snippets

    except Exception as e:
        print("[Search] Brave error: %s" % e)
        if r:
            try:
                r.close()
            except Exception:
                pass
        gc.collect()
        return []


def duckduckgo_search(query):
    """Search the web via DuckDuckGo HTML scraping. Free, no API key.
    Returns list of snippet strings (same format as brave_search)."""
    q = _url_encode(query)
    url = "https://html.duckduckgo.com/html/?q=%s" % q
    headers = {"User-Agent": "Mozilla/5.0"}
    gc.collect()

    r = None
    try:
        r = urequests.get(url, headers=headers)
        html = r.text
        r.close()
        r = None
        gc.collect()

        # Parse HTML results using str.find() — lightweight for MicroPython
        snippets = []
        pos = 0
        while len(snippets) < MAX_SEARCH_RESULTS:
            # Find next result block
            pos = html.find('class="result__a"', pos)
            if pos == -1:
                break

            # Extract title text: find the > after class attr, then </a>
            tag_end = html.find('>', pos)
            if tag_end == -1:
                break
            title_start = tag_end + 1
            title_end = html.find('</a>', title_start)
            if title_end == -1:
                break
            title = _strip_tags(html[title_start:title_end])

            # Extract snippet: find result__snippet between this result and next
            snip_pos = html.find('class="result__snippet', pos)
            next_result = html.find('class="result__a"', title_end + 1)
            desc = ""
            if snip_pos != -1 and (next_result == -1 or snip_pos < next_result):
                stag_end = html.find('>', snip_pos)
                if stag_end != -1:
                    snip_start = stag_end + 1
                    snip_end = html.find('</a>', snip_start)
                    if snip_end == -1:
                        snip_end = html.find('</td>', snip_start)
                    if snip_end != -1:
                        desc = _strip_tags(html[snip_start:snip_end])
                        if len(desc) > MAX_SNIPPET_LEN:
                            desc = desc[:MAX_SNIPPET_LEN]

            if title:
                snippets.append("%s: %s" % (title, desc))

            pos = title_end + 1

        # Free HTML immediately — it's 30-60KB
        del html
        gc.collect()

        print("[Search] DDG '%s' -> %d results" % (query, len(snippets)))
        return snippets

    except Exception as e:
        print("[Search] DDG error: %s" % e)
        if r:
            try:
                r.close()
            except Exception:
                pass
        gc.collect()
        return []


def web_search(query, engine):
    """Dispatch search to the active engine. Returns list of snippet strings."""
    if engine == "brave" and BRAVE_API_KEY:
        return brave_search(query)
    return duckduckgo_search(query)

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def get_session(user_id):
    """Get or create per-user session (in-memory only)."""
    if user_id not in sessions:
        # Evict oldest session if at capacity (save RAM)
        if len(sessions) >= MAX_SESSIONS:
            oldest_key = next(iter(sessions))
            del sessions[oldest_key]
            gc.collect()
        # Pick first available provider or default
        prov = DEFAULT_PROVIDER if DEFAULT_PROVIDER in PROVIDERS else next(iter(PROVIDERS), "")
        # Pick default search engine
        eng = "brave" if (SEARCH_ENGINE == "brave" and BRAVE_API_KEY) else "duckduckgo"
        sessions[user_id] = {
            "provider": prov,
            "model": PROVIDERS[prov]["default_model"] if prov else "",
            "history": [],
            "web_search": True,
            "search_engine": eng,
        }
    return sessions[user_id]

# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def handle_command(chat_id, text, user_id):
    """Route /commands. Returns True if handled."""
    parts = text.strip().split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    # Strip @botname suffix from commands
    if "@" in cmd:
        cmd = cmd.split("@")[0]

    s = get_session(user_id)

    if cmd == "/start":
        prov_list = ", ".join(PROVIDERS.keys())
        tg_send(chat_id,
                "ESP32-C3 AI Bot\n\n"
                "Provider: %s\nModel: %s\n"
                "Available: %s\n\n"
                "Commands: /help" % (
                    s["provider"], s["model"], prov_list))
        return True

    if cmd == "/help":
        help_text = ("Commands:\n"
                "/provider [name] - switch provider\n"
                "/models - list models (auto-fetched!)\n"
                "/model [id] - switch model\n"
                "/refresh - re-fetch model lists from APIs\n"
                "/clear - clear history\n"
                "/status - device info\n"
                "/web [on|off|brave|ddg] - web search\n"
                "/help - this message\n\n"
                "Just type a message to chat!")
        tg_send(chat_id, help_text)
        return True

    if cmd == "/provider":
        if not arg:
            prov_list = ", ".join(PROVIDERS.keys())
            tg_send(chat_id,
                    "Current: %s\nAvailable: %s\n\n"
                    "Use: /provider <name>" % (s["provider"], prov_list))
            return True
        arg_lower = arg.lower()
        if arg_lower not in PROVIDERS:
            tg_send(chat_id,
                    "Provider '%s' not found.\nAvailable: %s" % (
                        arg, ", ".join(PROVIDERS.keys())))
            return True
        s["provider"] = arg_lower
        s["model"] = PROVIDERS[arg_lower]["default_model"]
        s["history"] = []
        tg_send(chat_id,
                "Switched to %s\nModel: %s\nHistory cleared." % (
                    PROVIDERS[arg_lower]["name"], s["model"]))
        return True

    if cmd == "/models":
        prov = PROVIDERS.get(s["provider"])
        if not prov:
            tg_send(chat_id, "No provider selected.")
            return True
        models = get_models(s["provider"])
        source = "API" if prov["can_fetch"] else "hardcoded"
        lines = ["%s models (%s, %d):" % (prov["name"], source, len(models))]
        for m in models:
            marker = " [current]" if m == s["model"] else ""
            lines.append("- %s%s" % (m, marker))
        lines.append("\nUse: /model <id>")
        if prov["can_fetch"]:
            lines.append("/refresh to re-fetch from API")
        tg_send(chat_id, "\n".join(lines))
        return True

    if cmd == "/refresh":
        tg_send(chat_id, "Refreshing model lists...")
        refresh_all_models()
        # Report results
        lines = ["Model lists refreshed:"]
        for k, p in PROVIDERS.items():
            source = "API" if p["can_fetch"] else "hardcoded"
            lines.append("- %s: %d models (%s)" % (p["name"], len(p["models"]), source))
        tg_send(chat_id, "\n".join(lines))
        return True

    if cmd == "/model":
        if not arg:
            tg_send(chat_id,
                    "Current model: %s\nUse: /model <id>" % s["model"])
            return True
        prov = PROVIDERS.get(s["provider"])
        if not prov:
            tg_send(chat_id, "No provider selected.")
            return True
        models = get_models(s["provider"])
        if arg in models:
            s["model"] = arg
            tg_send(chat_id, "Switched to model: %s" % arg)
        else:
            tg_send(chat_id,
                    "Model '%s' not in list.\n"
                    "Use /models to see available.\n\n"
                    "Tip: you can force any model by typing the exact ID — "
                    "it may or may not work with the provider." % arg)
            # Allow setting arbitrary model IDs (user knows what they're doing)
            s["model"] = arg
        return True

    if cmd == "/clear":
        s["history"] = []
        gc.collect()
        tg_send(chat_id, "History cleared.")
        return True

    if cmd == "/status":
        wlan = network.WLAN(network.STA_IF)
        uptime = time.time() - boot_time
        h = uptime // 3600
        m = (uptime % 3600) // 60
        rssi = "N/A"
        try:
            rssi_val = wlan.status("rssi")
            if rssi_val >= -50:
                strength = "Strong"
            elif rssi_val >= -70:
                strength = "Medium"
            else:
                strength = "Weak"
            rssi = "%d dBm (%s)" % (rssi_val, strength)
        except Exception:
            pass
        # Fetch public IP (lightweight plain-text API)
        pub_ip = "N/A"
        r = None
        try:
            r = urequests.get("http://api.ipify.org")
            pub_ip = r.text.strip()
            r.close()
            r = None
        except Exception:
            if r:
                try:
                    r.close()
                except Exception:
                    pass
        gc.collect()
        free_ram = gc.mem_free()
        used_ram = gc.mem_alloc()
        total_ram = free_ram + used_ram
        ram_pct = (used_ram * 100) // total_ram if total_ram else 0
        cpu_mhz = freq() // 1000000
        ddns = duckdns_status
        tg_send(chat_id,
                "ESP32-C3 Status\n\n"
                "WiFi: %s\n"
                "RSSI: %s\n"
                "IP: %s\n"
                "Public IP: %s\n"
                "DuckDNS: %s\n"
                "Uptime: %dh %dm\n"
                "CPU: %d MHz\n"
                "RAM: %d/%d bytes (%d%% used)\n"
                "Provider: %s\n"
                "Model: %s\n"
                "History: %d msgs" % (
                    WIFI_SSID, rssi,
                    wlan.ifconfig()[0] if wlan.isconnected() else "disconnected",
                    pub_ip,
                    ddns,
                    h, m,
                    cpu_mhz,
                    used_ram, total_ram, ram_pct,
                    s["provider"], s["model"],
                    len(s["history"])))
        return True

    if cmd == "/web":
        if not arg:
            status = "ON" if s.get("web_search") else "OFF"
            eng = s.get("search_engine", "brave")
            eng_label = "Brave API" if eng == "brave" else "DuckDuckGo"
            tg_send(chat_id,
                    "Web search: %s\nEngine: %s\n\n"
                    "Use: /web on | /web off\n"
                    "/web brave - switch to Brave API\n"
                    "/web ddg - switch to DuckDuckGo (free)" % (status, eng_label))
            return True
        a = arg.lower()
        if a == "on":
            s["web_search"] = True
            eng = s.get("search_engine", "brave")
            eng_label = "Brave API" if eng == "brave" else "DuckDuckGo"
            tg_send(chat_id, "Web search enabled (%s)." % eng_label)
        elif a == "off":
            s["web_search"] = False
            tg_send(chat_id, "Web search disabled.")
        elif a == "brave":
            if not BRAVE_API_KEY:
                tg_send(chat_id, "Brave API key not configured.\nUse /web ddg for free search.")
            else:
                s["search_engine"] = "brave"
                s["web_search"] = True
                tg_send(chat_id, "Switched to Brave Search API.")
        elif a == "ddg" or a == "duckduckgo":
            s["search_engine"] = "duckduckgo"
            s["web_search"] = True
            tg_send(chat_id, "Switched to DuckDuckGo (free, no API key).")
        else:
            tg_send(chat_id, "Use: /web on|off|brave|ddg")
        return True

    # Unknown command
    tg_send(chat_id, "Unknown command: %s\nTry /help" % cmd)
    return True

# ============================================================================
# MESSAGE HANDLER
# ============================================================================

def _parse_search_response(response):
    """Parse AI response for SEARCH: directive. Returns cleaned query or None.
    Handles common model quirks: repeated prefixes, quoted queries, extra text."""
    stripped = response.strip()
    upper = stripped.upper()

    # Must start with SEARCH: (possibly after whitespace)
    if not upper.startswith("SEARCH:"):
        return None

    # Take only the first line (models sometimes append full answers)
    raw = stripped[7:].strip()
    query = raw.split('\n')[0].strip()

    # Strip quotes some models wrap around the query
    if len(query) > 2 and query[0] == '"' and query[-1] == '"':
        query = query[1:-1].strip()
    if len(query) > 2 and query[0] == "'" and query[-1] == "'":
        query = query[1:-1].strip()

    # Strip repeated SEARCH: prefixes (some models echo it)
    safety = 3  # max iterations to prevent infinite loop
    while safety > 0 and query.upper().startswith('SEARCH:'):
        query = query[7:].strip()
        safety -= 1

    query = query[:150]  # cap length for search API
    return query if query else None


def _do_search_and_answer(chat_id, s, query):
    """Perform web search and call AI with results. Returns response string.
    Shared by both direct-search and AI-requested-search paths."""
    engine = s.get("search_engine", "brave")
    print("[Bot] Searching (%s): '%s'" % (engine, query))
    tg_send_action(chat_id)
    if wdt:
        wdt.feed()
    snippets = web_search(query, engine)

    if not snippets:
        # Search returned nothing — let AI answer from its knowledge
        del snippets
        gc.collect()
        return ai_chat(s["provider"], s["model"], s["history"])

    # Build search context (use list+join instead of += to avoid O(n²) copies)
    parts = ["Web results for '%s':" % query]
    for i, snip in enumerate(snippets):
        parts.append("%d. %s" % (i + 1, snip))
    ctx = '\n'.join(parts)
    del snippets
    del parts
    gc.collect()

    # Build messages for second pass (shallow copy history, add search context)
    search_msgs = list(s["history"])
    search_msgs.append({"role": "assistant", "content": "I'll search the web for that."})
    search_msgs.append({"role": "user", "content": ctx + "\nUsing these search results, answer my original question concisely."})
    tg_send_action(chat_id)
    if wdt:
        wdt.feed()
    response = ai_chat(s["provider"], s["model"], search_msgs)
    del search_msgs
    del ctx
    gc.collect()
    return response


def handle_message(chat_id, text, user_id):
    """Handle regular text messages — forward to AI provider.

    Smart search flow (3 tiers):
      1. 'no'     → no search prompt, no search API call  (cheapest)
      2. 'direct' → skip AI ask, search immediately       (1 AI call saved)
      3. 'maybe'  → inject search prompt, let AI decide   (current behavior)

    This saves RAM, tokens, and latency for the majority of messages
    that don't need web search (greetings, coding, math, creative tasks)."""
    s = get_session(user_id)

    # Add to history (truncate very long messages to protect RAM)
    if len(text) > MAX_RESPONSE_LEN:
        text = text[:MAX_RESPONSE_LEN]
    s["history"].append({"role": "user", "content": text})

    # Trim history to save RAM
    if len(s["history"]) > MAX_HISTORY:
        s["history"] = s["history"][-MAX_HISTORY:]

    # Send typing indicator
    tg_send_action(chat_id)

    # ── Decide search strategy ──────────────────────────────────────────
    web_on = s.get("web_search")
    search_tier = _wants_search(text) if web_on else 'no'
    print("[Bot] msg=%d chars, search_tier=%s" % (len(text), search_tier))

    # ── Tier 1: DIRECT SEARCH ───────────────────────────────────────────
    # High-confidence search need → extract query from user text, search
    # immediately, then call AI ONCE with results.  Saves one API call.
    if search_tier == 'direct':
        query = _extract_search_query(text)
        response = _do_search_and_answer(chat_id, s, query)

    # ── Tier 2: MAYBE SEARCH ────────────────────────────────────────────
    # Possible search need → inject search prompt, let AI decide.
    # If AI says SEARCH:, perform it. If not, use the direct answer.
    elif search_tier == 'maybe':
        prompt = SYSTEM_PROMPT + get_search_prompt()
        response = ai_chat(s["provider"], s["model"], s["history"], prompt)

        query = _parse_search_response(response)
        if query:
            del response
            gc.collect()
            response = _do_search_and_answer(chat_id, s, query)
            del query
        else:
            del query

    # ── Tier 3: NO SEARCH ───────────────────────────────────────────────
    # No search signals → plain AI call, no search prompt overhead.
    else:
        response = ai_chat(s["provider"], s["model"], s["history"])

    # ── Store response & send ───────────────────────────────────────────
    # Don't pollute limited history with error messages (wastes precious slots)
    is_error = (response.startswith("Error: provider") or
                response.startswith("Error calling") or
                response.startswith("API Error:") or
                response.startswith("(no response from"))
    if is_error:
        # Roll back the user message that caused the error
        if s["history"] and s["history"][-1].get("role") == "user":
            s["history"].pop()
    else:
        s["history"].append({"role": "assistant", "content": response})

        # Trim again after adding response
        if len(s["history"]) > MAX_HISTORY:
            s["history"] = s["history"][-MAX_HISTORY:]

    tg_send(chat_id, response)
    del response
    gc.collect()

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    global boot_time, tg_offset, last_duckdns

    boot_time = time.time()
    print("=" * 40)
    print("ESP32-C3 Telegram Bot starting...")
    print("Providers:", ", ".join(PROVIDERS.keys()) if PROVIDERS else "NONE!")
    print("=" * 40)

    if not PROVIDERS:
        print("ERROR: No API keys configured! Edit the config section.")
        return

    # Hardware watchdog — auto-reboots if loop hangs for 120s
    global wdt
    wdt = WDT(timeout=120000)  # 120 seconds
    print("[WDT] Watchdog enabled (120s timeout)")

    # Step 1: Connect WiFi (retries forever)
    wlan = wifi_connect()

    # Step 1.5: Sync time via NTP (so AI knows the current date)
    sync_ntp()

    # Step 2: Initial DuckDNS update
    last_duckdns = 0  # force immediate update
    update_duckdns()

    # Step 3: Fetch model lists (dynamic for supported providers)
    print("[Bot] Fetching model lists...")
    refresh_all_models()

    # Step 4: Telegram polling loop
    print("[Bot] Starting Telegram polling...")

    while True:
        # Feed watchdog — prevents auto-reboot during normal operation
        wdt.feed()

        # Check WiFi — reconnect if dropped
        if not wlan.isconnected():
            print("[WiFi] Connection lost, reconnecting...")
            wlan = wifi_connect()
            sync_ntp()  # re-sync time after reconnect
            last_duckdns = 0  # force DuckDNS update after reconnect
            update_duckdns()

        # Periodic DuckDNS update
        update_duckdns()

        # Poll Telegram
        try:
            updates = tg_get_updates(tg_offset)
        except Exception as e:
            print("[Bot] Poll error:", e)
            time.sleep(5)
            gc.collect()
            continue

        for upd in updates:
            if wdt:
                wdt.feed()
            tg_offset = upd.get("update_id", 0) + 1
            msg = upd.get("message")
            if not msg:
                continue

            chat_id = msg.get("chat", {}).get("id")
            text = msg.get("text", "")
            user_id = msg.get("from", {}).get("id", 0)

            if not chat_id or not text:
                continue

            # Access control
            if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
                tg_send(chat_id, "Not authorized.")
                continue

            # Route command or message
            try:
                if text.startswith("/"):
                    handle_command(chat_id, text, user_id)
                else:
                    handle_message(chat_id, text, user_id)
            except Exception as e:
                print("[Bot] Handler error:", e)
                tg_send(chat_id, "Error: %s" % str(e))

            gc.collect()

        # Free the updates list after processing
        del updates
        gc.collect()


# ============================================================================
# ENTRY POINT — crash recovery with auto-reboot
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[Bot] Interrupted by user.")
    except Exception as e:
        print("[Bot] FATAL ERROR:", e)
        print("[Bot] Rebooting in 10 seconds...")
        time.sleep(10)
        reset()
