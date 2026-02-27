"""
ESP32-C3 Super Mini — MicroPython Telegram Bot
================================================
Multi-provider AI chat bot (Groq, Gemini, OpenRouter, Cerebras, NVIDIA)
with DuckDNS updater and WiFi retry loop.

Hardware: ESP32-C3 Super Mini (400 KB SRAM, 4 MB flash)
Runtime:  MicroPython 1.20+

ZERO DISK WRITES — all state is in-memory, reboot resets everything.

CHANGES FROM v1:
  - Replaced Perplexity prompt with lean custom prompt (~250 tokens)
  - Collapsed 'maybe' search tier into 'direct' — saves one full API round-trip
  - History trimmed in pairs (user+assistant) to preserve conversational coherence
  - Error detection now uses a result-type flag instead of fragile string prefix matching
  - Pre-built SYSTEM_PROMPT_WITH_SEARCH to avoid re-allocation on every search call
  - Per-user rate limiting (ignore messages within RATE_LIMIT_SECS of last one)
  - Search snippets formatted with source index so model can reference them cleanly
  - Removed LaTeX, citation-number, and academic formatting instructions
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
WIFI_SSID     = "YOUR_WIFI_SSID"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"

# Telegram
TELEGRAM_TOKEN   = "YOUR_TELEGRAM_BOT_TOKEN"
ALLOWED_USER_IDS = []   # e.g. [123456789, 987654321] — empty = allow all

# DuckDNS (set to "" to disable)
DUCKDNS_TOKEN    = "YOUR_DUCKDNS_TOKEN"
DUCKDNS_DOMAIN   = "YOUR_SUBDOMAIN"   # just the subdomain, not full URL
DUCKDNS_INTERVAL = 300                 # seconds between updates

# AI Provider API Keys (set to "" to disable a provider)
GROQ_API_KEY      = ""
GEMINI_API_KEY    = ""
OPENROUTER_API_KEY = ""
CEREBRAS_API_KEY  = ""
NVIDIA_API_KEY    = ""

# Search
BRAVE_API_KEY    = ""       # set to "" to use DuckDuckGo (free, no key)
MAX_SEARCH_RESULTS = 3      # keep low for RAM
MAX_SNIPPET_LEN  = 200      # chars per snippet

# Bot behaviour
DEFAULT_PROVIDER   = "groq"
MAX_TOKENS         = 512
TEMPERATURE        = 0.7
MAX_HISTORY        = 6      # must be even — stores N/2 exchange pairs
MAX_SESSIONS       = 3
MAX_RESPONSE_LEN   = 4000
RATE_LIMIT_SECS    = 2      # ignore messages arriving faster than this per user

# ============================================================================
# SYSTEM PROMPT — lean, Telegram-native, ~250 tokens
# No Perplexity prompt. No citation numbers. No LaTeX. No ## headers.
# ============================================================================

_PROMPT_BASE = (
    "You are a helpful, concise AI assistant running on an ESP32 microcontroller "
    "and speaking through Telegram.\n\n"
    "FORMATTING RULES — follow strictly:\n"
    "- Use plain Telegram Markdown only: *bold*, _italic_, `code`, ```code blocks```\n"
    "- Never use ## headers — they do not render in Telegram\n"
    "- Never use LaTeX math notation\n"
    "- Use flat bullet lists (- item). Never nest lists\n"
    "- Keep responses concise. Avoid long preambles and sign-offs\n"
    "- For comparisons, use a simple table with | pipes |\n\n"
    "BEHAVIOUR:\n"
    "- Be direct and practical\n"
    "- If you don't know something, say so rather than guessing\n"
    "- When code is requested, use fenced code blocks with the language name"
)

_PROMPT_SEARCH_ADDENDUM = (
    "\n\nSEARCH RESULTS FORMAT:\n"
    "When web search results are provided, they appear as numbered snippets "
    "(1. Title: text). Use them to answer accurately. "
    "You may refer to a result naturally (e.g. 'according to X') but do NOT "
    "write citation numbers like [1] or [2] — there is no reference list."
)

# Pre-built combined prompt — allocated ONCE at startup, reused on every search call
# This avoids string concatenation in the hot path (saves RAM on each request)
SYSTEM_PROMPT             = _PROMPT_BASE
SYSTEM_PROMPT_WITH_SEARCH = _PROMPT_BASE + _PROMPT_SEARCH_ADDENDUM

# ============================================================================
# SEARCH HEURISTIC — two tiers (direct / no)
#
# v2 CHANGE: 'maybe' tier is gone.
#   Old flow: no-search signal → ask AI → AI says SEARCH: → search → answer
#             = 2 API calls for one answer
#   New flow: any search signal → extract query, search immediately → answer
#             = 1 API call for one answer
#
# The _extract_search_query() function is good enough that we don't need to
# waste a round-trip asking the AI to reformulate the query.
# ============================================================================

_KW_TIME = (
    'today', 'tonight', 'now', 'latest', 'current', 'recent',
    'yesterday', 'tomorrow', 'this week', 'this month', 'this year',
    'last week', 'last month', 'last year', 'next week', 'next month',
)
_KW_TOPIC = (
    'price', 'stock', 'crypto', 'bitcoin', 'btc', 'eth', 'market', 'trading',
    'weather', 'forecast', 'temperature',
    'news', 'headline', 'election', 'vote', 'poll',
    'score', 'game', 'match', 'tournament', 'playoff', 'standings',
    'release', 'released', 'launched', 'announced', 'update',
    'dead', 'died', 'alive', 'arrested', 'fired', 'hired', 'resigned',
    'war', 'earthquake', 'hurricane', 'disaster', 'outbreak',
)
_KW_NO_SEARCH = (
    'write', 'code', 'create', 'generate', 'build', 'make', 'implement',
    'explain', 'define', 'describe', 'summarize', 'rewrite', 'translate',
    'fix', 'debug', 'refactor', 'optimize', 'calculate', 'solve', 'convert',
    'hello', 'hi ', 'hey ', 'thanks', 'thank you', 'bye', 'good morning',
    'tell me a joke', 'poem', 'story', 'essay', 'list',
)


def _wants_search(text):
    """Return True if message likely needs a web search, False otherwise.

    v2: Binary decision — no 'maybe' tier. Any positive search signal
    triggers a direct search. This saves one full API round-trip per
    search-eligible message compared to the old ask-AI-first approach."""
    low = text.lower()
    length = len(low)

    if length < 12:
        return False

    # Strong no-search signals
    for kw in _KW_NO_SEARCH:
        if kw in low:
            # Check if any search signal overrides it
            for kw2 in _KW_TIME:
                if kw2 in low:
                    return True
            for kw2 in _KW_TOPIC:
                if kw2 in low:
                    return True
            return False

    # Time signals
    for kw in _KW_TIME:
        if kw in low:
            return True

    # Topic signals
    for kw in _KW_TOPIC:
        if kw in low:
            return True

    # Year mention (current ± 1)
    t = time.localtime()
    year = t[0]
    for y in (str(year - 1), str(year), str(year + 1)):
        if y in low:
            return True

    # Question patterns that likely need live data
    if '?' in low and length > 20:
        if 'who ' in low or 'how much' in low or 'how many' in low or 'where ' in low:
            return True

    return False


def _extract_search_query(text):
    """Strip question prefixes and return a clean 1-6 word search query."""
    low = text.strip()
    for prefix in (
        'what is the', 'what are the', 'what is', 'what are',
        'how much is', 'how much does', 'how much',
        'who is the', 'who is', 'who won the', 'who won',
        'where is the', 'where is',
        'when is the', 'when is', 'when does',
        'tell me about', 'search for', 'look up', 'find',
    ):
        if low.lower().startswith(prefix):
            low = low[len(prefix):].strip()
            break
    low = low.rstrip('?').strip()
    return low[:120] if low else text[:120]

# ============================================================================
# PROVIDER DEFINITIONS
# ============================================================================

PROVIDERS = {}

if GROQ_API_KEY:
    PROVIDERS["groq"] = {
        "name": "Groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "models_url": "https://api.groq.com/openai/v1/models",
        "can_fetch": True,
        "key": GROQ_API_KEY,
        "default_model": "llama-3.3-70b-versatile",
        "fallback": [
            "llama-3.3-70b-versatile",
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
        "can_fetch": True,
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
        "can_fetch": False,
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
        "can_fetch": True,
        "key": CEREBRAS_API_KEY,
        "default_model": "llama-3.3-70b",
        "fallback": [
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
        "can_fetch": False,
        "key": NVIDIA_API_KEY,
        "default_model": "meta/llama-3.3-70b-instruct",
        "fallback": [
            "meta/llama-3.3-70b-instruct",
            "meta/llama-3.1-405b-instruct",
            "deepseek-ai/deepseek-v3.2",
            "qwen/qwen3-235b-a22b",
            "mistralai/mixtral-8x22b-instruct-v0.1",
            "moonshotai/kimi-k2-instruct-0905",
        ],
        "models": [],
    }

# ============================================================================
# DYNAMIC MODEL FETCHING
# ============================================================================

def _model_sort_key(model_id):
    mid = model_id.lower()
    i = 0
    while i < len(mid):
        if mid[i].isdigit():
            j = i
            while j < len(mid) and (mid[j].isdigit() or mid[j] == '.'):
                j += 1
            if j < len(mid) and mid[j] == 'b':
                try:
                    return -float(mid[i:j])
                except Exception:
                    pass
            i = j
        else:
            i += 1
    return 0


def fetch_models(provider_key):
    prov = PROVIDERS.get(provider_key)
    if not prov or not prov["can_fetch"] or not prov["models_url"]:
        return None
    headers = {"Authorization": "Bearer " + prov["key"]}
    gc.collect()
    r = None
    try:
        r = urequests.get(prov["models_url"], headers=headers)
        data = r.json()
        r.close(); r = None
        gc.collect()
        models_data = data.get("data", [])
        model_ids = [m.get("id", "") for m in models_data if m.get("id")]
        del data, models_data
        gc.collect()
        model_ids.sort(key=_model_sort_key)
        if model_ids:
            prov["models"] = model_ids
            print("[Models] %s: fetched %d models" % (prov["name"], len(model_ids)))
            return model_ids
        return None
    except Exception as e:
        print("[Models] %s fetch error: %s" % (prov["name"], e))
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
        return None


def refresh_all_models():
    for key, prov in PROVIDERS.items():
        if wdt: wdt.feed()
        if prov["can_fetch"]:
            if fetch_models(key) is None:
                prov["models"] = list(prov["fallback"])
        else:
            prov["models"] = list(prov["fallback"])
        gc.collect()


def get_models(provider_key):
    prov = PROVIDERS.get(provider_key)
    if not prov:
        return []
    if not prov["models"]:
        if prov["can_fetch"]:
            if fetch_models(provider_key) is None:
                prov["models"] = list(prov["fallback"])
        else:
            prov["models"] = list(prov["fallback"])
    return prov["models"]

# ============================================================================
# GLOBAL STATE
# ============================================================================

# AI call result types — avoids fragile string-prefix error detection
AI_OK    = 0
AI_ERROR = 1

sessions      = {}
boot_time     = 0
tg_offset     = 0
last_duckdns  = 0
duckdns_status = "Disabled"
wdt           = None

# ============================================================================
# WIFI
# ============================================================================

def wifi_connect():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        print("[WiFi] Already connected:", wlan.ifconfig()[0])
        return wlan
    print("[WiFi] Connecting to", WIFI_SSID)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    delay = 2
    while not wlan.isconnected():
        if wdt: wdt.feed()
        print("[WiFi] Waiting... retry in %ds" % delay)
        time.sleep(delay)
        delay = min(delay * 2, 60)
        if not wlan.isconnected():
            try: wlan.connect(WIFI_SSID, WIFI_PASSWORD)
            except Exception: pass
    print("[WiFi] Connected!", wlan.ifconfig()[0])
    return wlan


def sync_ntp():
    import ntptime
    for server in ("pool.ntp.org", "time.google.com"):
        try:
            ntptime.host = server
            ntptime.settime()
            t = time.localtime()
            print("[NTP] Synced via %s: %04d-%02d-%02d %02d:%02d UTC" % (
                server, t[0], t[1], t[2], t[3], t[4]))
            return
        except Exception as e:
            print("[NTP] %s failed: %s" % (server, e))
    print("[NTP] All servers failed — date in prompts may be wrong")

# ============================================================================
# DUCKDNS
# ============================================================================

def update_duckdns():
    global last_duckdns, duckdns_status
    if not DUCKDNS_TOKEN or not DUCKDNS_DOMAIN:
        return
    now = time.time()
    if now - last_duckdns < DUCKDNS_INTERVAL:
        return
    last_duckdns = now
    url = ("https://www.duckdns.org/update?domains=%s&token=%s&verbose=true"
           % (DUCKDNS_DOMAIN, DUCKDNS_TOKEN))
    r = None
    try:
        r = urequests.get(url)
        result = r.text.strip()
        r.close(); r = None
        print("[DuckDNS]", result)
        duckdns_status = "Running" if result.startswith("OK") else "Error: %s" % result.split("\n")[0]
    except Exception as e:
        print("[DuckDNS] Error:", e)
        duckdns_status = "Error: %s" % str(e)
        if r:
            try: r.close()
            except Exception: pass
    gc.collect()

# ============================================================================
# TELEGRAM API
# ============================================================================

TG_BASE = "https://api.telegram.org/bot" + TELEGRAM_TOKEN


def tg_get_updates(offset):
    url = TG_BASE + "/getUpdates?timeout=30&offset=%d" % offset
    r = None
    try:
        r = urequests.get(url)
        data = r.json()
        r.close(); r = None
        gc.collect()
        if data.get("ok"):
            result = data.get("result", [])
            del data
            return result
        del data
    except Exception as e:
        print("[TG] getUpdates error:", e)
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
    return []


def tg_send(chat_id, text):
    MAX_LEN = 4096
    url = TG_BASE + "/sendMessage"
    chunks = []
    split_len = MAX_LEN - 10
    while len(text) > MAX_LEN:
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
        for try_markdown in (True, False):
            payload_obj = {"chat_id": chat_id, "text": chunk}
            if try_markdown:
                payload_obj["parse_mode"] = "Markdown"
            payload = ujson.dumps(payload_obj)
            del payload_obj
            r = None
            try:
                r = urequests.post(url, data=payload,
                                   headers={"Content-Type": "application/json"})
                body = r.json()
                r.close(); r = None
                ok = bool(body.get("ok"))
                del body
                del payload
                gc.collect()
                if ok:
                    break
            except Exception as e:
                print("[TG] send error:", e)
                if r:
                    try: r.close()
                    except Exception: pass
            del payload
            gc.collect()


def tg_send_action(chat_id):
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
# AI PROVIDER
# Returns a tuple: (AI_OK | AI_ERROR, response_text)
# Using a result type instead of string-prefix matching to detect errors.
# ============================================================================

def ai_chat(provider_key, model, messages, sys_prompt=None):
    prov = PROVIDERS.get(provider_key)
    if not prov:
        return AI_ERROR, "Error: provider '%s' not available." % provider_key

    prompt = sys_prompt if sys_prompt is not None else SYSTEM_PROMPT
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
    del body, chat_msgs
    gc.collect()

    r = None
    try:
        r = urequests.post(prov["url"], data=payload, headers=headers)
        del payload
        data = r.json()
        r.close(); r = None
        gc.collect()

        choices = data.get("choices")
        if choices and len(choices) > 0:
            content = choices[0].get("message", {}).get("content", "(empty response)")
            del data
            if len(content) > MAX_RESPONSE_LEN:
                content = content[:MAX_RESPONSE_LEN] + "\n\n[truncated — response too long for ESP32]"
            return AI_OK, content

        err = data.get("error")
        del data
        if err:
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return AI_ERROR, "API Error: %s" % msg

        return AI_ERROR, "(no response from %s)" % prov["name"]

    except Exception as e:
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
        return AI_ERROR, "Error calling %s: %s" % (prov["name"], str(e))

# ============================================================================
# WEB SEARCH
# ============================================================================

def _url_encode(s):
    out = []
    for b in s.encode('utf-8'):
        if (65 <= b <= 90) or (97 <= b <= 122) or (48 <= b <= 57) or b in (45, 46, 95, 126):
            out.append(chr(b))
        elif b == 32:
            out.append('+')
        else:
            out.append('%%%02X' % b)
    return ''.join(out)


def _strip_tags(html_str):
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
    for old, new in (('&amp;','&'),('&lt;','<'),('&gt;','>'),('&quot;','"'),
                     ('&#39;',"'"),('&#x27;',"'"),('&nbsp;',' ')):
        text = text.replace(old, new)
    return text.strip()


def brave_search(query):
    if not BRAVE_API_KEY:
        return []
    q = _url_encode(query)
    url = "https://api.search.brave.com/res/v1/web/search?q=%s&count=%d" % (q, MAX_SEARCH_RESULTS)
    headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}
    gc.collect()
    r = None
    try:
        r = urequests.get(url, headers=headers)
        data = r.json()
        r.close(); r = None
        gc.collect()
        results = data.get("web", {}).get("results", [])
        snippets = []
        for item in results:
            title = item.get("title", "")
            desc  = item.get("description", "")[:MAX_SNIPPET_LEN]
            url_  = item.get("url", "")
            if title or desc:
                snippets.append((title, desc, url_))
        del data, results
        gc.collect()
        print("[Search] Brave '%s' -> %d results" % (query, len(snippets)))
        return snippets
    except Exception as e:
        print("[Search] Brave error: %s" % e)
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
        return []


def duckduckgo_search(query):
    q = _url_encode(query)
    url = "https://html.duckduckgo.com/html/?q=%s" % q
    headers = {"User-Agent": "Mozilla/5.0"}
    gc.collect()
    r = None
    try:
        r = urequests.get(url, headers=headers)
        html = r.text
        r.close(); r = None
        gc.collect()
        snippets = []
        pos = 0
        while len(snippets) < MAX_SEARCH_RESULTS:
            pos = html.find('class="result__a"', pos)
            if pos == -1:
                break
            tag_end = html.find('>', pos)
            if tag_end == -1:
                break
            title_start = tag_end + 1
            title_end = html.find('</a>', title_start)
            if title_end == -1:
                break
            title = _strip_tags(html[title_start:title_end])
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
                        desc = _strip_tags(html[snip_start:snip_end])[:MAX_SNIPPET_LEN]
            if title:
                snippets.append((title, desc, ""))
            pos = title_end + 1
        del html
        gc.collect()
        print("[Search] DDG '%s' -> %d results" % (query, len(snippets)))
        return snippets
    except Exception as e:
        print("[Search] DDG error: %s" % e)
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
        return []


def web_search(query, engine):
    """Returns list of (title, snippet, url) tuples."""
    if engine == "brave" and BRAVE_API_KEY:
        return brave_search(query)
    return duckduckgo_search(query)

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def get_session(user_id):
    if user_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            oldest_key = next(iter(sessions))
            del sessions[oldest_key]
            gc.collect()
        prov = DEFAULT_PROVIDER if DEFAULT_PROVIDER in PROVIDERS else next(iter(PROVIDERS), "")
        eng  = "brave" if (BRAVE_API_KEY) else "duckduckgo"
        sessions[user_id] = {
            "provider":      prov,
            "model":         PROVIDERS[prov]["default_model"] if prov else "",
            "history":       [],
            "web_search":    True,
            "search_engine": eng,
            "last_msg_time": 0,   # for rate limiting
        }
    return sessions[user_id]

# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def handle_command(chat_id, text, user_id):
    parts = text.strip().split(None, 1)
    cmd   = parts[0].lower()
    arg   = parts[1].strip() if len(parts) > 1 else ""
    if "@" in cmd:
        cmd = cmd.split("@")[0]

    s = get_session(user_id)

    if cmd == "/start":
        prov_list = ", ".join(PROVIDERS.keys())
        tg_send(chat_id,
            "ESP32-C3 AI Bot\n\n"
            "Provider: %s\nModel: %s\n"
            "Available: %s\n\n"
            "Type /help for commands." % (s["provider"], s["model"], prov_list))
        return True

    if cmd == "/help":
        tg_send(chat_id,
            "*Commands*\n"
            "/provider [name] — switch AI provider\n"
            "/models — list available models\n"
            "/model [id] — switch model\n"
            "/refresh — re-fetch model lists\n"
            "/clear — clear conversation history\n"
            "/web [on|off|brave|ddg] — web search toggle\n"
            "/status — device info\n"
            "/help — this message\n\n"
            "Just type to chat!")
        return True

    if cmd == "/provider":
        if not arg:
            tg_send(chat_id,
                "Current: %s\nAvailable: %s\n\nUse: /provider <name>" % (
                    s["provider"], ", ".join(PROVIDERS.keys())))
            return True
        arg_lower = arg.lower()
        if arg_lower not in PROVIDERS:
            tg_send(chat_id, "Provider '%s' not found.\nAvailable: %s" % (
                arg, ", ".join(PROVIDERS.keys())))
            return True
        s["provider"] = arg_lower
        s["model"]    = PROVIDERS[arg_lower]["default_model"]
        s["history"]  = []
        tg_send(chat_id, "Switched to %s\nModel: %s\nHistory cleared." % (
            PROVIDERS[arg_lower]["name"], s["model"]))
        return True

    if cmd == "/models":
        prov = PROVIDERS.get(s["provider"])
        if not prov:
            tg_send(chat_id, "No provider selected.")
            return True
        models = get_models(s["provider"])
        source = "API" if prov["can_fetch"] else "hardcoded"
        lines  = ["%s models (%s, %d):" % (prov["name"], source, len(models))]
        for m in models:
            marker = " [current]" if m == s["model"] else ""
            lines.append("- %s%s" % (m, marker))
        lines.append("\nUse: /model <id>")
        tg_send(chat_id, "\n".join(lines))
        return True

    if cmd == "/refresh":
        tg_send(chat_id, "Refreshing model lists...")
        refresh_all_models()
        lines = ["Model lists refreshed:"]
        for k, p in PROVIDERS.items():
            source = "API" if p["can_fetch"] else "hardcoded"
            lines.append("- %s: %d models (%s)" % (p["name"], len(p["models"]), source))
        tg_send(chat_id, "\n".join(lines))
        return True

    if cmd == "/model":
        if not arg:
            tg_send(chat_id, "Current model: %s\nUse: /model <id>" % s["model"])
            return True
        prov   = PROVIDERS.get(s["provider"])
        models = get_models(s["provider"]) if prov else []
        s["model"] = arg
        if arg in models:
            tg_send(chat_id, "Switched to model: %s" % arg)
        else:
            tg_send(chat_id,
                "Model '%s' not in list — set anyway.\n"
                "Use /models to see known models." % arg)
        return True

    if cmd == "/clear":
        s["history"] = []
        gc.collect()
        tg_send(chat_id, "History cleared.")
        return True

    if cmd == "/status":
        wlan   = network.WLAN(network.STA_IF)
        uptime = time.time() - boot_time
        h      = uptime // 3600
        m      = (uptime % 3600) // 60
        rssi   = "N/A"
        try:
            v = wlan.status("rssi")
            strength = "Strong" if v >= -50 else "Medium" if v >= -70 else "Weak"
            rssi = "%d dBm (%s)" % (v, strength)
        except Exception:
            pass
        pub_ip = "N/A"
        r = None
        try:
            r = urequests.get("http://api.ipify.org")
            pub_ip = r.text.strip()
            r.close(); r = None
        except Exception:
            if r:
                try: r.close()
                except Exception: pass
        gc.collect()
        free_ram  = gc.mem_free()
        used_ram  = gc.mem_alloc()
        total_ram = free_ram + used_ram
        ram_pct   = (used_ram * 100) // total_ram if total_ram else 0
        search_eng = s.get("search_engine", "duckduckgo")
        search_label = "Brave" if search_eng == "brave" else "DuckDuckGo"
        tg_send(chat_id,
            "ESP32-C3 Status\n\n"
            "WiFi: %s\nRSSI: %s\n"
            "IP: %s\nPublic IP: %s\n"
            "DuckDNS: %s\n"
            "Uptime: %dh %dm\nCPU: %d MHz\n"
            "RAM: %d/%d bytes (%d%% used)\n"
            "Provider: %s\nModel: %s\n"
            "History: %d msgs\n"
            "Web search: %s (%s)" % (
                WIFI_SSID, rssi,
                wlan.ifconfig()[0] if wlan.isconnected() else "disconnected",
                pub_ip,
                duckdns_status,
                h, m, freq() // 1000000,
                used_ram, total_ram, ram_pct,
                s["provider"], s["model"],
                len(s["history"]),
                "ON" if s.get("web_search") else "OFF", search_label))
        return True

    if cmd == "/web":
        if not arg:
            status = "ON" if s.get("web_search") else "OFF"
            eng    = s.get("search_engine", "duckduckgo")
            tg_send(chat_id,
                "Web search: %s\nEngine: %s\n\n"
                "/web on | /web off\n"
                "/web brave — Brave Search API\n"
                "/web ddg — DuckDuckGo (free)" % (
                    status, "Brave API" if eng == "brave" else "DuckDuckGo"))
            return True
        a = arg.lower()
        if a == "on":
            s["web_search"] = True
            eng = s.get("search_engine", "duckduckgo")
            tg_send(chat_id, "Web search enabled (%s)." % ("Brave" if eng == "brave" else "DuckDuckGo"))
        elif a == "off":
            s["web_search"] = False
            tg_send(chat_id, "Web search disabled.")
        elif a == "brave":
            if not BRAVE_API_KEY:
                tg_send(chat_id, "Brave API key not configured.\nUse /web ddg for free search.")
            else:
                s["search_engine"] = "brave"
                s["web_search"]    = True
                tg_send(chat_id, "Switched to Brave Search API.")
        elif a in ("ddg", "duckduckgo"):
            s["search_engine"] = "duckduckgo"
            s["web_search"]    = True
            tg_send(chat_id, "Switched to DuckDuckGo (free).")
        else:
            tg_send(chat_id, "Use: /web on|off|brave|ddg")
        return True

    tg_send(chat_id, "Unknown command: %s\nTry /help" % cmd)
    return True

# ============================================================================
# MESSAGE HANDLER
# ============================================================================

def _build_search_context(query, results):
    """Format search results as numbered snippets for the AI.
    Returns a context string the model can refer to naturally."""
    t = time.localtime()
    today = "%04d-%02d-%02d" % (t[0], t[1], t[2])
    parts = ["Today is %s. Web search results for '%s':" % (today, query)]
    for i, (title, desc, url) in enumerate(results):
        line = "%d. %s" % (i + 1, title)
        if desc:
            line += ": %s" % desc
        parts.append(line)
    return '\n'.join(parts)


def _trim_history(history):
    """Trim history to MAX_HISTORY keeping complete user/assistant pairs.
    Always removes from the front in pairs so context stays coherent.
    MAX_HISTORY should be even (e.g. 6 = 3 pairs)."""
    while len(history) > MAX_HISTORY:
        # Drop oldest pair (user + assistant)
        if len(history) >= 2:
            history.pop(0)
            history.pop(0)
        else:
            history.pop(0)
    return history


def handle_message(chat_id, text, user_id):
    s = get_session(user_id)

    # ── Rate limiting ────────────────────────────────────────────────────
    now = time.time()
    if now - s["last_msg_time"] < RATE_LIMIT_SECS:
        print("[Bot] Rate limit hit for user %d" % user_id)
        return
    s["last_msg_time"] = now

    # ── Sanitise input ───────────────────────────────────────────────────
    if len(text) > MAX_RESPONSE_LEN:
        text = text[:MAX_RESPONSE_LEN]

    # ── Add user message to history ──────────────────────────────────────
    s["history"].append({"role": "user", "content": text})
    _trim_history(s["history"])

    tg_send_action(chat_id)
    if wdt: wdt.feed()

    # ── Decide search strategy ───────────────────────────────────────────
    web_on = s.get("web_search", True)
    do_search = web_on and _wants_search(text)
    print("[Bot] msg=%d chars, do_search=%s" % (len(text), do_search))

    result_type = AI_OK
    response    = ""

    if do_search:
        # ── SEARCH PATH ─────────────────────────────────────────────────
        # Extract query directly from user message — no AI round-trip needed
        query = _extract_search_query(text)
        engine = s.get("search_engine", "duckduckgo")
        print("[Bot] Searching (%s): '%s'" % (engine, query))

        tg_send_action(chat_id)
        if wdt: wdt.feed()

        results = web_search(query, engine)

        if results:
            ctx = _build_search_context(query, results)
            del results
            gc.collect()

            # Build messages: history + search context as a user turn
            search_msgs = list(s["history"][:-1])  # history minus the just-added user msg
            search_msgs.append({"role": "user", "content": text + "\n\n" + ctx})

            tg_send_action(chat_id)
            if wdt: wdt.feed()

            result_type, response = ai_chat(
                s["provider"], s["model"], search_msgs, SYSTEM_PROMPT_WITH_SEARCH
            )
            del search_msgs, ctx
            gc.collect()
        else:
            # Search came back empty — fall through to plain AI call
            del results
            gc.collect()
            result_type, response = ai_chat(s["provider"], s["model"], s["history"])
    else:
        # ── DIRECT AI PATH ───────────────────────────────────────────────
        result_type, response = ai_chat(s["provider"], s["model"], s["history"])

    # ── Update history or roll back on error ─────────────────────────────
    if result_type == AI_OK:
        s["history"].append({"role": "assistant", "content": response})
        _trim_history(s["history"])
    else:
        # Roll back the user message that triggered the error
        if s["history"] and s["history"][-1].get("role") == "user":
            s["history"].pop()
        print("[Bot] AI error — user message rolled back from history")

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
    print("ESP32-C3 Telegram Bot v2 starting...")
    print("Providers:", ", ".join(PROVIDERS.keys()) if PROVIDERS else "NONE!")
    print("=" * 40)

    if not PROVIDERS:
        print("ERROR: No API keys configured!")
        return

    global wdt
    wdt = WDT(timeout=120000)
    print("[WDT] Watchdog enabled (120s)")

    wlan = wifi_connect()
    sync_ntp()

    last_duckdns = 0
    update_duckdns()

    print("[Bot] Fetching model lists...")
    refresh_all_models()

    print("[Bot] Starting Telegram polling...")

    while True:
        wdt.feed()

        if not wlan.isconnected():
            print("[WiFi] Lost connection, reconnecting...")
            wlan = wifi_connect()
            sync_ntp()
            last_duckdns = 0
            update_duckdns()

        update_duckdns()

        try:
            updates = tg_get_updates(tg_offset)
        except Exception as e:
            print("[Bot] Poll error:", e)
            time.sleep(5)
            gc.collect()
            continue

        for upd in updates:
            if wdt: wdt.feed()
            tg_offset = upd.get("update_id", 0) + 1
            msg = upd.get("message")
            if not msg:
                continue

            chat_id = msg.get("chat", {}).get("id")
            text    = msg.get("text", "")
            user_id = msg.get("from", {}).get("id", 0)

            if not chat_id or not text:
                continue

            if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
                tg_send(chat_id, "Not authorized.")
                continue

            try:
                if text.startswith("/"):
                    handle_command(chat_id, text, user_id)
                else:
                    handle_message(chat_id, text, user_id)
            except Exception as e:
                print("[Bot] Handler error:", e)
                tg_send(chat_id, "Error: %s" % str(e))

            gc.collect()

        del updates
        gc.collect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[Bot] Interrupted.")
    except Exception as e:
        print("[Bot] FATAL:", e)
        print("[Bot] Rebooting in 10s...")
        time.sleep(10)
        reset()
