"""
ESP32-C3 Super Mini — MicroPython Telegram Bot
================================================
Multi-provider AI chat bot (Groq, Gemini, OpenRouter, Cerebras, NVIDIA)
with LLM function-calling web search, DuckDNS updater and WiFi retry loop.

Hardware: ESP32-C3 Super Mini (400 KB SRAM, 4 MB flash)
Runtime:  MicroPython 1.20+

ZERO DISK WRITES — all state is in-memory, reboot resets everything.
TIP: flash as .mpy (mpy-cross) to skip on-device compilation — saves RAM.

CHANGES v4 → v5 (architecture ported from full-size bot.py):
  - Search routing is now NATIVE FUNCTION CALLING (OpenAI tools schema),
    not a separate SEARCH/NOSEARCH decision call. Removed ai_decide_search(),
    _parse_search_query() and all greeting/keyword heuristics (~150 lines).
    Normal chat = 1 API call. Search = 2 calls (tool_call → result → answer).
  - Raw tool_calls dict is echoed back verbatim in the follow-up call —
    Gemini's OpenAI-compat endpoint signs function calls and 400s otherwise.
  - Transient-error retry (429/5xx/timeout keywords) with backoff, from bot.py.
  - <think>/<reasoning> tag stripping for reasoning models (pure string ops,
    no ure import).
  - System prompt carries the tool-usage policy + per-request date injection.
  - SearXNG added as an engine option (JSON API — lighter than DDG HTML).

  CHANGES v5 → v6 (parity with current bot.py):
  - Exa search engine: one small HTTPS POST returns a synthesized answer —
    replaces 'scrape ~20 KB HTML + feed raw snippets', so it is BOTH higher
    quality AND smaller peak heap. Default engine when EXA_API_KEY is set.
  - Provider defaults/fallbacks refreshed to match bot.py; Vercel and a
    user-filled Custom OpenAI-compatible endpoint added.
  - /web exa command; Exa-aware default-engine picker; /model empty-list guard.
  - Rolling summary memory (bot.py smart-compaction, C3-tuned): when the
    6-message window overflows, the oldest turns are compressed by one extra
    LLM call into session["summary"] (~400 chars) and injected ahead of the
    last 2 verbatim messages on every following chat call. Falls back to
    plain trim when free heap < COMPACT_MIN_FREE so a tight chip never OOMs.

  ESP32-C3 RAM fixes (400 KB SRAM):
  - getUpdates: limit=5 + allowed_updates filter. The default (up to 100
    updates) could deliver >100 KB JSON and OOM the heap instantly.
  - Boot-time offset flush (offset=-1) — a WDT reboot no longer re-answers
    the old message backlog.
  - DuckDuckGo moved html/ → lite/ endpoint with a 20 KB capped read
    (the full page is 60-100 KB).
  - Model lists are NOT fetched at boot — fallback lists are seeded instantly,
    real fetch happens only on /models or /refresh.
  - History clipped to 1200 chars/message when stored.
  - Emergency session-eviction floor raised 20 KB → 40 KB (one TLS handshake
    needs ~30-40 KB headroom).
  - Typing-indicator calls cut 4 → 2 per message (each one is a full TLS POST).
"""

import network
import urequests
import ujson
import time
import gc
import ntptime
from machine import WDT, reset, freq

# ============================================================================
# CONFIGURATION — Edit these values before flashing
# ============================================================================

WIFI_SSID        = "YOUR_WIFI_SSID"
WIFI_PASSWORD    = "YOUR_WIFI_PASSWORD"

TELEGRAM_TOKEN   = "YOUR_TELEGRAM_BOT_TOKEN"
ALLOWED_USER_IDS = []   # e.g. [123456789] — empty = allow all

DUCKDNS_TOKEN    = "YOUR_DUCKDNS_TOKEN"
DUCKDNS_DOMAIN   = "YOUR_SUBDOMAIN"
DUCKDNS_INTERVAL = 900   # DuckDNS accepts 15 min fine — halves per-hour TLS handshakes

GROQ_API_KEY       = ""
GEMINI_API_KEY     = ""
OPENROUTER_API_KEY = ""
CEREBRAS_API_KEY   = ""
NVIDIA_API_KEY     = ""
VERCEL_API_KEY     = ""

BRAVE_API_KEY      = ""
SEARXNG_URL        = ""     # e.g. "http://192.168.1.10:8080" — no trailing slash
EXA_API_KEY        = ""     # https://dashboard.exa.ai/ — synthesized answer via HTTPS, lighter than scraping
MAX_SEARCH_RESULTS = 3
MAX_SNIPPET_LEN    = 200
MAX_ANSWER_LEN     = 1200   # cap on Exa's synthesized answer fed to the model

# ── Custom OpenAI-compatible endpoint (fill these three to enable) ──────────
CUSTOM_API_KEY       = ""
CUSTOM_BASE_URL      = ""   # e.g. "https://api.example.com" — "/v1/chat/completions" is appended
CUSTOM_DEFAULT_MODEL = ""

DEFAULT_PROVIDER = "groq"
MAX_TOKENS       = 512
TEMPERATURE      = 0.7
MAX_HISTORY      = 6
MAX_SESSIONS     = 3
MAX_RESPONSE_LEN = 4000
HISTORY_MSG_MAX  = 1200   # per-message clip when stored in history
RATE_LIMIT_SECS  = 2
MEM_FLOOR        = 40000  # one TLS handshake needs ~30-40 KB free

_DEBUG = True  # False for production flash — suppresses print() string allocs

# ── SMART COMPACTION (rolling summary memory, C3-tuned) ─────────────────────
# When the history window overflows, the oldest turns are summarized by one
# extra LLM call into session["summary"] instead of being dropped. The chat
# call then receives summary + last COMPACT_KEEP_RECENT messages.
# Budgets are ~10x leaner than bot.py (400 chars ≈ 100 tokens) — remember
# durable facts, not conversation texture.
COMPACT_THRESHOLD   = MAX_HISTORY  # compact when history would exceed this
COMPACT_KEEP_RECENT = 2            # verbatim messages kept after compaction
MAX_SUMMARY_CHARS   = 400
COMPACT_MAX_TOKENS  = 180
# Skip compaction (fall back to plain trim) unless this much heap is free —
# the extra request must not cannibalize the chat call that follows.
COMPACT_MIN_FREE    = 80000

# ============================================================================
# SYSTEM PROMPT — ported from bot.py, tool-usage policy included.
# {date} is replaced per-request. No separate search router prompt exists
# anymore: the model routes itself via function calling.
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
    "- For comparisons: use *bold item name* on its own line, then bullet points for attributes. "
    "Never use | pipe | tables — they do not render in Telegram\n\n"
    "BEHAVIOUR (in priority order — earlier wins):\n"
    "1. If you're not sure, say so explicitly and early. It is ALWAYS better to say "
    "'I don't know' than to state something wrong with confidence.\n"
    "2. Be direct and practical\n"
    "3. When code is requested, use fenced code blocks with the language name\n\n"
    "WEB SEARCH TOOL:\n"
    "You have access to a `web_search` tool. Call it automatically whenever the user's "
    "question requires current, real-time, or time-sensitive information such as: "
    "news, prices, sports scores, weather, product availability, recent events, "
    "people's current status, or any fact you are not fully confident about. "
    "Do NOT call the tool for: coding help, math, creative writing, definitions, "
    "greetings, opinions, or general knowledge you can answer confidently. "
    "When search results are provided to you, base your answer ONLY on those results. "
    "Do NOT use your training data to fill in facts absent from the results — "
    "if the results lack enough info, say so clearly. "
    "If the search returns no usable results at all, say plainly 'I couldn't find "
    "current information on that' and stop — do not fall back to memory. "
    "Today's date is {date}."
)


def _log(*args):
    """Debug print — compiled out on production flashes (_DEBUG=False)."""
    if _DEBUG:
        print(*args)


_prompt_cache = {"date": None, "text": None}

def _build_system_prompt():
    """Daily-cached: rebuilding the 1.3 KB prompt per message was ~1.3 KB of
    string churn on every chat call. Only the {date} substitution varies."""
    t = time.localtime()
    today = (t[0], t[1], t[2])
    if _prompt_cache["date"] != today:
        _prompt_cache["date"] = today
        _prompt_cache["text"] = _PROMPT_BASE.replace("{date}", "%04d-%02d-%02d" % today)
    return _prompt_cache["text"]


# ============================================================================
# FUNCTION CALLING — WEB SEARCH TOOL DEFINITION (from bot.py)
#
# Passed to every provider via the OpenAI-compatible tools parameter.
# The model decides autonomously whether to search; no router call needed.
# ============================================================================

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, real-time, or time-sensitive information. "
            "Call this when the user asks about recent news, live prices, sports scores, "
            "weather, product availability, people's current status, recent events, "
            "or any fact you are not fully confident about from training data. "
            "Do NOT call for coding help, math, definitions, greetings, or stable general knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A specific, self-contained web search query. "
                        "Include the subject noun and, for time-sensitive topics, the current year. "
                        "Aim for 4-8 words. Must be fully formed and searchable."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

# ============================================================================
# TRANSIENT API RETRY (from bot.py)
# ============================================================================

_TRANSIENT_ERRORS = (
    "rate limit", "too many requests", "429", "timeout", "timed out",
    "503", "502", "500", "529", "overloaded", "temporarily unavailable",
    "service unavailable", "connection error", "connection reset",
)
_CHAT_MAX_RETRIES = 2


def _is_transient(err_msg):
    low = err_msg.lower()
    for kw in _TRANSIENT_ERRORS:
        if kw in low:
            return True
    return False


# ============================================================================
# PROVIDER DEFINITIONS
# ============================================================================

PROVIDERS = {}

# Defaults/fallbacks mirror the maintained bot.py. Cached "models" lists live
# only in RAM — boot reseeds them, so edits here DO survive flashing (no cache file).
if GROQ_API_KEY:
    PROVIDERS["groq"] = {
        "name": "Groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "models_url": "https://api.groq.com/openai/v1/models",
        "can_fetch": True,
        "key": GROQ_API_KEY,
        "default_model": "openai/gpt-oss-120b",
        "fallback": ["openai/gpt-oss-120b","llama-3.3-70b-versatile","llama-3.1-8b-instant","openai/gpt-oss-20b"],
        "models": [],
    }

if GEMINI_API_KEY:
    PROVIDERS["gemini"] = {
        "name": "Gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "models_url": "https://generativelanguage.googleapis.com/v1beta/openai/models",
        "can_fetch": True,
        "key": GEMINI_API_KEY,
        "default_model": "gemini-flash-lite-latest",
        "fallback": ["gemini-flash-lite-latest","gemini-2.5-flash","gemini-2.5-pro"],
        "models": [],
    }

if OPENROUTER_API_KEY:
    PROVIDERS["openrouter"] = {
        "name": "OpenRouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "models_url": "",
        "can_fetch": False,
        "key": OPENROUTER_API_KEY,
        "default_model": "openrouter/free",
        "fallback": [
            "openrouter/free",
            "meta-llama/llama-3.3-70b-instruct:free",
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
        "default_model": "llama3.1-8b",
        "fallback": ["llama3.1-8b","gpt-oss-120b","llama-3.3-70b"],
        "models": [],
    }

if NVIDIA_API_KEY:
    PROVIDERS["nvidia"] = {
        "name": "NVIDIA",
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "models_url": "",
        "can_fetch": False,
        "key": NVIDIA_API_KEY,
        "default_model": "nvidia/nemotron-3-super-120b-a12b",
        "fallback": [
            "nvidia/nemotron-3-super-120b-a12b",
            "openai/gpt-oss-120b",
            "deepseek-ai/deepseek-v3.2",
            "qwen/qwen3-235b-a22b",
            "moonshotai/kimi-k2.5",
        ],
        "models": [],
    }

if VERCEL_API_KEY:
    PROVIDERS["vercel"] = {
        "name": "Vercel",
        "url": "https://ai-gateway.vercel.sh/v1/chat/completions",
        "models_url": "https://ai-gateway.vercel.sh/v1/models",
        "can_fetch": True,
        "key": VERCEL_API_KEY,
        "default_model": "perplexity/sonar",
        "fallback": ["perplexity/sonar"],
        "models": [],
    }

if CUSTOM_API_KEY and CUSTOM_BASE_URL and CUSTOM_DEFAULT_MODEL:
    PROVIDERS["custom"] = {
        "name": "Custom",
        "url": CUSTOM_BASE_URL + "/v1/chat/completions",
        "models_url": CUSTOM_BASE_URL + "/v1/models",
        "can_fetch": True,
        "key": CUSTOM_API_KEY,
        "default_model": CUSTOM_DEFAULT_MODEL,
        "fallback": [CUSTOM_DEFAULT_MODEL],
        "models": [],
    }

# ============================================================================
# DYNAMIC MODEL FETCHING — lazy: boot seeds fallback lists, real fetch only
# on /models, /refresh or provider operations that need the live list.
# Fetching 3 providers' model JSON at boot was a startup RAM spike.
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
                try: return -float(mid[i:j])
                except Exception: pass
            i = j
        else:
            i += 1
    return 0


def fetch_models(provider_key):
    prov = PROVIDERS.get(provider_key)
    if not prov or not prov["can_fetch"] or not prov["models_url"]:
        return None
    r = None
    try:
        r = urequests.get(prov["models_url"], headers={"Authorization": "Bearer " + prov["key"]})
        data = r.json(); r.close(); r = None; gc.collect()
        ids = [m.get("id","") for m in data.get("data",[]) if m.get("id")]
        del data; gc.collect()
        ids.sort(key=_model_sort_key)
        if ids:
            prov["models"] = ids
            print("[Models] %s: %d" % (prov["name"], len(ids)))
            return ids
        return None
    except Exception as e:
        print("[Models] %s error: %s" % (provider_key, e))
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
        return None


def refresh_all_models(fetch=True):
    for key, prov in PROVIDERS.items():
        if wdt: wdt.feed()
        if fetch and prov["can_fetch"]:
            if fetch_models(key) is None:
                prov["models"] = list(prov["fallback"])
        elif not prov["models"]:
            prov["models"] = list(prov["fallback"])
        gc.collect()


def get_models(provider_key):
    prov = PROVIDERS.get(provider_key)
    if not prov: return []
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

AI_OK               = 0
AI_ERROR            = 1
AI_TOOLS_UNSUPPORTED = 2   # provider/model rejected the tools parameter

sessions       = {}
boot_time      = 0
tg_offset      = 0
last_duckdns   = 0
duckdns_status = "Disabled"
wdt            = None

# ============================================================================
# WIFI + NTP
# ============================================================================

def wifi_connect():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        print("[WiFi] Already connected:", wlan.ifconfig()[0])
        return wlan
    _log("[WiFi] Connecting to", WIFI_SSID)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    delay = 2
    while not wlan.isconnected():
        if wdt: wdt.feed()
        _log("[WiFi] Waiting %ds..." % delay)
        time.sleep(delay)
        delay = min(delay * 2, 60)
        if not wlan.isconnected():
            try: wlan.connect(WIFI_SSID, WIFI_PASSWORD)
            except Exception: pass
    print("[WiFi] Connected!", wlan.ifconfig()[0])
    return wlan


def sync_ntp():
    for server in ("pool.ntp.org", "time.google.com"):
        try:
            ntptime.host = server; ntptime.settime()
            t = time.localtime()
            _log("[NTP] %s: %04d-%02d-%02d %02d:%02d" % (server,t[0],t[1],t[2],t[3],t[4]))
            return
        except Exception as e:
            _log("[NTP] %s failed: %s" % (server, e))
    print("[NTP] All servers failed")

# ============================================================================
# DUCKDNS
# ============================================================================

def update_duckdns():
    global last_duckdns, duckdns_status
    if not DUCKDNS_TOKEN or not DUCKDNS_DOMAIN: return
    now = time.time()
    if now - last_duckdns < DUCKDNS_INTERVAL: return
    last_duckdns = now
    r = None
    try:
        r = urequests.get(
            "https://www.duckdns.org/update?domains=%s&token=%s&verbose=true" % (DUCKDNS_DOMAIN, DUCKDNS_TOKEN))
        result = _read_body_limited(r, 256).strip(); r.close(); r = None
        _log("[DuckDNS]", result)
        duckdns_status = "Running" if result.startswith("OK") else "Error: %s" % result.split("\n")[0]
    except Exception as e:
        _log("[DuckDNS] Error:", e)
        duckdns_status = "Error: %s" % str(e)
        if r:
            try: r.close()
            except Exception: pass
    gc.collect()

# ============================================================================
# TELEGRAM
# ============================================================================

TG_BASE = "https://api.telegram.org/bot" + TELEGRAM_TOKEN

# %% escapes are required — this string goes through the % operator.
# limit=5 + allowed_updates keeps the worst-case JSON payload tiny.
# (Default is up to 100 updates of any type: a >100 KB heap bomb.)
# edited_message is deliberately NOT filtered in: re-answering an edit costs
# a full tool round-trip the C3 can't afford, and edited batches can move
# tg_offset past text the user still expects a reply to.
TG_POLL_URL = ("/getUpdates?timeout=25&limit=5"
               "&allowed_updates=%%5B%%22message%%22%%5D"
               "&offset=%d")


def tg_get_updates(offset):
    r = None
    try:
        r = urequests.get(TG_BASE + TG_POLL_URL % offset)
        data = r.json(); r.close(); r = None; gc.collect()
        if data.get("ok"):
            result = data.get("result", []); del data; return result
        del data
        # API-level failure (401/502 HTML/etc.) — caller would re-poll instantly.
        time.sleep(5)
    except Exception as e:
        print("[TG] getUpdates error:", e)
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
    return []


def tg_flush_updates():
    """Fast-forward past the pending backlog on boot.

    After a WDT reset tg_offset is 0, so Telegram re-delivers old messages
    and the bot re-answers them. offset=-1 asks for just the latest update;
    we ack it so polling resumes past it.
    """
    global tg_offset
    r = None
    try:
        r = urequests.get(TG_BASE + "/getUpdates?timeout=0&limit=1&offset=-1")
        data = r.json(); r.close(); r = None; gc.collect()
        result = data.get("result") or []
        if result:
            tg_offset = result[-1].get("update_id", 0) + 1
            print("[TG] Flushed backlog, offset=%d" % tg_offset)
        del data
    except Exception as e:
        print("[TG] Flush error:", e)
        if r:
            try: r.close()
            except Exception: pass
    gc.collect()


def tg_send(chat_id, text):
    MAX_LEN = 4096
    url = TG_BASE + "/sendMessage"
    chunks = []
    split_len = MAX_LEN - 10
    while len(text) > MAX_LEN:
        split_at = text.rfind("\n", 0, split_len)
        if split_at < 1: split_at = split_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text: chunks.append(text)
    n = len(chunks)
    for i, chunk in enumerate(chunks):
        if n > 1:
            # reserve prefix room BEFORE slicing judgment: "[99/99]\n" is 8 bytes
            chunk = "[%d/%d]\n%s" % (i+1, n, chunk)
        if len(chunk) > MAX_LEN:          # prefix pushed it over; trim tail
            chunk = chunk[:MAX_LEN]
        for try_md in (True, False):
            obj = {"chat_id": chat_id, "text": chunk}
            if try_md: obj["parse_mode"] = "Markdown"
            payload = ujson.dumps(obj); del obj
            r = None
            ok = False
            try:
                r = urequests.post(url, data=payload, headers={"Content-Type":"application/json"})
                body = r.json(); r.close(); r = None
                ok = bool(body.get("ok")); del body
            except Exception as e:
                print("[TG] send error:", e)
                if r:
                    try: r.close()
                    except Exception: pass
            del payload
            gc.collect()
            if ok: break


def tg_send_action(chat_id):
    r = None
    try:
        r = urequests.post(TG_BASE + "/sendChatAction",
                           data=ujson.dumps({"chat_id": chat_id, "action": "typing"}),
                           headers={"Content-Type": "application/json"})
        r.close(); r = None
    except Exception: pass
    if r:
        try: r.close()
        except Exception: pass
    gc.collect()

# ============================================================================
# AI PROVIDER — one chat-completion call with optional tools + retry.
#
# Returns (status, content, tool_calls):
#   AI_OK                — content may be "", tool_calls may be [] or populated
#   AI_ERROR             — content holds the error message
#   AI_TOOLS_UNSUPPORTED — model/provider rejected the tools parameter;
#                          caller should retry without tools and remember.
# tool_calls entries are the RAW dicts from the response JSON — echoing them
# back verbatim in Call 2 keeps Gemini's signed function calls intact.
# ============================================================================

def ai_chat(provider_key, model, messages, sys_prompt, tools=None):
    prov = PROVIDERS.get(provider_key)
    if not prov:
        return AI_ERROR, "Error: provider '%s' not available." % provider_key, None
    chat_msgs = [{"role":"system","content":sys_prompt}]
    chat_msgs.extend(messages)
    body = {"model":model,"messages":chat_msgs,"temperature":TEMPERATURE,"max_tokens":MAX_TOKENS}
    if tools: body["tools"] = tools
    headers = {"Content-Type":"application/json","Authorization":"Bearer "+prov["key"]}
    payload = ujson.dumps(body); del body, chat_msgs; gc.collect()

    try:
        for attempt in range(_CHAT_MAX_RETRIES + 1):
            r = None
            try:
                r = urequests.post(prov["url"], data=payload, headers=headers)
                data = r.json(); r.close(); r = None; gc.collect()
                choices = data.get("choices")
                if choices and len(choices) > 0:
                    msg = choices[0].get("message", {}) or {}
                    content = msg.get("content") or ""
                    tcs = msg.get("tool_calls") or []
                    del data
                    if len(content) > MAX_RESPONSE_LEN:
                        content = content[:MAX_RESPONSE_LEN] + "\n[truncated]"
                        tcs = []   # truncated content can't pair with tool flow safely
                    return AI_OK, content, tcs
                err = data.get("error"); del data
                if err:
                    emsg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                else:
                    emsg = "(no response from %s)" % prov["name"]
                low = emsg.lower()
                if tools and ("tool" in low or "function" in low):
                    return AI_TOOLS_UNSUPPORTED, emsg, None
                if attempt < _CHAT_MAX_RETRIES and _is_transient(low):
                    if wdt: wdt.feed()
                    wait = 3 * (1 << attempt)
                    print("[Bot] Transient error (%s), retry in %ds" % (emsg, wait))
                    time.sleep(wait)
                    if wdt: wdt.feed()
                    continue
                return AI_ERROR, "API Error: %s" % emsg, None
            except Exception as e:
                if r:
                    try: r.close()
                    except Exception: pass
                gc.collect()
                emsg = str(e)
                if attempt < _CHAT_MAX_RETRIES and _is_transient(emsg):
                    if wdt: wdt.feed()
                    time.sleep(3 * (1 << attempt))
                    if wdt: wdt.feed()
                    continue
                return AI_ERROR, "Error calling %s: %s" % (prov["name"], emsg), None
    finally:
        del payload, headers
        gc.collect()
    return AI_ERROR, "(unreachable)", None


# ============================================================================
# THINKING-TAG STRIPPER — pure string ops, no ure import (~2 KB saved).
# Reasoning models (Qwen / DeepSeek / NVIDIA) leak <think> blocks into
# the visible reply on OpenAI-compat endpoints.
# ============================================================================

def strip_thinking(text):
    for tag in ("think", "thinking", "reasoning"):
        open_t  = "<" + tag + ">"
        close_t = "</" + tag + ">"
        while True:
            low = text.lower()
            i = low.find(open_t)
            if i == -1:
                break
            j = low.find(close_t, i)
            if j == -1:
                text = text[:i]      # unterminated block — drop to end
                break
            text = text[:i] + text[j + len(close_t):]
    return text.strip()


# ============================================================================
# WEB SEARCH ENGINES
# ============================================================================

def _url_encode(s):
    """RFC3986 quote_plus, allocation-lean: one pass over a bytes source,
    appending multi-char fragments instead of one str object per byte."""
    out = []
    add = out.append
    for b in s.encode('utf-8'):
        if (65<=b<=90)or(97<=b<=122)or(48<=b<=57)or b in (45,46,95,126):
            add(chr(b))
        elif b == 32:
            add('+')
        else:
            add('%%%02X' % b)
    return ''.join(out)


def _strip_tags(html_str):
    out, in_tag = [], False
    for c in html_str:
        if c=='<': in_tag=True
        elif c=='>': in_tag=False
        elif not in_tag: out.append(c)
    text = ''.join(out)
    for old,new in (('&amp;','&'),('&lt;','<'),('&gt;','>'),('&quot;','"'),
                    ('&#39;',"'"),('&#x27;',"'"),('&nbsp;',' ')):
        text = text.replace(old,new)
    return text.strip()


def _read_body_limited(r, limit):
    """Read at most `limit` bytes of a response body.

    r.text/.content buffer the ENTIRE body — fatal for a 60-100 KB HTML
    page on a 400 KB chip. Reading the raw socket caps the allocation.
    """
    try:
        data = r.raw.read(limit)
        return data.decode('utf-8', 'ignore') if isinstance(data, bytes) else (data or "")
    except AttributeError:
        return (r.text or "")[:limit]


def brave_search(query):
    if not BRAVE_API_KEY: return []
    r = None
    try:
        r = urequests.get(
            "https://api.search.brave.com/res/v1/web/search?q=%s&count=%d" % (_url_encode(query), MAX_SEARCH_RESULTS),
            headers={"Accept":"application/json","X-Subscription-Token":BRAVE_API_KEY})
        data = r.json(); r.close(); r = None; gc.collect()
        snippets = []
        for item in data.get("web",{}).get("results",[]):
            title = (item.get("title","") or "").strip()
            desc  = (item.get("description","") or "").strip()[:MAX_SNIPPET_LEN]
            if title and len(desc) >= 15:
                snippets.append("%s: %s" % (title, desc))
        del data; gc.collect()
        print("[Search] Brave '%s' -> %d" % (query, len(snippets)))
        return snippets
    except Exception as e:
        print("[Search] Brave error: %s" % e)
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
        return []


def searxng_search(query):
    if not SEARXNG_URL: return []
    r = None
    try:
        r = urequests.get(
            "%s/search?q=%s&format=json&count=%d" % (SEARXNG_URL, _url_encode(query), MAX_SEARCH_RESULTS),
            headers={"Accept":"application/json"})
        data = r.json(); r.close(); r = None; gc.collect()
        snippets = []
        for item in data.get("results", [])[:MAX_SEARCH_RESULTS]:
            title = (item.get("title","") or "").strip()
            content = (item.get("content","") or "").strip()[:MAX_SNIPPET_LEN]
            if title and len(content) >= 15:
                snippets.append("%s: %s" % (title, content))
        del data; gc.collect()
        print("[Search] SearXNG '%s' -> %d" % (query, len(snippets)))
        return snippets
    except Exception as e:
        print("[Search] SearXNG error: %s" % e)
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
        return []


def duckduckgo_search(query):
    """DuckDuckGo via the lite endpoint.

    lite.duckduckgo.com/lite/ returns a small table-based page (~10-25 KB)
    vs 60-100 KB for html.duckduckgo.com/html/. Read is additionally capped
    at 20 KB — the first 3 results live at the top, so truncation is safe.
    """
    r = None
    try:
        r = urequests.get("https://lite.duckduckgo.com/lite/?q=%s" % _url_encode(query),
                          headers={"User-Agent":"Mozilla/5.0"})
        html = _read_body_limited(r, 20480)
        r.close(); r = None; gc.collect()
        snippets, pos = [], 0
        while len(snippets) < MAX_SEARCH_RESULTS:
            pos = html.find("result-link", pos)
            if pos == -1: break
            tag_end = html.find('>', pos)
            if tag_end == -1: break
            title_end = html.find('</a>', tag_end+1)
            if title_end == -1: break
            title = _strip_tags(html[tag_end+1:title_end])
            snip_pos = html.find("result-snippet", title_end)
            next_r   = html.find("result-link", title_end+1)
            desc = ""
            if snip_pos != -1 and (next_r == -1 or snip_pos < next_r):
                se = html.find('>', snip_pos)
                if se != -1:
                    end = html.find('</td>', se+1)
                    if end != -1: desc = _strip_tags(html[se+1:end])[:MAX_SNIPPET_LEN]
            if title and len(desc) >= 15:
                snippets.append("%s: %s" % (title, desc))
            elif title and len(snippets) < MAX_SEARCH_RESULTS and not desc:
                # result with no snippet block at all — keep title-only
                snippets.append(title)
            pos = title_end+1
        del html; gc.collect()
        print("[Search] DDG '%s' -> %d" % (query, len(snippets)))
        return snippets
    except Exception as e:
        print("[Search] DDG error: %s" % e)
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
        return []


def exa_search(query):
    """Exa answer endpoint — a synthesized, ready-to-read answer via HTTPS POST.

    Lighter than scraping for the C3: replaces 'fetch 20 KB HTML + parse' with
    one small JSON POST/response. Citations are dropped at the source so the
    chat model never echoes them into Telegram replies (bot.py contract).
    """
    if not EXA_API_KEY: return []
    r = None
    try:
        body = {"model": "exa", "messages": [{"role": "user", "content": query}]}
        payload = ujson.dumps(body); del body
        r = urequests.post(
            "https://api.exa.ai/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + EXA_API_KEY})
        data = r.json(); r.close(); r = None; del payload; gc.collect()
        choices = data.get("choices") or []
        answer = ""
        if choices:
            answer = ((choices[0].get("message") or {}).get("content") or "").strip()
        del data; gc.collect()
        if not answer:
            print("[Search] Exa '%s' -> empty answer" % query)
            return []
        print("[Search] Exa '%s' -> %d chars" % (query, len(answer)))
        return [answer[:MAX_ANSWER_LEN]]
    except Exception as e:
        print("[Search] Exa error: %s" % e)
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
        return []


def web_search(query, engine):
    """Engine preference with DDG as universal fallback (from bot.py)."""
    if engine == "exa" and EXA_API_KEY:
        res = exa_search(query)
        if res: return res
        print("[Search] Exa empty, falling back to DDG")
    if engine == "brave" and BRAVE_API_KEY:
        res = brave_search(query)
        if res: return res
        print("[Search] Brave empty, falling back to DDG")
    if engine == "searxng" and SEARXNG_URL:
        res = searxng_search(query)
        if res: return res
        print("[Search] SearXNG empty, falling back to DDG")
    return duckduckgo_search(query)


# ============================================================================
# FUNCTION-CALLING SEARCH FLOW (ported from bot.py)
#
#   Call 1 — chat completion with WEB_SEARCH_TOOL attached:
#     - model answers directly        -> done in 1 call
#     - model emits tool_calls        -> execute search, inject result, Call 2
#   Call 2 — follow-up with tool result, tools removed:
#     - model answers using the search data
#
# FALLBACK: model/provider without tool support -> AI_TOOLS_UNSUPPORTED ->
# session flag "no_tools" set, plain single call from then on.
# ============================================================================

_NO_RESULTS_MSG = (
    "The web search for '%s' returned no usable results. "
    "Tell the user plainly that you could not find current "
    "information on this topic, and do NOT answer from memory."
)


def _format_search_results(query, snippets):
    t = time.localtime()
    today = "%04d-%02d-%02d" % (t[0], t[1], t[2])
    parts = ["Today is %s. Web search results for '%s':" % (today, query)]
    for i, snip in enumerate(snippets):
        parts.append("%d. %s" % (i+1, snip))
    parts.append(
        "\nAnswer using ONLY the information above. If it does not answer the "
        "question, say so plainly instead of filling gaps from memory."
    )
    return "\n".join(parts)


def _parse_tool_call(tc):
    """Normalize a raw tool_calls entry -> (name, args_dict, call_id)."""
    fn = tc.get("function") or {}
    name = fn.get("name", "")
    args = fn.get("arguments", "")
    if isinstance(args, str):
        try: args = ujson.loads(args)
        except Exception: args = {}
    if not isinstance(args, dict): args = {}
    return name, args, tc.get("id") or "call_0"


def ai_chat_with_search(s):
    """Full message flow for a session (history already holds the new user msg).

    Returns (AI_OK, final_text) or (AI_ERROR, error_text).
    """
    provider_key = s["provider"]
    model        = s["model"]
    sys_prompt   = _build_system_prompt()
    use_tools    = s.get("web_search", True) and not s.get("no_tools")
    tools        = [WEB_SEARCH_TOOL] if use_tools else None

    # Fold compacted memory in front of any surviving verbatim history — one
    # ~400-byte message, so the chat call sees old context without the bulk.
    req = list(s["history"])
    if s.get("summary"):
        req.insert(0, {"role": "system",
                       "content": "Conversation so far (summary): " + s["summary"]})

    # ── Call 1 ──────────────────────────────────────────────────────────
    status, content, tcs = ai_chat(provider_key, model, req, sys_prompt, tools)

    if status == AI_TOOLS_UNSUPPORTED:
        print("[Bot] %s/%s rejected tools — disabling for this session" % (provider_key, model))
        s["no_tools"] = True
        status, content, tcs = ai_chat(provider_key, model, req, sys_prompt)

    if status != AI_OK:
        return AI_ERROR, content

    # ── Direct answer, no tool call ─────────────────────────────────────
    if not tcs:
        return AI_OK, strip_thinking(content)

    # ── Tool call — execute the first one (web_search is the only tool) ──
    name, args, call_id = _parse_tool_call(tcs[0])
    if name == "web_search":
        query = (args.get("query") or "").strip()
        if query:
            engine = s.get("search_engine", "duckduckgo")
            print("[FuncCall] web_search('%s') via %s" % (query, engine))
            if wdt: wdt.feed()
            snippets = web_search(query, engine)
            tool_result = _format_search_results(query, snippets) if snippets else _NO_RESULTS_MSG % query
            del snippets
        else:
            tool_result = "Error: web_search called with empty query."
    else:
        tool_result = "Unknown tool: %s" % name
    gc.collect()

    # OpenAI spec: append the assistant message WITH the raw tool_calls,
    # then the tool result, then call again without tools.
    follow = list(req)
    follow.append({"role": "assistant",
                   "content": content or None,
                   "tool_calls": [tcs[0]]})
    follow.append({"role": "tool",
                   "tool_call_id": call_id,
                   "content": tool_result})
    del tool_result
    if wdt: wdt.feed()

    # ── Call 2 — final answer from search results ───────────────────────
    status2, content2, _ = ai_chat(provider_key, model, follow, sys_prompt)
    del follow, req; gc.collect()
    if status2 != AI_OK:
        return AI_ERROR, content2
    return AI_OK, strip_thinking(content2)

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def get_session(user_id):
    if user_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            oldest = min(sessions, key=lambda uid: sessions[uid]["last_msg_time"])
            del sessions[oldest]; gc.collect()
        prov = DEFAULT_PROVIDER if DEFAULT_PROVIDER in PROVIDERS else next(iter(PROVIDERS), "")
        if EXA_API_KEY: eng = "exa"
        elif BRAVE_API_KEY: eng = "brave"
        elif SEARXNG_URL: eng = "searxng"
        else: eng = "duckduckgo"
        sessions[user_id] = {
            "provider": prov,
            "model":    PROVIDERS[prov]["default_model"] if prov else "",
            "history":  [],
            "summary":  "",
            "web_search":    True,
            "search_engine": eng,
            "no_tools":      False,
            "last_msg_time": 0,
        }
    return sessions[user_id]

# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def handle_command(chat_id, text, user_id):
    parts = text.strip().split(None, 1)
    cmd   = parts[0].lower()
    arg   = parts[1].strip() if len(parts) > 1 else ""
    if "@" in cmd: cmd = cmd.split("@")[0]
    s = get_session(user_id)

    if cmd == "/start":
        tg_send(chat_id,
            "ESP32-C3 AI Bot v6 (function calling)\n\nProvider: %s\nModel: %s\nAvailable: %s\n\nType /help." %
            (s["provider"], s["model"], ", ".join(PROVIDERS.keys())))
        return True

    if cmd == "/help":
        tg_send(chat_id,
            "*Commands*\n"
            "/provider [name] — switch provider\n"
            "/models — list models\n"
            "/model [id] — switch model\n"
            "/refresh — re-fetch model lists\n"
            "/clear — clear history\n"
            "/web [on|off|exa|brave|searxng|ddg] — search toggle\n"
            "/status — device info\n"
            "/help — this message\n\nJust type to chat!")
        return True

    if cmd == "/provider":
        if not arg:
            tg_send(chat_id, "Current: %s\nAvailable: %s\n\nUse: /provider <name>" %
                    (s["provider"], ", ".join(PROVIDERS.keys())))
            return True
        al = arg.lower()
        if al not in PROVIDERS:
            tg_send(chat_id, "Not found: %s\nAvailable: %s" % (arg, ", ".join(PROVIDERS.keys())))
            return True
        s["provider"] = al
        s["model"]    = PROVIDERS[al]["default_model"]
        s["history"]  = []
        s["summary"]  = ""
        s["no_tools"] = False
        tg_send(chat_id, "Switched to %s\nModel: %s\nHistory cleared." % (PROVIDERS[al]["name"], s["model"]))
        return True

    if cmd == "/models":
        prov   = PROVIDERS.get(s["provider"])
        if not prov: tg_send(chat_id, "No provider selected."); return True
        models = get_models(s["provider"])
        lines  = ["%s models (%s, %d):" % (prov["name"], "API" if prov["can_fetch"] else "hardcoded", len(models))]
        for m in models:
            lines.append("- %s%s" % (m, " [current]" if m == s["model"] else ""))
        lines.append("\nUse: /model <id>")
        tg_send(chat_id, "\n".join(lines))
        return True

    if cmd == "/refresh":
        tg_send(chat_id, "Refreshing...")
        refresh_all_models(fetch=True)
        lines = ["Refreshed:"]
        for k, p in PROVIDERS.items():
            lines.append("- %s: %d (%s)" % (p["name"], len(p["models"]), "API" if p["can_fetch"] else "hardcoded"))
        tg_send(chat_id, "\n".join(lines))
        return True

    if cmd == "/model":
        if not arg:
            tg_send(chat_id, "Current: %s\nUse: /model <id>" % s["model"])
            return True
        models = get_models(s["provider"])
        if not models:
            tg_send(chat_id, "Model list unavailable (fetch failed) — try /refresh first.")
            return True
        s["model"] = arg
        s["no_tools"] = False   # new model may support tools
        if arg in models:
            tg_send(chat_id, "Switched to: %s" % arg)
        else:
            tg_send(chat_id, "Model '%s' not in list — set anyway.\nUse /models to see known." % arg)
        return True

    if cmd == "/clear":
        s["history"] = []; s["summary"] = ""; gc.collect()
        tg_send(chat_id, "History and summary cleared.")
        return True

    if cmd == "/status":
        wlan   = network.WLAN(network.STA_IF)
        uptime = time.time() - boot_time
        rssi   = "N/A"
        try:
            v = wlan.status("rssi")
            rssi = "%d dBm (%s)" % (v, "Strong" if v>=-50 else "Medium" if v>=-70 else "Weak")
        except Exception: pass
        pub_ip = "N/A"; r = None
        try:
            r = urequests.get("http://api.ipify.org"); pub_ip = r.text.strip(); r.close(); r = None
        except Exception:
            if r:
                try: r.close()
                except Exception: pass
        gc.collect()
        fr = gc.mem_free(); ua = gc.mem_alloc(); tot = fr+ua
        eng  = s.get("search_engine","duckduckgo")
        eng_name = {"exa":"Exa","brave":"Brave","searxng":"SearXNG"}.get(eng,"DuckDuckGo")
        tg_send(chat_id,
            "ESP32-C3 Status\n\n"
            "WiFi: %s\nRSSI: %s\nIP: %s\nPublic: %s\n"
            "DuckDNS: %s\nUptime: %dh %dm\nCPU: %d MHz\n"
            "RAM: %d/%d (%d%%)\nProvider: %s\nModel: %s\n"
            "History: %d msgs\nSummary: %s\nWeb search: %s (%s)%s" % (
                WIFI_SSID, rssi,
                wlan.ifconfig()[0] if wlan.isconnected() else "disconnected",
                pub_ip, duckdns_status,
                uptime//3600, (uptime%3600)//60, freq()//1000000,
                ua, tot, (ua*100)//tot if tot else 0,
                s["provider"], s["model"], len(s["history"]),
                ("%d chars" % len(s["summary"])) if s.get("summary") else "none",
                "ON" if s.get("web_search") else "OFF",
                eng_name,
                ", tools unsupported" if s.get("no_tools") else ""))
        return True

    if cmd == "/web":
        if not arg:
            eng = s.get("search_engine","duckduckgo")
            eng_name = {"exa":"Exa","brave":"Brave API","searxng":"SearXNG"}.get(eng,"DuckDuckGo")
            tg_send(chat_id,
                "Web search: %s\nEngine: %s\n\n/web on|off\n/web exa\n/web brave\n/web searxng\n/web ddg" % (
                    "ON" if s.get("web_search") else "OFF", eng_name))
            return True
        a = arg.lower()
        if a=="on":
            s["web_search"]=True
            tg_send(chat_id,"Web search enabled.")
        elif a=="off":
            s["web_search"]=False; tg_send(chat_id,"Web search disabled.")
        elif a=="exa":
            if not EXA_API_KEY: tg_send(chat_id,"EXA_API_KEY not configured.")
            else: s["search_engine"]="exa"; s["web_search"]=True; tg_send(chat_id,"Switched to Exa (synthesized answer, a bit slower).")
        elif a=="brave":
            if not BRAVE_API_KEY: tg_send(chat_id,"Brave API key not configured.")
            else: s["search_engine"]="brave"; s["web_search"]=True; tg_send(chat_id,"Switched to Brave.")
        elif a=="searxng":
            if not SEARXNG_URL: tg_send(chat_id,"SEARXNG_URL not configured.")
            else: s["search_engine"]="searxng"; s["web_search"]=True; tg_send(chat_id,"Switched to SearXNG.")
        elif a in("ddg","duckduckgo"):
            s["search_engine"]="duckduckgo"; s["web_search"]=True; tg_send(chat_id,"Switched to DuckDuckGo.")
        else:
            tg_send(chat_id,"Use: /web on|off|exa|brave|searxng|ddg")
        return True

    tg_send(chat_id, "Unknown: %s\nTry /help" % cmd)
    return True

# ============================================================================
# MESSAGE HANDLER
# ============================================================================

_SUMMARIZER_PROMPT = (
    "Compress this conversation prefix into at most %d characters of durable "
    "facts about the user and the discussion: names, goals, constraints, "
    "decisions made, unresolved questions. Rewrite the existing summary with "
    "the new material merged in — do not append. Output ONLY the summary text, "
    "no preamble."
) % MAX_SUMMARY_CHARS


def _serialize_for_summary(messages):
    """Role-prefixed one-line-per-message rendering, clipped for the summarizer."""
    parts = []
    for m in messages:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "").replace("\n", " ")
        parts.append("%s: %s" % (role, content[:600]))
    return "\n".join(parts)


def _summarize_prefix(session, prefix_messages, provider_key, model):
    """One extra LLM call: fold prefix_messages (+ old summary) into a <=MAX_SUMMARY_CHARS note."""
    old = session.get("summary", "")
    user_payload = _serialize_for_summary(prefix_messages)
    if old:
        user_payload = ("Existing summary:\n%s\n\nNew messages to merge:\n%s"
                        % (old, user_payload))
    msgs = [{"role": "user", "content": user_payload}]
    status, content, _ = ai_chat(provider_key, model, msgs,
                                 _SUMMARIZER_PROMPT, tools=None)
    del msgs, user_payload, prefix_messages; gc.collect()
    if status != AI_OK or not content:
        return old   # keep old summary on failure
    return content.strip()[:MAX_SUMMARY_CHARS]


def _compact_if_needed(session):
    """Called AFTER the user message is appended, BEFORE the chat call.

    If history exceeds the threshold, fold the oldest turns into a rolling
    summary so the chat call receives summary + recent verbatim messages
    instead of losing the oldest context entirely.
    """
    if len(session["history"]) <= COMPACT_THRESHOLD:
        return
    # Tasks a single TLS session must pull off; skip when the heap is tight.
    if gc.mem_free() < COMPACT_MIN_FREE:
        print("[Compact] heap low (%d) — plain trim" % gc.mem_free())
        _trim_history(session["history"])
        return
    cut = len(session["history"]) - COMPACT_KEEP_RECENT
    if cut < 2:
        return   # need at least one user+assistant pair to be worth summarizing
    prefix = session["history"][:cut]
    session["history"] = session["history"][cut:]
    if wdt: wdt.feed()
    print("[Compact] folding %d msgs into summary" % cut)
    session["summary"] = _summarize_prefix(
        session, prefix, session["provider"], session["model"])
    del prefix; gc.collect()
    # Paranoia belt: if anything above left history over-threshold, hard-trim.
    _trim_history(session["history"])


def _trim_history(history):
    while len(history) > MAX_HISTORY:
        if len(history) >= 2:
            history.pop(0); history.pop(0)
        else:
            history.pop(0)
    return history


def _clip(text):
    return text if len(text) <= HISTORY_MSG_MAX else text[:HISTORY_MSG_MAX]


def handle_message(chat_id, text, user_id):
    s = get_session(user_id)

    now = time.time()
    if now - s["last_msg_time"] < RATE_LIMIT_SECS:
        print("[Bot] Rate limit for user %d" % user_id)
        return
    s["last_msg_time"] = now

    # Emergency memory cleanup — evict another user's session if critically low.
    # Floor is 40 KB: a single TLS handshake already needs ~30-40 KB.
    if gc.mem_free() < MEM_FLOOR:
        gc.collect()
        if gc.mem_free() < MEM_FLOOR:
            for uid in list(sessions):
                if uid != user_id:
                    del sessions[uid]
                    gc.collect()
                    print("[MEM] Freed session for %d, %d bytes free" % (uid, gc.mem_free()))
                    break

    s["history"].append({"role":"user","content":_clip(text)})

    # Fold oldest turns into the rolling summary BEFORE the reply call —
    # a plain 6-message window would otherwise silently forget them.
    _compact_if_needed(s)

    tg_send_action(chat_id)
    if wdt: wdt.feed()

    result_type, response = ai_chat_with_search(s)

    # ── Empty response fallback ──────────────────────────────────────────
    if result_type == AI_OK and not response.strip():
        response = "The AI returned an empty response. Try again or /model to switch."

    # ── Update history or roll back on error ─────────────────────────────
    if result_type == AI_OK:
        s["history"].append({"role":"assistant","content":_clip(response)})
        _trim_history(s["history"])
    else:
        if s["history"] and s["history"][-1].get("role") == "user":
            s["history"].pop()
        print("[Bot] AI error — user message rolled back")

    tg_send_action(chat_id)
    tg_send(chat_id, response)
    del response; gc.collect()

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    global boot_time, tg_offset, last_duckdns

    boot_time = time.time()
    print("=" * 40)
    print("ESP32-C3 Telegram Bot v6 starting...")
    print("Providers:", ", ".join(PROVIDERS.keys()) if PROVIDERS else "NONE!")
    print("=" * 40)

    if not PROVIDERS:
        print("ERROR: No API keys configured!"); return

    global wdt
    wdt = WDT(timeout=120000)
    print("[WDT] Watchdog enabled (120s)")

    wlan = wifi_connect()
    sync_ntp()
    last_duckdns = 0
    update_duckdns()

    # Seed fallback model lists only — no HTTPS fetch at boot (RAM spike).
    refresh_all_models(fetch=False)

    # Skip any messages queued while the bot was down/rebooting.
    tg_flush_updates()

    print("[Bot] Polling... free heap: %d" % gc.mem_free())

    while True:
        wdt.feed()
        if not wlan.isconnected():
            _log("[WiFi] Lost, reconnecting...")
            wlan = wifi_connect(); wdt.feed(); sync_ntp()
            last_duckdns = 0; update_duckdns()
        update_duckdns()
        try:
            updates = tg_get_updates(tg_offset)
        except Exception as e:
            print("[Bot] Poll error:", e); time.sleep(5); gc.collect(); continue
        for upd in updates:
            if wdt: wdt.feed()
            tg_offset = upd.get("update_id", 0) + 1
            msg = upd.get("message")
            if not msg: continue
            chat_id = msg.get("chat",{}).get("id")
            text    = msg.get("text","")
            user_id = msg.get("from",{}).get("id", 0)
            if not chat_id or not text: continue
            if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
                tg_send(chat_id, "Not authorized."); continue
            try:
                if text.startswith("/"):
                    handle_command(chat_id, text, user_id)
                else:
                    handle_message(chat_id, text, user_id)
            except Exception as e:
                print("[Bot] Handler error:", e)
                tg_send(chat_id, "Error: %s" % str(e))
            gc.collect()
        del updates; gc.collect()


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
