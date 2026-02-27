"""
ESP32-C3 Super Mini — MicroPython Telegram Bot
================================================
Multi-provider AI chat bot (Groq, Gemini, OpenRouter, Cerebras, NVIDIA)
with DuckDNS updater and WiFi retry loop.

Hardware: ESP32-C3 Super Mini (400 KB SRAM, 4 MB flash)
Runtime:  MicroPython 1.20+

ZERO DISK WRITES — all state is in-memory, reboot resets everything.

CHANGES v2 → v3 (3-tier search restored + parser hardened):
  - 'maybe' tier RESTORED: ambiguous queries ask AI with search-decision prompt.
    Binary heuristic missed: "is X better than Y?", "recommend a laptop",
    "what happened with OpenAI?" — no keyword signal but clearly need search.
    Cost: 1 extra API call only on genuinely ambiguous queries.
  - _parse_search_query() hardened against inline apologies (same root-cause
    bug that hit bot.py — model returns 'SEARCH: queryI'm sorry...' on one line):
      - Scans for 12 apology/refusal keywords, truncates at first match
      - Truncates at first sentence-ending punctuation . ! ?
      - Requires min 2 chars — short result means parsing failed, falls back
      - Pure string ops only, no ure/re import needed (saves ~2KB RAM)
  - _PROMPT_SEARCH_DECISION_TEMPLATE tightened with explicit CRITICAL: rule
    to prevent models from appending text after SEARCH: in the first place
  - WDT fed before every blocking call in all three tiers
  - Typing indicator sent before every blocking operation
  - Empty AI response caught with friendly fallback message
  - History rollback on any error
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

WIFI_SSID        = "YOUR_WIFI_SSID"
WIFI_PASSWORD    = "YOUR_WIFI_PASSWORD"

TELEGRAM_TOKEN   = "YOUR_TELEGRAM_BOT_TOKEN"
ALLOWED_USER_IDS = []   # e.g. [123456789] — empty = allow all

DUCKDNS_TOKEN    = "YOUR_DUCKDNS_TOKEN"
DUCKDNS_DOMAIN   = "YOUR_SUBDOMAIN"
DUCKDNS_INTERVAL = 300

GROQ_API_KEY       = ""
GEMINI_API_KEY     = ""
OPENROUTER_API_KEY = ""
CEREBRAS_API_KEY   = ""
NVIDIA_API_KEY     = ""

BRAVE_API_KEY      = ""
MAX_SEARCH_RESULTS = 3
MAX_SNIPPET_LEN    = 200

DEFAULT_PROVIDER = "groq"
MAX_TOKENS       = 512
TEMPERATURE      = 0.7
MAX_HISTORY      = 6
MAX_SESSIONS     = 3
MAX_RESPONSE_LEN = 4000
RATE_LIMIT_SECS  = 2

# ============================================================================
# SYSTEM PROMPTS
#
# SYSTEM_PROMPT              — normal AI calls
# SYSTEM_PROMPT_WITH_RESULTS — when search results are passed in
# _make_search_decision_prompt() — builds the 'maybe' decision prompt with
#                                  today's date injected at call time
# ============================================================================

_PROMPT_BASE = (
    "You are a helpful, concise AI assistant running on an ESP32 microcontroller "
    "and speaking through Telegram.\n\n"
    "FORMATTING RULES:\n"
    "- Telegram Markdown only: *bold*, _italic_, `code`, ```blocks```\n"
    "- Never use ## headers — they do not render in Telegram\n"
    "- Never use LaTeX. Flat bullet lists only (- item)\n"
    "- Keep responses concise. No long preambles or sign-offs\n"
    "- For comparisons: use *bold item name* on its own line, then bullet points for attributes. "
    "Never use | pipe | tables — they do not render in Telegram\n\n"
    "BEHAVIOUR:\n"
    "- Be direct and practical\n"
    "- Admit uncertainty rather than guessing\n"
    "- Use fenced code blocks with language name for all code"
)

_PROMPT_SEARCH_DECISION_TEMPLATE = (
    "\n\nSEARCH DECISION:\n"
    "You can search the web before answering. Today's date is {date}.\n\n"
    "RULE: Your entire response must be EITHER:\n"
    "  A) Exactly one line: SEARCH: <2-6 word query>\n"
    "     Use when the question needs current/real-time data: prices, news, "
    "weather, scores, recent events, product releases, living people's status.\n"
    "  B) A direct answer to the question.\n"
    "     Use for timeless facts, code, math, definitions, creative writing.\n\n"
    "CRITICAL: If you output SEARCH:, do NOT add any other text on that line "
    "or after it. No apologies. No explanations. Just: SEARCH: <keywords>\n\n"
    "Query: 2-6 words, no punctuation. Example: SEARCH: bitcoin price today"
)

_PROMPT_SEARCH_RESULTS = (
    "\n\nSEARCH RESULTS FORMAT:\n"
    "Results appear as numbered snippets (1. Title: text). "
    "Use them to answer accurately. Refer to sources naturally "
    "(e.g. 'according to X'). Do NOT write citation numbers like [1][2]."
)

SYSTEM_PROMPT              = _PROMPT_BASE
SYSTEM_PROMPT_WITH_RESULTS = _PROMPT_BASE + _PROMPT_SEARCH_RESULTS


def _make_search_decision_prompt():
    t = time.localtime()
    today = "%04d-%02d-%02d" % (t[0], t[1], t[2])
    return _PROMPT_BASE + _PROMPT_SEARCH_DECISION_TEMPLATE.replace("{date}", today)


# ============================================================================
# SEARCH TIER CLASSIFICATION
#
# 'no'     — coding, creative, math, greetings → 1 AI call
# 'direct' — strong time + topic signal → extract query, search, 1 AI call
# 'maybe'  — ambiguous → ask AI with search-decision prompt
#             AI replies SEARCH: <query> → search → 1 final AI call
#             AI answers directly        → use that answer, done
# ============================================================================

_KW_TIME = (
    'today','tonight','now','latest','current','recent',
    'yesterday','tomorrow','this week','this month','this year',
    'last week','last month','last year','next week','next month',
)
_KW_TOPIC = (
    'price','stock','crypto','bitcoin','btc','eth','market','trading',
    'weather','forecast','temperature',
    'news','headline','election','vote','poll',
    'score','game','match','tournament','playoff','standings',
    'release','released','launched','announced','update',
    'dead','died','alive','arrested','fired','hired','resigned',
    'war','earthquake','hurricane','disaster','outbreak',
)
_KW_NO_SEARCH = (
    'write','code','create','generate','build','make','implement',
    'explain','define','describe','summarize','rewrite','translate',
    'fix','debug','refactor','optimize','calculate','solve','convert',
    'hello','hi ','hey ','thanks','thank you','bye','good morning',
    'tell me a joke','poem','story','essay','list',
)
_KW_COMPARE = (
    ' vs ',' versus ','compare','better than','best ',
    'difference between','recommend','should i use',
    'which is better','which one','alternative to',
)


def _search_tier(text):
    low    = text.lower()
    length = len(low)

    if length < 12:
        return 'no'

    has_no = False
    for kw in _KW_NO_SEARCH:
        if kw in low:
            has_no = True
            break

    has_time = False
    for kw in _KW_TIME:
        if kw in low:
            has_time = True
            break

    has_topic = False
    for kw in _KW_TOPIC:
        if kw in low:
            has_topic = True
            break

    if has_time and has_topic:
        return 'direct'

    if has_no and not has_time and not has_topic:
        return 'no'

    if has_time or has_topic:
        return 'maybe'

    for kw in _KW_COMPARE:
        if kw in low:
            return 'maybe'

    t    = time.localtime()
    year = t[0]
    for y in (str(year - 1), str(year), str(year + 1)):
        if y in low:
            return 'maybe'

    if '?' in low and length > 20:
        for kw in ('who ', 'where ', 'when ', 'how much', 'how many', 'what is the'):
            if kw in low:
                return 'maybe'

    if has_no:
        return 'no'

    if '?' in low and length > 30:
        return 'maybe'

    return 'no'


def _extract_search_query(text):
    low = text.strip()
    for prefix in (
        'what is the','what are the','what is','what are',
        'how much is','how much does','how much',
        'who is the','who is','who won the','who won',
        'where is the','where is',
        'when is the','when is','when does',
        'tell me about','search for','look up','find',
    ):
        if low.lower().startswith(prefix):
            low = low[len(prefix):].strip()
            break
    low = low.rstrip('?').strip()
    return low[:120] if low else text[:120]


def _parse_search_query(response):
    """Extract SEARCH: query from AI decision response.
    Returns clean query string or None if AI answered directly.

    Handles all known model quirks — pure string ops, no ure/re import:
    - Apology glued inline (no newline):
        'SEARCH: current US presidentI am sorry, I cannot...'
        Root cause: small models ignore 'ONLY this line' instruction.
        Fix: scan for apology keywords, truncate at first match.
    - Period-terminated with explanation:
        'SEARCH: who is president. I cannot answer that.'
        Fix: truncate at first . ! ?
    - Answer appended after newline:
        'SEARCH: bitcoin price\n\nBitcoin is currently...'
        Fix: split('\n')[0]
    - Quoted query:     SEARCH: \"bitcoin price\"
    - Repeated prefix:  SEARCH: SEARCH: something
    - Mixed case:       search: / Search:
    """
    stripped = response.strip()
    if not stripped.upper().startswith('SEARCH:'):
        return None

    raw = stripped[7:].strip()

    # Take only the first line
    nl = raw.find('\n')
    if nl != -1:
        raw = raw[:nl].strip()

    # Strip surrounding quotes
    if len(raw) > 2 and raw[0] in ('"', "'") and raw[-1] == raw[0]:
        raw = raw[1:-1].strip()

    # Strip repeated SEARCH: prefix (rare but happens)
    for _ in range(3):
        if raw.upper().startswith('SEARCH:'):
            raw = raw[7:].strip()
        else:
            break

    # Cut at inline apology/refusal — models that ignore 'CRITICAL:' instruction
    # still sometimes glue apology text directly after the query without a newline.
    # Using pure str.find() — no ure import needed, saves ~2KB RAM on ESP32.
    _CUTS = (
        "i'm sorry", "i am sorry", "i cannot", "i don't", "i do not",
        "i am unable", "i can't", "please note", "note that",
        "however,", "unfortunately", "as of my",
    )
    raw_lower = raw.lower()
    cut_at = len(raw)
    for pat in _CUTS:
        idx = raw_lower.find(pat)
        if idx != -1 and idx < cut_at:
            cut_at = idx
    raw = raw[:cut_at].strip()

    # Cut at sentence boundary — catches 'who is president. I cannot answer.'
    # Find the earliest . ! ? and truncate there
    for punct in ('.', '!', '?'):
        idx = raw.find(punct)
        if idx != -1:
            raw = raw[:idx].strip()
            break

    # Strip trailing punctuation artifacts
    raw = raw.rstrip('.,!?;:').strip()

    raw = raw[:120]
    # Require at least 2 chars — 1-char result means parsing failed, fall back
    return raw if len(raw) >= 2 else None


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
        "fallback": ["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768","gemma2-9b-it"],
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
        "fallback": ["gemini-2.0-flash","gemini-2.0-flash-lite","gemini-1.5-flash","gemini-1.5-pro"],
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
        "fallback": ["llama-3.3-70b","llama3.1-8b"],
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

AI_OK    = 0
AI_ERROR = 1

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
    print("[WiFi] Connecting to", WIFI_SSID)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    delay = 2
    while not wlan.isconnected():
        if wdt: wdt.feed()
        print("[WiFi] Waiting %ds..." % delay)
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
            ntptime.host = server; ntptime.settime()
            t = time.localtime()
            print("[NTP] %s: %04d-%02d-%02d %02d:%02d" % (server,t[0],t[1],t[2],t[3],t[4]))
            return
        except Exception as e:
            print("[NTP] %s failed: %s" % (server, e))
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
        result = r.text.strip(); r.close(); r = None
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
# TELEGRAM
# ============================================================================

TG_BASE = "https://api.telegram.org/bot" + TELEGRAM_TOKEN


def tg_get_updates(offset):
    r = None
    try:
        r = urequests.get(TG_BASE + "/getUpdates?timeout=30&offset=%d" % offset)
        data = r.json(); r.close(); r = None; gc.collect()
        if data.get("ok"):
            result = data.get("result", []); del data; return result
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
        if split_at < 1: split_at = split_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text: chunks.append(text)
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            chunk = "[%d/%d]\n%s" % (i+1, len(chunks), chunk)
        for try_md in (True, False):
            obj = {"chat_id": chat_id, "text": chunk}
            if try_md: obj["parse_mode"] = "Markdown"
            payload = ujson.dumps(obj); del obj
            r = None
            try:
                r = urequests.post(url, data=payload, headers={"Content-Type":"application/json"})
                body = r.json(); r.close(); r = None
                ok = bool(body.get("ok")); del body, payload; gc.collect()
                if ok: break
            except Exception as e:
                print("[TG] send error:", e)
                if r:
                    try: r.close()
                    except Exception: pass
            del payload; gc.collect()


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
# AI PROVIDER
# ============================================================================

def ai_chat(provider_key, model, messages, sys_prompt=None):
    prov = PROVIDERS.get(provider_key)
    if not prov:
        return AI_ERROR, "Error: provider '%s' not available." % provider_key
    prompt = sys_prompt if sys_prompt is not None else SYSTEM_PROMPT
    chat_msgs = [{"role":"system","content":prompt}]
    chat_msgs.extend(messages)
    body = {"model":model,"messages":chat_msgs,"temperature":TEMPERATURE,"max_tokens":MAX_TOKENS}
    headers = {"Content-Type":"application/json","Authorization":"Bearer "+prov["key"]}
    payload = ujson.dumps(body); del body, chat_msgs; gc.collect()
    r = None
    try:
        r = urequests.post(prov["url"], data=payload, headers=headers)
        del payload
        data = r.json(); r.close(); r = None; gc.collect()
        choices = data.get("choices")
        if choices and len(choices) > 0:
            content = choices[0].get("message",{}).get("content","(empty)")
            del data
            if len(content) > MAX_RESPONSE_LEN:
                content = content[:MAX_RESPONSE_LEN] + "\n[truncated]"
            return AI_OK, content
        err = data.get("error"); del data
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
        if (65<=b<=90)or(97<=b<=122)or(48<=b<=57)or b in(45,46,95,126): out.append(chr(b))
        elif b==32: out.append('+')
        else: out.append('%%%02X'%b)
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


def brave_search(query):
    if not BRAVE_API_KEY: return []
    r = None
    try:
        r = urequests.get(
            "https://api.search.brave.com/res/v1/web/search?q=%s&count=%d" % (_url_encode(query), MAX_SEARCH_RESULTS),
            headers={"Accept":"application/json","X-Subscription-Token":BRAVE_API_KEY})
        data = r.json(); r.close(); r = None; gc.collect()
        results = data.get("web",{}).get("results",[])
        snippets = [(item.get("title",""), item.get("description","")[:MAX_SNIPPET_LEN])
                    for item in results if item.get("title") or item.get("description")]
        del data, results; gc.collect()
        print("[Search] Brave '%s' -> %d" % (query, len(snippets)))
        return snippets
    except Exception as e:
        print("[Search] Brave error: %s" % e)
        if r:
            try: r.close()
            except Exception: pass
        gc.collect()
        return []


def duckduckgo_search(query):
    r = None
    try:
        r = urequests.get("https://html.duckduckgo.com/html/?q=%s" % _url_encode(query),
                          headers={"User-Agent":"Mozilla/5.0"})
        html = r.text; r.close(); r = None; gc.collect()
        snippets, pos = [], 0
        while len(snippets) < MAX_SEARCH_RESULTS:
            pos = html.find('class="result__a"', pos)
            if pos == -1: break
            tag_end = html.find('>', pos)
            if tag_end == -1: break
            title_end = html.find('</a>', tag_end+1)
            if title_end == -1: break
            title = _strip_tags(html[tag_end+1:title_end])
            snip_pos = html.find('class="result__snippet', pos)
            next_r   = html.find('class="result__a"', title_end+1)
            desc = ""
            if snip_pos != -1 and (next_r == -1 or snip_pos < next_r):
                se = html.find('>', snip_pos)
                if se != -1:
                    end = html.find('</a>', se+1)
                    if end == -1: end = html.find('</td>', se+1)
                    if end != -1: desc = _strip_tags(html[se+1:end])[:MAX_SNIPPET_LEN]
            if title: snippets.append((title, desc))
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


def web_search(query, engine):
    if engine == "brave" and BRAVE_API_KEY:
        return brave_search(query)
    return duckduckgo_search(query)

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def get_session(user_id):
    if user_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            oldest = next(iter(sessions))
            del sessions[oldest]; gc.collect()
        prov = DEFAULT_PROVIDER if DEFAULT_PROVIDER in PROVIDERS else next(iter(PROVIDERS), "")
        eng  = "brave" if BRAVE_API_KEY else "duckduckgo"
        sessions[user_id] = {
            "provider": prov,
            "model":    PROVIDERS[prov]["default_model"] if prov else "",
            "history":  [],
            "web_search":    True,
            "search_engine": eng,
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
            "ESP32-C3 AI Bot v3\n\nProvider: %s\nModel: %s\nAvailable: %s\n\nType /help." %
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
            "/web [on|off|brave|ddg] — search toggle\n"
            "/status — device info\n"
            "/help — this message\n\nJust type to chat!")
        return True

    if cmd == "/provider":
        if not arg:
            tg_send(chat_id, "Current: %s\nAvailable: %s\n\nUse: /provider <n>" %
                    (s["provider"], ", ".join(PROVIDERS.keys())))
            return True
        al = arg.lower()
        if al not in PROVIDERS:
            tg_send(chat_id, "Not found: %s\nAvailable: %s" % (arg, ", ".join(PROVIDERS.keys())))
            return True
        s["provider"] = al
        s["model"]    = PROVIDERS[al]["default_model"]
        s["history"]  = []
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
        refresh_all_models()
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
        s["model"] = arg
        if arg in models:
            tg_send(chat_id, "Switched to: %s" % arg)
        else:
            tg_send(chat_id, "Model '%s' not in list — set anyway.\nUse /models to see known." % arg)
        return True

    if cmd == "/clear":
        s["history"] = []; gc.collect()
        tg_send(chat_id, "History cleared.")
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
        tg_send(chat_id,
            "ESP32-C3 Status\n\n"
            "WiFi: %s\nRSSI: %s\nIP: %s\nPublic: %s\n"
            "DuckDNS: %s\nUptime: %dh %dm\nCPU: %d MHz\n"
            "RAM: %d/%d (%d%%)\nProvider: %s\nModel: %s\n"
            "History: %d msgs\nWeb search: %s (%s)" % (
                WIFI_SSID, rssi,
                wlan.ifconfig()[0] if wlan.isconnected() else "disconnected",
                pub_ip, duckdns_status,
                uptime//3600, (uptime%3600)//60, freq()//1000000,
                ua, tot, (ua*100)//tot if tot else 0,
                s["provider"], s["model"], len(s["history"]),
                "ON" if s.get("web_search") else "OFF",
                "Brave" if s.get("search_engine")=="brave" else "DuckDuckGo"))
        return True

    if cmd == "/web":
        if not arg:
            eng = s.get("search_engine","duckduckgo")
            tg_send(chat_id,
                "Web search: %s\nEngine: %s\n\n/web on|off\n/web brave\n/web ddg" % (
                    "ON" if s.get("web_search") else "OFF",
                    "Brave API" if eng=="brave" else "DuckDuckGo"))
            return True
        a = arg.lower()
        if a=="on":
            s["web_search"]=True
            tg_send(chat_id,"Web search enabled (%s)." % ("Brave" if s.get("search_engine")=="brave" else "DDG"))
        elif a=="off":
            s["web_search"]=False; tg_send(chat_id,"Web search disabled.")
        elif a=="brave":
            if not BRAVE_API_KEY: tg_send(chat_id,"Brave API key not configured. Use /web ddg.")
            else: s["search_engine"]="brave"; s["web_search"]=True; tg_send(chat_id,"Switched to Brave.")
        elif a in("ddg","duckduckgo"):
            s["search_engine"]="duckduckgo"; s["web_search"]=True; tg_send(chat_id,"Switched to DuckDuckGo.")
        else:
            tg_send(chat_id,"Use: /web on|off|brave|ddg")
        return True

    tg_send(chat_id, "Unknown: %s\nTry /help" % cmd)
    return True

# ============================================================================
# MESSAGE HANDLER — 3-tier search loop
#
# tier='no':
#   user msg → [1 AI call] → reply
#
# tier='direct':
#   user msg → _extract_search_query → web_search → [1 AI call with results] → reply
#
# tier='maybe':
#   user msg → [1 AI call with search-decision prompt]
#     → AI replies "SEARCH: <query>" → web_search → [1 AI call with results] → reply
#     → AI answers directly          → use that answer → reply
# ============================================================================

def _build_search_context(query, results):
    t = time.localtime()
    today = "%04d-%02d-%02d" % (t[0], t[1], t[2])
    parts = ["Today is %s. Web search results for '%s':" % (today, query)]
    for i, (title, desc) in enumerate(results):
        line = "%d. %s" % (i+1, title)
        if desc: line += ": %s" % desc
        parts.append(line)
    return '\n'.join(parts)


def _trim_history(history):
    while len(history) > MAX_HISTORY:
        if len(history) >= 2:
            history.pop(0); history.pop(0)
        else:
            history.pop(0)
    return history


def handle_message(chat_id, text, user_id):
    s = get_session(user_id)

    now = time.time()
    if now - s["last_msg_time"] < RATE_LIMIT_SECS:
        print("[Bot] Rate limit for user %d" % user_id)
        return
    s["last_msg_time"] = now

    if len(text) > MAX_RESPONSE_LEN:
        text = text[:MAX_RESPONSE_LEN]

    s["history"].append({"role":"user","content":text})

    tg_send_action(chat_id)
    if wdt: wdt.feed()

    web_on = s.get("web_search", True)
    tier   = _search_tier(text) if web_on else 'no'
    print("[Bot] tier=%s len=%d provider=%s" % (tier, len(text), s["provider"]))

    result_type = AI_OK
    response    = ""

    # ── TIER: NO SEARCH ──────────────────────────────────────────────────
    if tier == 'no':
        result_type, response = ai_chat(s["provider"], s["model"], s["history"])

    # ── TIER: DIRECT SEARCH ──────────────────────────────────────────────
    elif tier == 'direct':
        query  = _extract_search_query(text)
        engine = s.get("search_engine","duckduckgo")
        print("[Bot] Direct search (%s): '%s'" % (engine, query))
        tg_send_action(chat_id)
        if wdt: wdt.feed()
        results = web_search(query, engine)
        if results:
            ctx = _build_search_context(query, results); del results; gc.collect()
            search_msgs = list(s["history"][:-1])
            search_msgs.append({"role":"user","content":text+"\n\n"+ctx})
            tg_send_action(chat_id)
            if wdt: wdt.feed()
            result_type, response = ai_chat(
                s["provider"], s["model"], search_msgs, SYSTEM_PROMPT_WITH_RESULTS)
            del search_msgs, ctx; gc.collect()
        else:
            del results; gc.collect()
            result_type, response = ai_chat(s["provider"], s["model"], s["history"])

    # ── TIER: MAYBE — ask AI to decide ───────────────────────────────────
    else:  # 'maybe'
        decision_prompt = _make_search_decision_prompt()
        decision_msgs   = list(s["history"])
        tg_send_action(chat_id)
        if wdt: wdt.feed()
        d_type, ai_decision = ai_chat(s["provider"], s["model"], decision_msgs, decision_prompt)
        del decision_msgs; gc.collect()

        if d_type == AI_ERROR:
            # Decision call failed — fall back to plain AI
            result_type, response = ai_chat(s["provider"], s["model"], s["history"])
        else:
            search_query = _parse_search_query(ai_decision)
            if search_query:
                engine = s.get("search_engine","duckduckgo")
                print("[Bot] AI search (%s): '%s'" % (engine, search_query))
                tg_send_action(chat_id)
                if wdt: wdt.feed()
                results = web_search(search_query, engine)
                if results:
                    ctx = _build_search_context(search_query, results); del results; gc.collect()
                    search_msgs = list(s["history"][:-1])
                    search_msgs.append({"role":"user","content":text+"\n\n"+ctx})
                    tg_send_action(chat_id)
                    if wdt: wdt.feed()
                    result_type, response = ai_chat(
                        s["provider"], s["model"], search_msgs, SYSTEM_PROMPT_WITH_RESULTS)
                    del search_msgs, ctx; gc.collect()
                else:
                    del results; gc.collect()
                    result_type, response = ai_chat(s["provider"], s["model"], s["history"])
            else:
                # AI answered directly — use it
                result_type, response = AI_OK, ai_decision

        del decision_prompt; gc.collect()

    # ── Empty response fallback ──────────────────────────────────────────
    if result_type == AI_OK and not response.strip():
        response = "The AI returned an empty response. Try again or /model to switch."

    # ── Update history or roll back on error ─────────────────────────────
    if result_type == AI_OK:
        s["history"].append({"role":"assistant","content":response})
        _trim_history(s["history"])
    else:
        if s["history"] and s["history"][-1].get("role") == "user":
            s["history"].pop()
        print("[Bot] AI error — user message rolled back")

    tg_send(chat_id, response)
    del response; gc.collect()

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    global boot_time, tg_offset, last_duckdns

    boot_time = time.time()
    print("=" * 40)
    print("ESP32-C3 Telegram Bot v3 starting...")
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

    print("[Bot] Fetching model lists...")
    refresh_all_models()
    print("[Bot] Starting Telegram polling...")

    while True:
        wdt.feed()
        if not wlan.isconnected():
            print("[WiFi] Lost, reconnecting...")
            wlan = wifi_connect(); sync_ntp()
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
