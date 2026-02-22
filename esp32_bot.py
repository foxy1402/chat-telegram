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

# Bot behavior
DEFAULT_PROVIDER = "groq"
MAX_TOKENS = 512
TEMPERATURE = 0.7
MAX_HISTORY = 5  # Keep low for RAM — 5 messages ~2KB
MAX_SESSIONS = 3  # Limit concurrent user sessions to save RAM
MAX_RESPONSE_LEN = 4000  # Truncate AI responses beyond this to protect RAM
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise. No markdown tables."

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
            "openai/gpt-oss-120b",
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
        try:
            r = urequests.post(url, data=payload,
                               headers={"Content-Type": "application/json"})
            r.close()
        except Exception as e:
            print("[TG] send error:", e)
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

def ai_chat(provider_key, model, messages):
    """Send chat to any OpenAI-compatible provider. Returns response text."""
    prov = PROVIDERS.get(provider_key)
    if not prov:
        return "Error: provider '%s' not available." % provider_key

    # Build messages with system prompt
    chat_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
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
        sessions[user_id] = {
            "provider": prov,
            "model": PROVIDERS[prov]["default_model"] if prov else "",
            "history": [],
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
        tg_send(chat_id,
                "Commands:\n"
                "/provider [name] - switch provider\n"
                "/models - list models (auto-fetched!)\n"
                "/model [id] - switch model\n"
                "/refresh - re-fetch model lists from APIs\n"
                "/clear - clear history\n"
                "/status - device info\n"
                "/help - this message\n\n"
                "Just type a message to chat!")
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

    # Unknown command
    tg_send(chat_id, "Unknown command: %s\nTry /help" % cmd)
    return True

# ============================================================================
# MESSAGE HANDLER
# ============================================================================

def handle_message(chat_id, text, user_id):
    """Handle regular text messages — forward to AI provider."""
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

    # Call AI
    response = ai_chat(s["provider"], s["model"], s["history"])

    # Add response to history
    s["history"].append({"role": "assistant", "content": response})

    # Trim again after adding response
    if len(s["history"]) > MAX_HISTORY:
        s["history"] = s["history"][-MAX_HISTORY:]

    # Send to user (use history ref to avoid double-holding the string)
    tg_send(chat_id, s["history"][-1]["content"])
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
