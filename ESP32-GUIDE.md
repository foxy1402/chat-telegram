# ESP32-C3 Super Mini — AI Telegram Bot Setup Guide

Complete guide to flash MicroPython, configure, and deploy `esp32_bot.py` on an **ESP32-C3 Super Mini** board.

---

## What You Need

| Item | Details |
|---|---|
| **Board** | ESP32-C3 Super Mini (4 MB flash, 400 KB SRAM) |
| **USB Cable** | USB-C data cable (not charge-only!) |
| **Software** | [Thonny IDE](https://thonny.org/) (free, easiest way) |
| **Firmware** | MicroPython `.bin` for ESP32-C3 |
| **API Keys** | At least 1 AI provider + Telegram bot token |

---

## Step 1 — Install Thonny IDE

1. Download from **https://thonny.org**
2. Install normally (Windows/Mac/Linux)
3. Open Thonny

---

## Step 2 — Download MicroPython Firmware

1. Go to **https://micropython.org/download/ESP32_GENERIC_C3/**
2. Under **Releases**, download the latest **stable** `.bin` file
   - Example: `ESP32_GENERIC_C3-20251209-v1.27.0.bin`
3. Save it somewhere easy to find (e.g., Desktop)

---

## Step 3 — Connect Board & Install Firmware

### 3.1 — Plug in the ESP32-C3

1. Connect the ESP32-C3 Super Mini via USB-C to your computer
2. **Enter bootloader mode** (required for first flash):
   - Hold the **BOOT** button on the board
   - While holding BOOT, press and release the **RESET** button
   - Release the BOOT button
   - The board is now in download mode (no LED indication is normal)

### 3.2 — Flash MicroPython via Thonny

1. In Thonny, go to **Tools → Options → Interpreter**
2. Set interpreter to: **MicroPython (ESP32)**
3. Set port to: the COM port your board is on
   - Windows: `COM3`, `COM4`, etc.
   - Mac/Linux: `/dev/ttyUSB0` or `/dev/ttyACM0`
4. Click **Install or update MicroPython (esptool)**
5. In the dialog:
   - **Target port**: select your board's port
   - **MicroPython family**: ESP32-C3
   - **MicroPython variant**: Espressif • ESP32-C3
   - Or click **Select a file** and browse to your downloaded `.bin`
   - ✅ Check **Erase flash before installing** (recommended for first time)
6. Click **Install** — wait 1-2 minutes
7. When done, press the **RESET** button on the board
8. You should see `>>>` in Thonny's Shell — MicroPython is running! 🎉

> **Tip**: If Thonny can't find the port, try a different USB cable. Cheap cables are often charge-only.

---

## Step 4 — Configure the Script

Open `esp32_bot.py` in any text editor and fill in the configuration section at the top:

### 4.1 — WiFi (Required)

```python
WIFI_SSID = "MyWiFiNetwork"
WIFI_PASSWORD = "MyPassword123"
```

### 4.2 — Telegram Bot Token (Required)

1. Open Telegram, search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the token BotFather gives you

```python
TELEGRAM_TOKEN = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
```

### 4.3 — Restrict Access (Optional but Recommended)

Lock the bot to specific Telegram user IDs:

```python
ALLOWED_USER_IDS = [123456789]  # Your Telegram user ID
```

> **How to find your user ID**: Send a message to [@userinfobot](https://t.me/userinfobot) on Telegram.

Leave empty `[]` to allow anyone.

### 4.4 — AI Provider API Keys (At Least 1 Required)

Get free API keys from any of these providers:

| Provider | Free Tier | Get Key |
|---|---|---|
| **Groq** | Very fast, generous limits | [console.groq.com](https://console.groq.com) |
| **Gemini** | Good free tier | [aistudio.google.com](https://aistudio.google.com) |
| **OpenRouter** | Many free models | [openrouter.ai](https://openrouter.ai) |
| **Cerebras** | Fast inference | [cerebras.ai](https://cerebras.ai) |
| **NVIDIA** | Large model selection | [build.nvidia.com](https://build.nvidia.com) |

Fill in the keys you have (leave others as `""`):

```python
GROQ_API_KEY = "gsk_xxxxxxxxxxxx"
GEMINI_API_KEY = ""
OPENROUTER_API_KEY = ""
CEREBRAS_API_KEY = ""
NVIDIA_API_KEY = ""
```

### 4.5 — Web Search (Optional)

The bot has **two search engines** — only one active at a time:

| Engine | API Key Needed? | Speed | Quality |
|---|---|---|---|
| **Brave Search** | ✅ Yes (free tier available) | ~1 second | Excellent |
| **DuckDuckGo** | ❌ No key needed | ~2-3 seconds | Good |

```python
# Leave empty to auto-use DuckDuckGo (free)
BRAVE_API_KEY = ""

# Default engine: "brave" or "duckduckgo"
SEARCH_ENGINE = "brave"
```

**If you don't set a Brave API key**, the bot automatically uses DuckDuckGo. No config needed — it just works!

To get a Brave API key: [brave.com/search/api](https://brave.com/search/api/) (free tier: 2000 queries/month)

### 4.6 — DuckDNS Dynamic DNS (Optional)

If you want your ESP32 accessible by hostname:

```python
DUCKDNS_TOKEN = "your-duckdns-token"
DUCKDNS_DOMAIN = "myesp32"  # → myesp32.duckdns.org
DUCKDNS_INTERVAL = 300  # update every 5 minutes
```

Leave as `""` to disable.

### 4.7 — Default AI Provider

```python
DEFAULT_PROVIDER = "groq"  # Must match a provider with a key set above
```

---

## Step 5 — Upload Script to ESP32 (Auto-Run on Boot)

This is the key step. The ESP32 runs `main.py` automatically every time it powers on.

### 5.1 — Upload as `main.py`

1. In Thonny, open your configured `esp32_bot.py`
2. Go to **File → Save as...**
3. When asked **where to save**, choose **MicroPython device**
4. **Name the file `main.py`** ← This is critical!
5. Click OK — the file uploads to the ESP32

> **Why `main.py`?** MicroPython automatically runs `main.py` on every boot. This is how it auto-starts when plugged into power.

### 5.2 — Upload Prompt File (Required for Perplexity Prompt)

If your `esp32_bot.py` uses:

```python
SYSTEM_PROMPT_FILE = "perplexity-Prompt.txt"
```

you must also upload that file to the ESP32 filesystem.

1. In Thonny, open local `perplexity-Prompt.txt`
2. Go to **File → Save as...**
3. Choose **MicroPython device**
4. Save as exactly: **`perplexity-Prompt.txt`**

Store it in the same root location as `main.py` (no folder needed).

### 5.3 — Test It

1. Press the **RESET** button on the ESP32 (or unplug and replug)
2. Watch Thonny's Shell — you should see:

```
========================================
ESP32-C3 Telegram Bot starting...
Providers: groq
========================================
[Prompt] Loaded from file: perplexity-Prompt.txt
[WDT] Watchdog enabled (120s timeout)
[WiFi] Connecting to MyWiFiNetwork
[WiFi] Connected! 192.168.1.xx
[Bot] Fetching model lists...
[Models] Groq: fetched 12 models
[Bot] Starting Telegram polling...
```

3. Send a message to your bot on Telegram — it should reply! 🎉

If the prompt file is missing, you'll see:

```
[Prompt] Using fallback default (...)
```

and the bot will still run, but without the file-based Perplexity prompt.

### 5.4 — Disconnect & Power On

Once tested, you can:
- Close Thonny
- Unplug from computer
- Plug into **any USB power source** (phone charger, power bank, USB hub)
- The bot starts automatically — no computer needed!

---

## Bot Commands

Send these to your bot in Telegram:

### General

| Command | Description |
|---|---|
| `/start` | Show bot info and current provider |
| `/help` | List all commands |
| `/status` | Device info (WiFi, RAM, uptime, IP) |
| `/clear` | Clear conversation history |

### AI Provider & Model

| Command | Description |
|---|---|
| `/provider` | Show current provider + available providers |
| `/provider groq` | Switch to Groq |
| `/provider gemini` | Switch to Gemini |
| `/models` | List available models for current provider |
| `/model <id>` | Switch to a specific model |
| `/refresh` | Re-fetch model lists from provider APIs |

### Web Search

| Command | Description |
|---|---|
| `/web` | Show search status + active engine |
| `/web on` | Enable web search |
| `/web off` | Disable web search |
| `/web brave` | Switch to Brave Search API |
| `/web ddg` | Switch to DuckDuckGo (free) |

### How Web Search Works

When enabled, the bot can search the web for real-time information:

1. You ask: *"What's the weather in Bangkok today?"*
2. AI decides it needs current data → internally sends `SEARCH: weather Bangkok today`
3. Bot searches the web (Brave or DuckDuckGo) → gets snippets
4. AI reads the snippets → gives you an informed answer

**This is automatic** — you just chat normally. The AI decides when to search.

---

## Troubleshooting

### Board not showing up in Thonny

- Try a different USB cable (**must be a data cable**, not charge-only)
- Try a different USB port
- On Windows: check **Device Manager → Ports (COM & LPT)** for the port
- Enter bootloader mode: hold BOOT → press RESET → release BOOT

### `[WiFi] Waiting...` forever

- Double-check `WIFI_SSID` and `WIFI_PASSWORD` (case-sensitive!)
- Make sure WiFi is 2.4 GHz — the **ESP32-C3 does NOT support 5 GHz**
- Move closer to the router

### Bot doesn't respond on Telegram

- Check the Telegram token is correct (copy-paste from BotFather)
- Make sure `ALLOWED_USER_IDS` includes your user ID (or is empty `[]`)
- Check Thonny Shell for error messages

### `MemoryError` or random crashes

- This is normal — the ESP32-C3 has only 400 KB RAM
- Keep `MAX_HISTORY = 5` (don't increase it)
- Keep `MAX_SESSIONS = 3`
- The bot auto-reboots after fatal crashes (10 second delay)

### `[Search] DDG error` or `[Search] Brave error`

- Check WiFi is connected
- DuckDuckGo may occasionally block automated requests — just retry
- Brave: verify your API key is correct

### How to update the code

1. Connect ESP32 to computer via USB
2. Open Thonny
3. Click **Stop/Restart** (red button) to interrupt the running script
4. Open the new `esp32_bot.py` in Thonny
5. **File → Save as → MicroPython device → `main.py`**
6. If prompt changed, also upload `perplexity-Prompt.txt` to **MicroPython device**
7. Press RESET on the board

---

## RAM Optimization Settings

These defaults are tuned for the 400 KB ESP32-C3:

```python
MAX_TOKENS = 512       # AI response length limit
MAX_HISTORY = 5        # Chat messages kept in memory
MAX_SESSIONS = 3       # Concurrent user sessions
MAX_RESPONSE_LEN = 4000  # Truncate oversized responses
MAX_SEARCH_RESULTS = 3   # Search results per query
MAX_SNIPPET_LEN = 200    # Characters per search snippet
```

> ⚠️ **Do not increase these values** unless you know what you're doing. The ESP32-C3 will crash with `MemoryError` if RAM runs out.

---

## Architecture Overview

```
┌──────────────────────────────────────┐
│          ESP32-C3 Super Mini         │
│                                      │
│  main.py (auto-runs on boot)         │
│    ├── WiFi Connect (2.4 GHz)        │
│    ├── DuckDNS Updater (optional)    │
│    ├── Telegram Long-Poll Loop       │
│    │     ├── /commands → handler     │
│    │     └── messages → AI chat      │
│    │           ├── Pass 1: AI reply  │
│    │           ├── SEARCH: detected? │
│    │           │    ├── Brave API    │
│    │           │    └── DuckDuckGo   │
│    │           └── Pass 2: answer    │
│    └── Watchdog Timer (120s)         │
│                                      │
│  Providers: Groq, Gemini, OpenRouter │
│             Cerebras, NVIDIA         │
└──────────────────────────────────────┘
```

**All state is in-memory.** Reboot = fresh start. No files written to flash.
