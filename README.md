# 🤖 Multi-Provider AI Telegram Bot

> **A universal free AI model hub for Telegram** — Switch between Groq, Gemini, OpenRouter, Cerebras, and NVIDIA with simple commands. Built-in web search, model validation, and thinking mode!

[![GHCR](https://img.shields.io/badge/ghcr.io-foxy1402%2Fchat--telegram-blue?logo=github)](https://github.com/foxy1402/chat-telegram/pkgs/container/chat-telegram)

---

## ✨ Features

- 🔄 **5 AI Providers** — Groq, Gemini, OpenRouter, Cerebras, NVIDIA
- 🖼️ **Image OCR** — Send any photo and the bot extracts text or describes content via NVIDIA vision models
- 🌐 **Web Search** — AI can search the web for real-time info (Brave API, SearXNG, or DuckDuckGo)
- 💭 **Thinking Mode** — See AI reasoning traces with NVIDIA models
- ✅ **Model Validation** — Test which models actually work before using them
- ⚡ **Easy Switching** — Change providers and models with simple commands
- 🔒 **User Whitelisting** — Restrict access to specific Telegram users
- 💬 **Conversation History** — Context maintained across messages (including after OCR)
- 🆓 **100% Free** — Uses only free-tier APIs (no credit card required)
- 🐳 **Docker Ready** — Pre-built image on GitHub Container Registry (GHCR)

---

## 🚀 Quick Start

### 1. Get Your API Keys

You need at least **one** provider key (all are free):

| Provider | Get API Key | Free Tier |
|----------|-------------|-----------|
| **Groq** | [console.groq.com](https://console.groq.com) | 14,400 requests/day |
| **Gemini** | [aistudio.google.com](https://aistudio.google.com) | 100–1,000 requests/day |
| **OpenRouter** | [openrouter.ai](https://openrouter.ai) | 50–1,000 requests/day |
| **Cerebras** | [cerebras.ai](https://cerebras.ai) | Free tier, fast inference |
| **NVIDIA** | [build.nvidia.com](https://build.nvidia.com) | Free tier, thinking models 💭 |

### 2. Get Telegram Bot Token

1. Open Telegram → search `@BotFather`
2. Send `/newbot` and follow prompts
3. Copy your bot token

### 3. Get Your Telegram User ID (Optional)

To restrict access to only you:
1. Search `@userinfobot` on Telegram
2. It will show your numeric user ID (e.g., `123456789`)

---

## 📦 Deployment

### Option 1: GHCR (Recommended)

```bash
# Pull and run
docker run -d \
  -e TELEGRAM_TOKEN=your_token \
  -e GROQ_API_KEY=your_key \
  -e ALLOWED_USER_IDS=your_id \
  --name multi-ai-bot \
  ghcr.io/foxy1402/chat-telegram:latest

# Or use .env file
docker run -d --env-file .env --name multi-ai-bot ghcr.io/foxy1402/chat-telegram:latest
```

```bash
# Check logs
docker logs -f multi-ai-bot

# Stop / restart
docker stop multi-ai-bot
docker start multi-ai-bot
```

### Option 2: Cloud Platforms (Claw Cloud, etc.)

1. **Image**: `ghcr.io/foxy1402/chat-telegram:latest`
2. **Environment Variables**: Set your API keys
3. **Deploy**: Platform pulls and runs automatically

### Option 3: Build Locally

```bash
git clone <your-repo-url>
cd chat-telegram
cp .env.example .env   # Edit with your keys
docker build -t multi-ai-bot .
docker run -d --env-file .env --name multi-ai-bot multi-ai-bot
```

### Option 4: Run Without Docker

```bash
pip install -r requirements.txt
cp .env.example .env   # Edit with your keys
python bot.py
```

---

## ⚙️ Environment Variables

### Required

```env
TELEGRAM_TOKEN=your_telegram_bot_token
```

### AI Providers (at least one required)

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
CEREBRAS_API_KEY=your_cerebras_api_key
NVIDIA_API_KEY=your_nvidia_api_key
```

### Access Control

```env
# Comma-separated Telegram user IDs. Leave empty to allow anyone.
ALLOWED_USER_IDS=123456789,987654321
```

### Bot Behavior

```env
# Default provider: groq, gemini, openrouter, cerebras, or nvidia
DEFAULT_PROVIDER=groq

# Max tokens per response (512 = concise, 1024 = balanced, 2048 = detailed)
MAX_TOKENS=512

# Temperature (0.0 = deterministic, 0.7 = balanced, 1.0 = creative)
TEMPERATURE=0.7

# Max messages kept in conversation history (should be even: user+assistant pairs)
MAX_HISTORY_MESSAGES=20
```

### Image OCR

OCR requires `NVIDIA_API_KEY`. All image processing is done fully in-memory — no files are written to disk.

```env
# Vision model used for image OCR (default: google/gemma-3-27b-it)
# Other good options: meta/llama-3.2-11b-vision-instruct
#                     meta/llama-3.2-90b-vision-instruct
#                     microsoft/phi-3.5-vision-instruct
NVIDIA_VISION_MODEL=google/gemma-3-27b-it

# Maximum image size accepted (default: 15728640 = 15 MB)
MAX_IMAGE_BYTES=15728640
```

### Web Search

```env
# Brave Search API key (free: https://brave.com/search/api/)
# If not set, DuckDuckGo is used (free, no key needed)
BRAVE_API_KEY=your_brave_api_key

# SearXNG self-hosted instance URL (no API key needed)
# Leave empty if not using SearXNG
SEARXNG_URL=http://your-searxng-host

# Engine: "brave", "searxng", or "duckduckgo" (default: brave)
# Falls back to DuckDuckGo if the selected engine returns no results
SEARCH_ENGINE=brave

# Number of results to fetch (default: 3)
MAX_SEARCH_RESULTS=3

# Max snippet length per result (default: 300)
MAX_SNIPPET_LEN=300
```

---

## 💬 Bot Commands

### Provider & Model Management

| Command | Description |
|---------|-------------|
| `/provider` | Show current provider and available options |
| `/provider <name>` | Switch provider (`groq`, `gemini`, `openrouter`, `cerebras`, `nvidia`) |
| `/models` | List verified working models for current provider |
| `/models all` | List all available models from API |
| `/model <id>` | Switch to a specific model |
| `/refresh` | Re-fetch model lists from all provider APIs |

### Web Search 🌐

| Command | Description |
|---------|-------------|
| `/web` | Show current web search status and engine |
| `/web on` | Enable web search |
| `/web off` | Disable web search |
| `/web brave` | Switch to Brave Search API |
| `/web searxng` | Switch to SearXNG (self-hosted) |
| `/web ddg` | Switch to DuckDuckGo (free, no key) |

When web search is enabled, the AI automatically detects when your question needs real-time info (news, current events, live data) and searches the web. Works with any provider.

### Model Validation ✅

| Command | Description |
|---------|-------------|
| `/validate` | Test all models with real API calls to find working ones |
| `/verified` | Show only validated working models |
| `/clearvalidation` | Clear cache and re-test all models |

Validation results are cached to disk (`validated_models.json`) and persist across restarts. Smart validation skips already-tested models.

### Image OCR 🖼️

Just send a photo — no commands needed.

| Action | Result |
|--------|--------|
| Send a photo (no caption) | Extracts and transcribes all visible text |
| Send a photo with a caption | Uses your caption as the question/prompt |
| Send multiple photos at once | Processes all photos and replies once with numbered results |

- Requires `NVIDIA_API_KEY` to be set
- The OCR result is stored in conversation history as plain text, so you can ask follow-up questions about the content without re-uploading the image
- Failed photos in a multi-photo upload are reported inline; remaining photos still process
- The vision model can be changed via the `NVIDIA_VISION_MODEL` env var

### Thinking Mode 💭 (NVIDIA Only)

| Command | Description |
|---------|-------------|
| `/thinking` | Check current thinking mode status |
| `/thinking on` | Enable — see AI reasoning in responses |
| `/thinking off` | Disable — get concise responses |

Only NVIDIA models with a 💭 icon support thinking. The bot will warn you if your current model doesn't support it.

### General

| Command | Description |
|---------|-------------|
| `/start` | Welcome message with status overview |
| `/clear` | Clear conversation history |
| `/help` | Full command reference |

---

## 🎯 Usage Examples

### Basic Chat
```
You: Hello, what's the capital of France?
Bot: Paris.
```

### Switch Provider & Model
```
You: /provider nvidia
Bot: ✅ Switched to NVIDIA!
     Using model: openai/gpt-oss-120b

You: /models all
Bot: 🤖 NVIDIA (All Models):
     • openai/gpt-oss-120b - GPT-OSS 120B (Stable Free Tier) ✓
     • deepseek-ai/deepseek-v3.2 - DeepSeek V3.2 💭
     • qwen/qwen3-235b-a22b - Qwen3 235B 💭
     ...

You: /model deepseek-ai/deepseek-v3.2
Bot: ✅ Switched to model: deepseek-ai/deepseek-v3.2
```

### Thinking Mode
```
You: /thinking on
Bot: ✅ Thinking mode enabled! 💭

You: Explain quantum computing
Bot: 💭 **Thinking:**
     Let me break this down...

     Quantum computing uses qubits instead of classical bits...
```

### Image OCR — Single Photo
```
You: [sends a photo of a receipt]
Bot: Receipt — SM Supermarket
     Milk 1L        ₱85.00
     Eggs (12pcs)   ₱120.00
     Bread          ₱55.00
     Total          ₱260.00

You: what's the total amount?
Bot: The total is ₱260.00.
```

### Image OCR — Photo with Caption
```
You: [sends a photo of a menu] what's the most expensive dish?
Bot: The most expensive dish is the Wagyu Beef Steak at ₱1,850.
```

### Image OCR — Multiple Photos
```
You: [sends 3 photos of a menu album]
Bot: *Image 1/3*
     Starters: Spring Rolls ₱120, Calamari ₱150...

     ---

     *Image 2/3*
     Mains: Beef Adobo ₱280, Sinigang ₱320...

     ---

     *Image 3/3*
     Desserts: Halo-halo ₱150, Leche Flan ₱90...

You: what does the Sinigang taste like?
Bot: Sinigang is a Filipino sour soup with a tamarind-based broth...
```

### Web Search
```
You: /web on
Bot: ✅ Web search enabled (Brave API).

You: What happened in tech news today?
Bot: (Bot detects this needs real-time info, searches the web, then answers)
     Based on today's news: ...
```

### Model Validation
```
You: /validate
Bot: 🔍 Full Validation — Testing all 10 models
     ⏳ ~20s (2s delay to avoid rate limits)
     ...
     ✅ Validation Complete
     • Tested: 10
     • ✅ Newly validated: 7
     • ❌ Failed: 3

You: /verified
Bot: ✅ Verified Models for NVIDIA:
     • openai/gpt-oss-120b ✓
     • deepseek-ai/deepseek-v3.2
     ...
```

---

## 📊 Provider Comparison

| Provider | Daily Limit | Best For | Speed | Special Features |
|----------|-------------|----------|-------|------------------|
| **Groq** | 14,400 | Real-time chat | ⚡⚡⚡⚡⚡ | — |
| **Cerebras** | Free tier | Fast inference | ⚡⚡⚡⚡⚡ | — |
| **NVIDIA** | Free tier | Thinking + vision | ⚡⚡⚡⚡ | 💭 Reasoning mode, 🖼️ Image OCR |
| **Gemini** | 100–1,000 | Quality responses | ⚡⚡⚡ | — |
| **OpenRouter** | 50–1,000 | Model variety | ⚡⚡ | — |

---

## 🖼️ How Image OCR Works

1. **You send a photo** (with or without a caption)
2. **Bot downloads it to RAM** — no files written to disk, ever
3. **Raw bytes are base64-encoded** then the raw buffer is freed immediately
4. **NVIDIA vision API is called** with the encoded image and your prompt (or the default OCR prompt)
5. **Response is streamed back**, parsed, and sent as a Telegram message
6. **Base64 string is deleted** and `gc.collect()` is called — memory fully reclaimed
7. **OCR result is stored in conversation history as plain text** — you can ask follow-up questions about the content in the same chat without re-uploading

For **multi-photo uploads** (albums): Telegram sends each photo as a separate event. The bot collects them for 1.5 seconds, then processes them one at a time (only one image in memory at a time) and sends a single combined reply.

To change the OCR model, set `NVIDIA_VISION_MODEL` in your environment. The API call is retried up to 2 times on transient errors (rate limits, timeouts, 5xx responses).

---

## 🌐 How Web Search Works

1. **You ask a question** — e.g., "What's the latest iPhone?"
2. **Bot evaluates search need** — using a lightweight keyword/time heuristic
3. **If needed, bot searches directly** — Using Brave API, SearXNG, or DuckDuckGo
4. **Bot sends your question + snippets to AI** — in one answer pass
5. **AI returns an up-to-date response** — grounded in the fetched snippets

You can use `/web off` to disable this entirely, `/web searxng` to use your self-hosted SearXNG instance, or `/web ddg` to use the free DuckDuckGo engine without any API key.

---

## 🔧 Troubleshooting

### Bot doesn't respond
1. Check logs: `docker logs -f multi-ai-bot`
2. Verify API keys are correct
3. Ensure at least one provider key is set

### "Not authorized" message
- Add your Telegram user ID to `ALLOWED_USER_IDS`
- Or remove `ALLOWED_USER_IDS` to allow anyone

### Provider errors
- Switch provider: `/provider <name>`
- Check rate limits — try again in a few minutes
- Run `/validate` to find working models

### Web search not working
- Check if web search is enabled: `/web`
- For Brave: ensure `BRAVE_API_KEY` is set
- For SearXNG: ensure `SEARXNG_URL` is set and the instance has JSON format enabled
- Try DuckDuckGo (no key needed): `/web ddg`

### Models not loading
- Run `/refresh` to re-fetch from APIs
- Run `/validate` to test which models work
- Run `/clearvalidation` then `/validate` to start fresh

### Image OCR not working
- Ensure `NVIDIA_API_KEY` is set — OCR requires it regardless of your active chat provider
- Check that `NVIDIA_VISION_MODEL` is a vision-capable model (default `google/gemma-3-27b-it` supports vision)
- If you get empty responses, try a different model via `NVIDIA_VISION_MODEL=meta/llama-3.2-11b-vision-instruct`
- For rate limit errors the bot retries automatically (up to 2 retries with backoff)

---

## 🛠️ Project Structure

```
chat-telegram/
├── bot.py                  # Main bot — all features including image OCR
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build config
├── .env.example            # Example environment file
├── validated_models.json   # Auto-generated validation cache
├── clear_old_commands.py   # Utility to remove old bot commands
├── esp32_bot.py            # ESP32 version of the bot
├── ESP32-GUIDE.md          # ESP32 setup guide
└── README.md               # This file
```

---

## 📝 License

MIT License — Feel free to use and modify!

---

## 🙏 Credits

Built with:
- [python-telegram-bot](https://python-telegram-bot.org/)
- [Groq](https://groq.com/)
- [Google Gemini](https://ai.google.dev/)
- [OpenRouter](https://openrouter.ai/)
- [Cerebras](https://cerebras.ai/)
- [NVIDIA](https://build.nvidia.com/)
- [Brave Search](https://brave.com/search/api/)

---

**Need help?** Open an issue or check the logs with `docker logs -f multi-ai-bot`
