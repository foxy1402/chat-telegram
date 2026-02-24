# 🤖 Multi-Provider AI Telegram Bot

> **A universal free AI model hub for Telegram** — Switch between Groq, Gemini, OpenRouter, Cerebras, and NVIDIA with simple commands. Built-in web search, model validation, and thinking mode!

[![Docker Hub](https://img.shields.io/badge/docker-snoopylikefree5%2Fmulti--ai--bot-blue?logo=docker)](https://hub.docker.com/r/snoopylikefree5/multi-ai-bot)

---

## ✨ Features

- 🔄 **5 AI Providers** — Groq, Gemini, OpenRouter, Cerebras, NVIDIA
- 🌐 **Web Search** — AI can search the web for real-time info (Brave API or DuckDuckGo)
- 💭 **Thinking Mode** — See AI reasoning traces with NVIDIA models
- ✅ **Model Validation** — Test which models actually work before using them
- ⚡ **Easy Switching** — Change providers and models with simple commands
- 🔒 **User Whitelisting** — Restrict access to specific Telegram users
- 💬 **Conversation History** — Context maintained across messages
- 🆓 **100% Free** — Uses only free-tier APIs (no credit card required)
- 🐳 **Docker Ready** — Pre-built image on Docker Hub

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

### Option 1: Docker Hub (Recommended)

```bash
# Pull and run
docker run -d \
  -e TELEGRAM_TOKEN=your_token \
  -e GROQ_API_KEY=your_key \
  -e ALLOWED_USER_IDS=your_id \
  --name multi-ai-bot \
  snoopylikefree5/multi-ai-bot:latest

# Or use .env file
docker run -d --env-file .env --name multi-ai-bot snoopylikefree5/multi-ai-bot:latest
```

```bash
# Check logs
docker logs -f multi-ai-bot

# Stop / restart
docker stop multi-ai-bot
docker start multi-ai-bot
```

### Option 2: Cloud Platforms (Claw Cloud, etc.)

1. **Image**: `snoopylikefree5/multi-ai-bot:latest`
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

# Max messages kept in conversation history
MAX_HISTORY_MESSAGES=20

# System prompt (customize AI personality)
SYSTEM_PROMPT=You are a helpful AI assistant. Be concise and straight to the point.
```

### Web Search

```env
# Brave Search API key (free: https://brave.com/search/api/)
# If not set, DuckDuckGo is used (free, no key needed)
BRAVE_API_KEY=your_brave_api_key

# Engine: "brave" or "duckduckgo" (default: brave)
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
| `/web ddg` | Switch to DuckDuckGo (free, no key) |

When web search is enabled, the AI automatically detects when your question needs real-time info (news, current events, live data) and searches the web. Works with any provider.

### Model Validation ✅

| Command | Description |
|---------|-------------|
| `/validate` | Test all models with real API calls to find working ones |
| `/verified` | Show only validated working models |
| `/clearvalidation` | Clear cache and re-test all models |

Validation results are cached to disk (`validated_models.json`) and persist across restarts. Smart validation skips already-tested models.

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

### Web Search
```
You: /web on
Bot: ✅ Web search enabled (Brave API).

You: What happened in tech news today?
Bot: (AI detects this needs real-time info, searches the web, then answers)
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
| **NVIDIA** | Free tier | Thinking models | ⚡⚡⚡⚡ | 💭 Reasoning mode |
| **Gemini** | 100–1,000 | Quality responses | ⚡⚡⚡ | — |
| **OpenRouter** | 50–1,000 | Model variety | ⚡⚡ | — |

---

## 🌐 How Web Search Works

1. **You ask a question** — e.g., "What's the latest iPhone?"
2. **AI evaluates** — Does this need real-time info?
3. **If yes** → AI responds with `SEARCH: latest iPhone release`
4. **Bot searches the web** — Using Brave API or DuckDuckGo
5. **AI answers again** — Using search results to give an accurate, up-to-date response

The search is a two-pass process:
- **Pass 1**: AI decides if it needs to search (with a special prompt)
- **Pass 2**: AI answers using web results (results don't pollute your conversation history)

You can use `/web off` to disable this entirely, or `/web ddg` to use the free DuckDuckGo engine without any API key.

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
- Try DuckDuckGo (no key needed): `/web ddg`

### Models not loading
- Run `/refresh` to re-fetch from APIs
- Run `/validate` to test which models work
- Run `/clearvalidation` then `/validate` to start fresh

---

## 🛠️ Project Structure

```
chat-telegram/
├── bot.py                  # Main bot (1862 lines — all features)
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