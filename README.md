# 🤖 Multi-Provider AI Telegram Bot

> **A universal free AI model hub for Telegram** - Switch between Groq, Gemini, OpenRouter, Cerebras, and NVIDIA with simple commands!

[![Docker Hub](https://img.shields.io/badge/docker-snoopylikefree5%2Fmulti--ai--bot-blue?logo=docker)](https://hub.docker.com/r/snoopylikefree5/multi-ai-bot)

---

## ✨ Features

- 🔄 **Multiple AI Providers**: Groq, Google Gemini, OpenRouter, Cerebras, NVIDIA
- 💭 **Thinking Mode**: See AI reasoning process with NVIDIA models (NEW!)
- ⚡ **Easy Switching**: Change providers and models with simple commands
- 🔒 **User Whitelisting**: Restrict access to specific users
- 💬 **Conversation History**: Maintains context across messages
- 🆓 **100% Free**: Uses only free tier APIs (no credit card required)
- 🐳 **Docker Ready**: Pre-built image available on Docker Hub

---

## 🚀 Quick Start

### 1. Get Your API Keys

You'll need at least **one** of these (all are free):

| Provider | Get API Key | Free Tier |
|----------|-------------|-----------|
| **Groq** | [console.groq.com](https://console.groq.com) | 14,400 requests/day |
| **Gemini** | [aistudio.google.com](https://aistudio.google.com) | 100-1,000 requests/day |
| **OpenRouter** | [openrouter.ai](https://openrouter.ai) | 50-1,000 requests/day |
| **Cerebras** | [cerebras.ai](https://cerebras.ai) | Free tier with fast inference |
| **NVIDIA** | [build.nvidia.com](https://build.nvidia.com) | Free tier with thinking models 💭 |

### 2. Get Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the prompts
3. Copy your bot token

### 3. Get Your Telegram User ID (Optional)

To restrict bot access to only you:
1. Search for `@userinfobot` on Telegram
2. Start a chat - it will show your user ID
3. Copy the ID (e.g., `123456789`)

---

## 📦 Deployment

### Option 1: Docker Hub (Recommended - Pre-built Image)

The bot is already available on Docker Hub! Just pull and run:

```bash
# Pull the latest image
docker pull snoopylikefree5/multi-ai-bot:latest

# Run with environment variables
docker run -d \
  -e TELEGRAM_TOKEN=your_telegram_bot_token \
  -e GROQ_API_KEY=your_groq_api_key \
  -e GEMINI_API_KEY=your_gemini_api_key \
  -e OPENROUTER_API_KEY=your_openrouter_api_key \
  -e CEREBRAS_API_KEY=your_cerebras_api_key \
  -e NVIDIA_API_KEY=your_nvidia_api_key \
  -e ALLOWED_USER_IDS=your_telegram_user_id \
  --name multi-ai-bot \
  snoopylikefree5/multi-ai-bot:latest

# Or use .env file
docker run -d --env-file .env --name multi-ai-bot snoopylikefree5/multi-ai-bot:latest
```

**Check logs:**
```bash
docker logs -f multi-ai-bot
```

**Stop/restart:**
```bash
docker stop multi-ai-bot
docker start multi-ai-bot
```

### Option 2: Claw Cloud / Cloud Platforms

Deploy on [Claw Cloud](https://claw.cloud) or any container platform:

1. **Image**: `snoopylikefree5/multi-ai-bot:latest`
2. **Environment Variables**: Set your API keys (see below)
3. **Deploy**: Platform will automatically pull and run

### Option 3: Build Locally

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd chat-telegram

# 2. Create .env file
cp .env.example .env
# Edit .env with your actual keys

# 3. Build and run
docker build -t multi-ai-bot .
docker run -d --env-file .env --name multi-ai-bot multi-ai-bot
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

### Optional

```env
# Restrict access to specific users (comma-separated)
ALLOWED_USER_IDS=123456789,987654321

# Default provider (groq, gemini, openrouter, cerebras, or nvidia)
DEFAULT_PROVIDER=groq

# Response Configuration (Control response length and style)
# Maximum tokens per response (lower = shorter responses, less TPM usage)
# Recommended: 512 (concise), 1024 (balanced), 2048 (detailed)
MAX_TOKENS=512

# Temperature (0.0-1.0): 0.7 balanced, 0.3 focused, 0.9 creative
TEMPERATURE=0.7

# Maximum conversation history messages
MAX_HISTORY_MESSAGES=20

# System prompt to guide AI behavior
# Instructs AI to be concise and use Telegram-friendly formatting (no tables)
SYSTEM_PROMPT=You are a helpful AI assistant. Be concise and straight to the point. Avoid unnecessary explanations unless specifically asked. IMPORTANT: Never use markdown tables (|---|---| format) as they are unreadable in Telegram. Instead, use bullet points, numbered lists, or simple line-by-line formatting.
```

---

## 💬 Bot Commands

### Provider Management

- `/provider` - Show current provider and available options
- `/provider <name>` - Switch to a different provider
  - Example: `/provider gemini`, `/provider cerebras`, or `/provider nvidia`
- `/models` - List verified working models for current provider
- `/models all` - List all available models from API
- `/model <name>` - Switch to a specific model
  - Example: `/model gemini-2.5-pro`

### Thinking Mode (NVIDIA Only) 💭

- `/thinking on` - Enable thinking/reasoning mode (shows AI's thought process)
- `/thinking off` - Disable thinking mode (more concise responses)
- `/thinking` - Check current thinking mode status

**Note:** Only NVIDIA provider supports thinking mode. Models with 💭 icon support this feature.

### Model Validation

- `/validate` - Test all models to see which actually work
- `/verified` - Show only validated working models
- `/clearvalidation` - Clear validation cache and re-test

### Other Commands

- `/start` - Show welcome message
- `/clear` - Clear conversation history
- `/refresh` - Refresh model lists from providers
- `/help` - Show help message

---

## 🎯 Usage Examples

```
You: /start
Bot: 🤖 Hello! I'm your Multi-Provider AI assistant.
     📡 Current Provider: Groq
     🔧 Available Providers: groq, gemini, openrouter, cerebras, nvidia

You: /provider nvidia
Bot: ✅ Switched to NVIDIA!
     Using model: openai/gpt-oss-120b

You: /models all
Bot: 🤖 Available Models for NVIDIA:
     • openai/gpt-oss-120b - GPT-OSS 120B (Stable Free Tier) ✓
     • deepseek-ai/deepseek-v3.2 - DeepSeek V3.2 💭
     • qwen/qwen3-235b-a22b - Qwen3 235B 💭

You: /model deepseek-ai/deepseek-v3.2
Bot: ✅ Switched to model: deepseek-ai/deepseek-v3.2

You: /thinking on
Bot: ✅ Thinking mode enabled! 💭
     Model deepseek-ai/deepseek-v3.2 supports thinking.

You: Explain quantum computing briefly
Bot: 💭 **Thinking:**
     Let me break this down into simple terms...
     
     Quantum computing uses quantum mechanics principles...
```

---

## 📊 Provider Comparison

| Provider | Daily Limit | Best For | Speed | Special Features |
|----------|-------------|----------|-------|------------------|
| **Groq** | 14,400 | Real-time chat | ⚡⚡⚡⚡⚡ | - |
| **Cerebras** | Free tier | Fast inference | ⚡⚡⚡⚡⚡ | - |
| **NVIDIA** | Free tier | Thinking models | ⚡⚡⚡⚡ | 💭 Reasoning mode |
| **Gemini** | 100-1,000 | Quality responses | ⚡⚡⚡ | - |
| **OpenRouter** | 50-1,000 | Model variety | ⚡⚡ | - |

---

## 🔧 Troubleshooting

### Bot doesn't respond

1. Check logs: `docker logs -f multi-ai-bot`
2. Verify API keys are correct
3. Ensure at least one provider API key is set

### "Not authorized" message

- Add your Telegram user ID to `ALLOWED_USER_IDS`
- Or remove `ALLOWED_USER_IDS` to allow anyone

### Provider errors

- Try switching to another provider: `/provider <name>`
- Check if you've hit rate limits
- Verify the specific provider's API key

---

## 🛠️ Development

### File Structure

```
chat-telegram/
├── bot.py              # Main bot code
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── .env.example       # Example environment file
└── README.md          # This file
```

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export TELEGRAM_TOKEN=your_token
export GROQ_API_KEY=your_key
# ... other keys

# 3. Run
python bot.py
```

---

## 📝 License

MIT License - Feel free to use and modify!

---

## 🙏 Credits

Built with:
- [python-telegram-bot](https://python-telegram-bot.org/)
- [Groq](https://groq.com/)
- [Google Gemini](https://ai.google.dev/)
- [OpenRouter](https://openrouter.ai/)
- [Cerebras](https://cerebras.ai/)
- [NVIDIA](https://build.nvidia.com/)

---

**Need help?** Open an issue or check the logs for error messages!