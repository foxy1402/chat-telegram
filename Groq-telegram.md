# 🤖 Groq-Powered Telegram Bot Deployment Guide

> **Deploy a free AI Telegram bot using Groq API on Claw Cloud (or any Docker platform)**

---

## 📋 What You'll Get

- ✅ AI-powered Telegram bot using Groq (14,400 free requests/day)
- ✅ Deployed on Claw Cloud for free (always running)
- ✅ Easy configuration via environment variables
- ✅ Lightweight Docker container
- ✅ No complex setup needed

---

## 🎯 Prerequisites

1. **Groq API Key** - Get from: https://console.groq.com
2. **Telegram Bot Token** - Get from @BotFather on Telegram
3. **Claw Cloud Account** - Sign up at: https://claw.cloud (free tier)
4. **GitHub Account** - To store your bot code

---

## 📦 Step 1: Create Your Bot Code

### 1.1 Create a New Directory

On your local machine (or on GitHub directly):

```bash
mkdir groq-telegram-bot
cd groq-telegram-bot
```

### 1.2 Create `bot.py`

```python
import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from groq import Groq

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
ALLOWED_USER_IDS = os.getenv('ALLOWED_USER_IDS', '').split(',')

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Store conversation history (in-memory, resets on restart)
conversations = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    user_id = str(update.effective_user.id)
    
    # Check if user is allowed (if ALLOWED_USER_IDS is set)
    if ALLOWED_USER_IDS[0] and user_id not in ALLOWED_USER_IDS:
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    await update.message.reply_text(
        "🤖 Hello! I'm your Groq-powered AI assistant.\n\n"
        "Just send me a message and I'll respond!\n\n"
        "Commands:\n"
        "/start - Show this message\n"
        "/clear - Clear conversation history\n"
        "/help - Show help"
    )

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history."""
    user_id = str(update.effective_user.id)
    if user_id in conversations:
        conversations[user_id] = []
    await update.message.reply_text("🗑️ Conversation history cleared!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message."""
    await update.message.reply_text(
        "💡 How to use:\n\n"
        "Just send me any message and I'll respond using Groq AI!\n\n"
        "Examples:\n"
        "• What's the weather like?\n"
        "• Explain quantum physics\n"
        "• Write a Python function\n"
        "• Tell me a joke\n\n"
        "Commands:\n"
        "/clear - Clear conversation history\n"
        "/help - Show this help"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages."""
    user_id = str(update.effective_user.id)
    
    # Check if user is allowed
    if ALLOWED_USER_IDS[0] and user_id not in ALLOWED_USER_IDS:
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    user_message = update.message.text
    
    # Initialize conversation history for new users
    if user_id not in conversations:
        conversations[user_id] = []
    
    # Add user message to history
    conversations[user_id].append({
        "role": "user",
        "content": user_message
    })
    
    # Keep only last 10 messages to avoid token limits
    if len(conversations[user_id]) > 10:
        conversations[user_id] = conversations[user_id][-10:]
    
    try:
        # Send typing indicator
        await update.message.chat.send_action(action="typing")
        
        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=conversations[user_id],
            model=GROQ_MODEL,
            temperature=0.7,
            max_tokens=1024,
        )
        
        # Get response
        bot_response = chat_completion.choices[0].message.content
        
        # Add bot response to history
        conversations[user_id].append({
            "role": "assistant",
            "content": bot_response
        })
        
        # Send response
        await update.message.reply_text(bot_response)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(
            f"❌ Error: {str(e)}\n\n"
            "Please try again or use /clear to reset the conversation."
        )

def main():
    """Start the bot."""
    # Validate environment variables
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is required!")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is required!")
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    logger.info("Bot started!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
```

### 1.3 Create `requirements.txt`

```txt
python-telegram-bot==21.0.1
groq==0.11.0
```

### 1.4 Create `Dockerfile`

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code
COPY bot.py .

# Run the bot
CMD ["python", "bot.py"]
```

### 1.5 Create `.dockerignore`

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env
.git
.gitignore
README.md
```

---

## 🚀 Step 2: Build and Push Docker Image

### 2.1 Create Docker Hub Account

1. Go to https://hub.docker.com
2. Sign up for free account
3. Remember your username (you'll need it)

### 2.2 Build and Push Docker Image

**On your local machine (or use GitHub Actions):**

```bash
# Login to Docker Hub
docker login

# Build the image (replace YOUR_DOCKERHUB_USERNAME)
docker build -t YOUR_DOCKERHUB_USERNAME/groq-telegram-bot:latest .

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/groq-telegram-bot:latest
```

**Example:**
```bash
docker build -t johndoe/groq-telegram-bot:latest .
docker push johndoe/groq-telegram-bot:latest
```

### 2.3 Deploy on Claw Cloud

1. **Go to Claw Cloud**: https://claw.cloud
2. **Sign up / Log in**
3. **Create New App**:
   - Click "New App" or "Deploy"
   - Enter your Docker image: `YOUR_DOCKERHUB_USERNAME/groq-telegram-bot:latest`
   - Example: `johndoe/groq-telegram-bot:latest`
4. **Configure Environment Variables**:
   - Click "Environment Variables" or "Settings"
   - Add the following:

   | Variable | Value | Example |
   |----------|-------|---------|
   | `TELEGRAM_TOKEN` | Your bot token from @BotFather | `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz` |
   | `GROQ_API_KEY` | Your Groq API key | `gsk_abc123...` |
   | `GROQ_MODEL` | Model to use (optional) | `llama-3.3-70b-versatile` |
   | `ALLOWED_USER_IDS` | Your Telegram user ID (optional) | `123456789` |

5. **Deploy**:
   - Click "Deploy" or "Start"
   - Wait for container to start (30 seconds - 1 minute)
   - Bot will start automatically!

### 2.4 Alternative: Use Pre-built Image (Skip Building)

**If you don't want to build locally**, you can use GitHub Actions to auto-build:

Create `.github/workflows/docker-build.yml`:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ secrets.DOCKERHUB_USERNAME }}/groq-telegram-bot:latest
```

Then:
1. Add secrets to GitHub repo: `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`
2. Push to GitHub - image builds automatically
3. Use the image on Claw Cloud

---

## 🔧 Alternative: Deploy on Other Platforms

### Railway.app

1. Go to https://railway.app
2. "New Project" → "Deploy from GitHub"
3. Select your repository
4. Add environment variables (same as above)
5. Deploy!

### Render.com

1. Go to https://render.com
2. "New" → "Web Service"
3. Connect GitHub repository
4. Set environment variables
5. Deploy!

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Launch app
flyctl launch

# Set environment variables
flyctl secrets set TELEGRAM_TOKEN=your_token
flyctl secrets set GROQ_API_KEY=your_key

# Deploy
flyctl deploy
```

---

## 📱 Step 3: Get Your Telegram Bot Token

1. **Open Telegram** and search for `@BotFather`
2. **Send** `/newbot`
3. **Follow prompts**:
   - Bot name: `My Groq Assistant`
   - Username: `mygroq_bot` (must end with `bot`)
4. **Copy the token** (looks like: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)

---

## 🔐 Step 4: Get Your Telegram User ID (Optional Security)

To restrict bot access to only you:

1. **Open Telegram** and search for `@userinfobot`
2. **Start chat** - it will show your user ID
3. **Copy the ID** (e.g., `123456789`)
4. **Add to environment variables** as `ALLOWED_USER_IDS`

---

## ✅ Step 5: Test Your Bot

1. **Find your bot** on Telegram (search for the username you created)
2. **Send** `/start`
3. **Try chatting**:
   - "Hello!"
   - "What's 2+2?"
   - "Explain Docker in simple terms"

---

## 🎛️ Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_TOKEN` | ✅ Yes | - | Your Telegram bot token from @BotFather |
| `GROQ_API_KEY` | ✅ Yes | - | Your Groq API key |
| `GROQ_MODEL` | ❌ No | `llama-3.3-70b-versatile` | Groq model to use |
| `ALLOWED_USER_IDS` | ❌ No | (anyone) | Comma-separated list of allowed user IDs |

### Available Groq Models

- `llama-3.3-70b-versatile` - Best overall (recommended)
- `llama-3.1-70b-versatile` - Good alternative
- `llama-3.1-8b-instant` - Fastest responses
- `mixtral-8x7b-32768` - Good for longer context

---

## 🐛 Troubleshooting

### Bot doesn't respond

**Check logs on Claw Cloud:**
1. Go to your app dashboard
2. Click "Logs"
3. Look for errors

**Common issues:**
- ❌ Wrong `TELEGRAM_TOKEN` - Double-check token from @BotFather
- ❌ Wrong `GROQ_API_KEY` - Verify key at https://console.groq.com
- ❌ Bot not started - Check if container is running

### "Not authorized" message

- Make sure your user ID is in `ALLOWED_USER_IDS`
- Or remove `ALLOWED_USER_IDS` to allow anyone

### Rate limit errors

- Groq free tier: 14,400 requests/day
- If exceeded, wait 24 hours or upgrade to paid tier

---

## 💡 Tips & Best Practices

1. **Security**: Always set `ALLOWED_USER_IDS` to restrict access
2. **Model Choice**: Use `llama-3.3-70b-versatile` for best quality
3. **Conversation History**: Bot remembers last 10 messages per user
4. **Clear History**: Use `/clear` to reset conversation
5. **Monitoring**: Check logs regularly on Claw Cloud

---

## 🆓 Free Tier Limits

### Groq
- 14,400 requests/day
- No credit card required

### Claw Cloud
- 512MB RAM
- 1 vCPU
- Always running
- Free forever

---

## 🎉 You're Done!

You now have a fully functional AI Telegram bot running 24/7 for free! 🚀

**Next Steps:**
- Customize the bot responses
- Add custom commands
- Integrate with other APIs
- Monitor usage on Groq dashboard

---

## 📚 Additional Resources

- **Groq Documentation**: https://console.groq.com/docs
- **Telegram Bot API**: https://core.telegram.org/bots/api
- **python-telegram-bot**: https://python-telegram-bot.org/
- **Claw Cloud Docs**: https://docs.claw.cloud/

---

**Need help?** Check the logs on Claw Cloud or review the Groq API documentation!
