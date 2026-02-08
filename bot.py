import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
ALLOWED_USER_IDS = os.getenv('ALLOWED_USER_IDS', '').split(',')
DEFAULT_PROVIDER = os.getenv('DEFAULT_PROVIDER', 'groq')

# Provider API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# ============================================================================
# PROVIDER ABSTRACTION LAYER
# ============================================================================

class AIProvider(ABC):
    """Base class for AI providers"""
    
    @abstractmethod
    def chat(self, messages: List[Dict], model: Optional[str] = None) -> str:
        """Send chat messages and get response"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get provider name"""
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get default model name"""
        pass


class GroqProvider(AIProvider):
    """Groq AI Provider"""
    
    def __init__(self, api_key: str):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.default_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
    
    def chat(self, messages: List[Dict], model: Optional[str] = None) -> str:
        model = model or self.default_model
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    
    def get_available_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B (Best Overall)"},
            {"id": "llama-3.1-70b-versatile", "name": "Llama 3.1 70B"},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B (Fastest)"},
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B (Long Context)"},
            {"id": "gemma2-9b-it", "name": "Gemma 2 9B"},
        ]
    
    def get_name(self) -> str:
        return "Groq"
    
    def get_default_model(self) -> str:
        return self.default_model


class GeminiProvider(AIProvider):
    """Google Gemini Provider"""
    
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.default_model = "gemini-2.5-flash"
    
    def chat(self, messages: List[Dict], model: Optional[str] = None) -> str:
        model_name = model or self.default_model
        model = self.genai.GenerativeModel(model_name)
        
        # Convert messages to Gemini format
        chat_history = []
        for msg in messages[:-1]:  # All except last
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [msg["content"]]})
        
        # Start chat with history
        chat = model.start_chat(history=chat_history)
        
        # Send last message
        response = chat.send_message(messages[-1]["content"])
        return response.text
    
    def get_available_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro (Best Quality)"},
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash (Balanced)"},
            {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash-Lite (Fastest)"},
        ]
    
    def get_name(self) -> str:
        return "Gemini"
    
    def get_default_model(self) -> str:
        return self.default_model


class OpenRouterProvider(AIProvider):
    """OpenRouter Provider"""
    
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.default_model = "openrouter/free"
    
    def chat(self, messages: List[Dict], model: Optional[str] = None) -> str:
        model = model or self.default_model
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    
    def get_available_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "openrouter/free", "name": "Auto-select Free Model"},
            {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3 70B"},
            {"id": "google/gemma-3-27b:free", "name": "Gemma 3 27B"},
            {"id": "deepseek/deepseek-r1:free", "name": "DeepSeek R1"},
            {"id": "mistralai/mistral-small-3.1-24b:free", "name": "Mistral Small 3.1"},
        ]
    
    def get_name(self) -> str:
        return "OpenRouter"
    
    def get_default_model(self) -> str:
        return self.default_model


# ============================================================================
# PROVIDER MANAGER
# ============================================================================

class ProviderManager:
    """Manages multiple AI providers"""
    
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers based on API keys"""
        if GROQ_API_KEY:
            try:
                self.providers['groq'] = GroqProvider(GROQ_API_KEY)
                logger.info("✅ Groq provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq: {e}")
        
        if GEMINI_API_KEY:
            try:
                self.providers['gemini'] = GeminiProvider(GEMINI_API_KEY)
                logger.info("✅ Gemini provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini: {e}")
        
        if OPENROUTER_API_KEY:
            try:
                self.providers['openrouter'] = OpenRouterProvider(OPENROUTER_API_KEY)
                logger.info("✅ OpenRouter provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenRouter: {e}")
        
        if not self.providers:
            raise ValueError("No AI providers available! Please set at least one API key.")
    
    def get_provider(self, provider_name: str) -> Optional[AIProvider]:
        """Get provider by name"""
        return self.providers.get(provider_name.lower())
    
    def list_providers(self) -> List[str]:
        """List available provider names"""
        return list(self.providers.keys())
    
    def get_default_provider(self) -> str:
        """Get default provider name"""
        if DEFAULT_PROVIDER in self.providers:
            return DEFAULT_PROVIDER
        return self.list_providers()[0]  # Return first available


# Initialize provider manager
provider_manager = ProviderManager()

# Store user sessions (in-memory, resets on restart)
user_sessions = {}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_user_session(user_id: str) -> Dict:
    """Get or create user session"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "provider": provider_manager.get_default_provider(),
            "model": None,  # None means use provider default
            "history": []
        }
    return user_sessions[user_id]


def is_user_allowed(user_id: str) -> bool:
    """Check if user is allowed to use the bot"""
    return not ALLOWED_USER_IDS[0] or user_id in ALLOWED_USER_IDS


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    user_id = str(update.effective_user.id)
    
    # Check if user is allowed
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    session = get_user_session(user_id)
    provider = provider_manager.get_provider(session["provider"])
    available_providers = ", ".join(provider_manager.list_providers())
    
    await update.message.reply_text(
        f"🤖 Hello! I'm your Multi-Provider AI assistant.\n\n"
        f"📡 Current Provider: **{provider.get_name()}**\n"
        f"🔧 Available Providers: {available_providers}\n\n"
        f"Just send me a message and I'll respond!\n\n"
        f"**Commands:**\n"
        f"/provider - Switch AI provider\n"
        f"/models - List available models\n"
        f"/model - Switch model\n"
        f"/clear - Clear conversation history\n"
        f"/help - Show help",
        parse_mode='Markdown'
    )

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history."""
    user_id = str(update.effective_user.id)
    
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    session = get_user_session(user_id)
    session["history"] = []
    await update.message.reply_text("🗑️ Conversation history cleared!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message."""
    if not is_user_allowed(str(update.effective_user.id)):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    await update.message.reply_text(
        "💡 **How to use:**\n\n"
        "Just send me any message and I'll respond using your selected AI provider!\n\n"
        "**Provider Management:**\n"
        "• `/provider` - Show current provider and available options\n"
        "• `/provider <name>` - Switch to a different provider (groq/gemini/openrouter)\n"
        "• `/models` - List available models for current provider\n"
        "• `/model <name>` - Switch to a specific model\n\n"
        "**Other Commands:**\n"
        "• `/clear` - Clear conversation history\n"
        "• `/help` - Show this help\n\n"
        "**Examples:**\n"
        "• What's the weather like?\n"
        "• Explain quantum physics\n"
        "• Write a Python function\n"
        "• Tell me a joke",
        parse_mode='Markdown'
    )

async def provider_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch AI provider or show current provider."""
    user_id = str(update.effective_user.id)
    
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    session = get_user_session(user_id)
    
    # If no argument, show current provider and available options
    if not context.args:
        current_provider = provider_manager.get_provider(session["provider"])
        available = ", ".join(provider_manager.list_providers())
        await update.message.reply_text(
            f"📡 **Current Provider:** {current_provider.get_name()}\n"
            f"🔧 **Available Providers:** {available}\n\n"
            f"Use `/provider <name>` to switch.\n"
            f"Example: `/provider gemini`",
            parse_mode='Markdown'
        )
        return
    
    # Switch provider
    new_provider_name = context.args[0].lower()
    new_provider = provider_manager.get_provider(new_provider_name)
    
    if not new_provider:
        available = ", ".join(provider_manager.list_providers())
        await update.message.reply_text(
            f"❌ Provider '{new_provider_name}' not found.\n"
            f"Available providers: {available}"
        )
        return
    
    session["provider"] = new_provider_name
    session["model"] = None  # Reset to default model
    await update.message.reply_text(
        f"✅ Switched to **{new_provider.get_name()}**!\n"
        f"Using default model: {new_provider.get_default_model()}",
        parse_mode='Markdown'
    )

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List available models for current provider."""
    user_id = str(update.effective_user.id)
    
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    session = get_user_session(user_id)
    provider = provider_manager.get_provider(session["provider"])
    models = provider.get_available_models()
    
    current_model = session["model"] or provider.get_default_model()
    
    model_list = "\n".join([
        f"• `{m['id']}` - {m['name']}" + (" ✓" if m['id'] == current_model else "")
        for m in models
    ])
    
    await update.message.reply_text(
        f"🤖 **Available Models for {provider.get_name()}:**\n\n"
        f"{model_list}\n\n"
        f"Current: `{current_model}`\n\n"
        f"Use `/model <id>` to switch.",
        parse_mode='Markdown'
    )

async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch model within current provider."""
    user_id = str(update.effective_user.id)
    
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    session = get_user_session(user_id)
    provider = provider_manager.get_provider(session["provider"])
    
    # If no argument, show current model
    if not context.args:
        current_model = session["model"] or provider.get_default_model()
        await update.message.reply_text(
            f"🤖 **Current Model:** `{current_model}`\n\n"
            f"Use `/models` to see available models.",
            parse_mode='Markdown'
        )
        return
    
    # Switch model
    new_model = " ".join(context.args)  # Support model names with spaces
    available_models = provider.get_available_models()
    model_ids = [m['id'] for m in available_models]
    
    if new_model not in model_ids:
        await update.message.reply_text(
            f"❌ Model '{new_model}' not found for {provider.get_name()}.\n"
            f"Use `/models` to see available models."
        )
        return
    
    session["model"] = new_model
    await update.message.reply_text(
        f"✅ Switched to model: `{new_model}`",
        parse_mode='Markdown'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages."""
    user_id = str(update.effective_user.id)
    
    # Check if user is allowed
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    user_message = update.message.text
    session = get_user_session(user_id)
    
    # Add user message to history
    session["history"].append({
        "role": "user",
        "content": user_message
    })
    
    # Keep only last 10 messages to avoid token limits
    if len(session["history"]) > 10:
        session["history"] = session["history"][-10:]
    
    try:
        # Send typing indicator
        await update.message.chat.send_action(action="typing")
        
        # Get current provider
        provider = provider_manager.get_provider(session["provider"])
        
        # Call AI provider
        bot_response = provider.chat(
            messages=session["history"],
            model=session["model"]
        )
        
        # Add bot response to history
        session["history"].append({
            "role": "assistant",
            "content": bot_response
        })
        
        # Send response
        await update.message.reply_text(bot_response)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        provider_name = provider_manager.get_provider(session["provider"]).get_name()
        await update.message.reply_text(
            f"❌ Error with {provider_name}: {str(e)}\n\n"
            f"Try:\n"
            f"• `/clear` to reset conversation\n"
            f"• `/provider` to switch to another provider"
        )

def main():
    """Start the bot."""
    # Validate environment variables
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is required!")
    
    # Check if at least one provider API key is set
    if not (GROQ_API_KEY or GEMINI_API_KEY or OPENROUTER_API_KEY):
        raise ValueError(
            "At least one AI provider API key is required!\n"
            "Set one of: GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY"
        )
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("provider", provider_command))
    application.add_handler(CommandHandler("models", models_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    logger.info("🚀 Multi-Provider AI Bot started!")
    logger.info(f"📡 Available providers: {', '.join(provider_manager.list_providers())}")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()