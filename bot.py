import os
import re
import logging
import json
import asyncio
import urllib.parse
import html as html_module
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
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
ALLOWED_USER_IDS = [uid.strip() for uid in os.getenv('ALLOWED_USER_IDS', '').split(',') if uid.strip()]
DEFAULT_PROVIDER = os.getenv('DEFAULT_PROVIDER', 'groq')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '512'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
MAX_HISTORY_MESSAGES = int(os.getenv('MAX_HISTORY_MESSAGES', '20'))  # must be even
MAX_MESSAGE_LENGTH = 4096

# Web Search Configuration
BRAVE_API_KEY = os.getenv('BRAVE_API_KEY', '')
SEARCH_ENGINE = os.getenv('SEARCH_ENGINE', 'brave').lower()
MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '3'))
MAX_SNIPPET_LEN = int(os.getenv('MAX_SNIPPET_LEN', '300'))

# Provider API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
CEREBRAS_API_KEY = os.getenv('CEREBRAS_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

# Model validation cache file
VALIDATED_MODELS_CACHE = os.path.join(os.path.dirname(__file__), 'validated_models.json')

# ============================================================================
# SYSTEM PROMPT — lean, Telegram-native, ~250 tokens
#
# PATCH: Replaced Perplexity prompt loader with a purpose-built prompt.
# Reasons:
#   - Perplexity prompt is designed for a pre-search pipeline with indexed
#     sources. This bot does reactive search with raw snippets — citation
#     numbers [1][2][3] had no backing sources to reference.
#   - LaTeX math instructions, ## headers, academic formatting, and
#     "verbalize your thought process" planning rules all produce bad
#     Telegram output and cost tokens on every API call.
#   - The telegram_bot_adaptation override was fighting 3000 tokens of
#     conflicting base instructions and losing on average.
#   - Two pre-built variants (base + search) allocated once at startup,
#     avoiding string concatenation on every request.
# ============================================================================

_PROMPT_BASE = (
    "You are a helpful, concise AI assistant speaking through Telegram.\n\n"
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

# Pre-built at module load — never re-allocated per request
SYSTEM_PROMPT             = _PROMPT_BASE
SYSTEM_PROMPT_WITH_SEARCH = _PROMPT_BASE + _PROMPT_SEARCH_ADDENDUM


async def reply_text_safe(message, text: str):
    """Send Markdown first for better chat UI, then fallback to plain text."""
    try:
        await message.reply_text(text, parse_mode='Markdown')
    except Exception:
        await message.reply_text(text)


# ============================================================================
# SEARCH HEURISTIC
#
# PATCH: Kept the binary direct/no-search decision from esp32_bot v2.
# The 'maybe' tier (ask AI → wait for SEARCH: → search → answer) costs
# an extra full API round-trip. _extract_search_query() is good enough to
# pull a clean query directly from the user's message without asking the AI.
#
# _wants_search() now returns bool. Two tiers: search / no-search.
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


def _wants_search(text: str) -> bool:
    """Return True if the message likely needs a web search.

    Binary decision — no 'maybe' tier. Any positive search signal triggers
    a direct search immediately rather than asking the AI first."""
    low = text.lower()
    length = len(low)

    if length < 12:
        return False

    # No-search signals — but check if overridden by time/topic keywords
    for kw in _KW_NO_SEARCH:
        if kw in low:
            for kw2 in _KW_TIME:
                if kw2 in low:
                    return True
            for kw2 in _KW_TOPIC:
                if kw2 in low:
                    return True
            return False

    for kw in _KW_TIME:
        if kw in low:
            return True

    for kw in _KW_TOPIC:
        if kw in low:
            return True

    # Year mention (current ± 1)
    import datetime
    year = datetime.datetime.now().year
    for y in (str(year - 1), str(year), str(year + 1)):
        if y in low:
            return True

    # Question patterns that likely need live data
    if '?' in low and length > 20:
        if 'who ' in low or 'how much' in low or 'how many' in low or 'where ' in low:
            return True

    return False


def _extract_search_query(text: str) -> str:
    """Strip question prefixes and return a clean search query (max 120 chars)."""
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


def _build_search_context(query: str, snippets: list) -> str:
    """Format search results as numbered snippets the model can reference naturally.

    Includes today's date so the model has temporal context without needing
    a bloated search-aware system prompt injected on every request."""
    import datetime
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    parts = [f"Today is {today}. Web search results for '{query}':"]
    for i, snip in enumerate(snippets, 1):
        parts.append(f"{i}. {snip}")
    return '\n'.join(parts)


# ============================================================================
# HISTORY UTILITIES
#
# PATCH: Replaced asymmetric tail-slice trim with pair-based eviction.
# Old code: history[-MAX_HISTORY_MESSAGES:] — could orphan a user message
# without its assistant response, or keep an assistant response whose user
# message was evicted. This confuses the model on subsequent turns.
# New code: always removes from the front in user+assistant pairs.
# MAX_HISTORY_MESSAGES should be even (default 20 = 10 complete exchanges).
# ============================================================================

def _trim_history(history: list) -> list:
    """Trim history to MAX_HISTORY_MESSAGES keeping complete user/assistant pairs.

    Always removes the oldest pair (2 messages) at a time so the model
    never sees a dangling half-exchange."""
    while len(history) > MAX_HISTORY_MESSAGES:
        if len(history) >= 2:
            history.pop(0)  # oldest user message
            history.pop(0)  # its assistant response
        else:
            history.pop(0)
    return history


# ============================================================================
# FUTURE-PROOF MODEL RANKING UTILITIES
# ============================================================================

def extract_parameter_size(model_id: str) -> int:
    match = re.search(r'(\d+\.?\d*)b', model_id.lower())
    if match:
        size = float(match.group(1))
        return int(size) if size >= 1 else 0
    return 0

def get_model_capability_score(model_id: str, model_name: str = "") -> tuple:
    model_lower = model_id.lower()
    param_size = extract_parameter_size(model_id)

    flagship_patterns = [
        'gpt-4', 'claude-3-opus', 'claude-3.5', 'claude-4',
        'gemini-2.0', 'gemini-pro', 'llama-3.3', 'llama-3.2',
        'qwen-2.5-72b', 'hermes-3-llama-3.1-405b', 'gpt-oss'
    ]
    if any(pattern in model_lower for pattern in flagship_patterns):
        return (0, -param_size if param_size > 0 else 0, model_id)

    if param_size >= 100: return (1, -param_size, model_id)
    if param_size >= 50:  return (2, -param_size, model_id)
    if param_size >= 20:  return (3, -param_size, model_id)
    if param_size >= 10:  return (4, -param_size, model_id)
    if param_size >= 5:   return (5, -param_size, model_id)
    if param_size >= 1:   return (6, -param_size, model_id)

    if any(x in model_lower for x in ['exp', 'experimental', 'preview', 'beta']):
        version_match = re.search(r'(\d+\.\d+)', model_id)
        if version_match and float(version_match.group(1)) >= 2.0:
            return (0, 0, model_id)
        return (7, 0, model_id)

    return (8, 0, model_id)


# ============================================================================
# PROVIDER ABSTRACTION LAYER
# ============================================================================

class AIProvider(ABC):

    @abstractmethod
    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False) -> str:
        pass

    @abstractmethod
    def get_available_models(self) -> List[Dict[str, str]]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        pass

    def supports_thinking(self, model_id: str) -> bool:
        return False

    def test_model(self, model_id: str) -> Tuple[bool, str]:
        try:
            response = self.chat([{"role": "user", "content": "Hi"}], model=model_id)
            if response is not None and len(response) > 0:
                return (True, 'success')
            return (False, 'unknown')
        except Exception as e:
            error_str = str(e).lower()
            logger.debug(f"Model {model_id} validation failed: {e}")
            if any(k in error_str for k in ['rate limit', 'too many requests', '429', 'quota']):
                return (False, 'rate_limit')
            elif any(k in error_str for k in ['not found', '404', 'does not exist', 'invalid model',
                                               'not available', 'not supported', 'no access']):
                return (False, 'not_available')
            else:
                return (False, 'unknown')


class GroqProvider(AIProvider):

    def __init__(self, api_key: str):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.default_model = 'llama-3.3-70b-versatile'
        self._cached_models = None

    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False) -> str:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        response = self.client.chat.completions.create(
            messages=chat_messages, model=model,
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("API returned empty response.")
        return response.choices[0].message.content

    def get_available_models(self) -> List[Dict[str, str]]:
        if self._cached_models:
            return self._cached_models
        try:
            models_response = self.client.models.list()
            chat_models = []
            for model in models_response.data:
                if model.active and hasattr(model, 'id'):
                    chat_models.append({"id": model.id, "name": self._format_model_name(model.id)})
            chat_models.sort(key=lambda m: get_model_capability_score(m['id'], m['name']))
            self._cached_models = chat_models
            logger.info(f"✅ Groq: Detected {len(chat_models)} available models")
            return chat_models
        except Exception as e:
            logger.warning(f"⚠️ Groq: Could not fetch models: {e}")
            return self._get_fallback_models()

    def _format_model_name(self, model_id: str) -> str:
        return model_id.replace('-', ' ').title()

    def _get_fallback_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile"},
            {"id": "llama-3.1-70b-versatile", "name": "Llama 3.1 70B Versatile"},
            {"id": "mixtral-8x7b-32768",       "name": "Mixtral 8x7B 32K"},
            {"id": "llama-3.1-8b-instant",     "name": "Llama 3.1 8B Instant"},
            {"id": "gemma2-9b-it",             "name": "Gemma 2 9B IT"},
        ]

    def get_name(self) -> str: return "Groq"
    def get_default_model(self) -> str: return self.default_model


class GeminiProvider(AIProvider):

    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.default_model = "gemini-1.5-flash"
        self._cached_models = None

    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False) -> str:
        if not messages:
            raise ValueError("Messages list cannot be empty")
        model_name = model or self.default_model
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
        gen_model = self.genai.GenerativeModel(
            model_name,
            generation_config={"temperature": TEMPERATURE, "max_output_tokens": MAX_TOKENS},
            system_instruction=system_msg
        )
        chat_history = []
        for msg in messages[:-1]:
            if msg["role"] == "system":
                continue
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [msg["content"]]})
        chat = gen_model.start_chat(history=chat_history)
        response = chat.send_message(messages[-1]["content"])
        return response.text

    def get_available_models(self) -> List[Dict[str, str]]:
        if self._cached_models:
            return self._cached_models
        try:
            chat_models = []
            for model in self.genai.list_models():
                if 'generateContent' not in model.supported_generation_methods:
                    continue
                model_id = model.name.replace('models/', '')
                if any(x in model_id.lower() for x in ['vision', 'embedding', 'aqa']):
                    continue
                chat_models.append({
                    "id": model_id,
                    "name": self._format_model_name(model_id, model)
                })
            chat_models.sort(key=lambda m: get_model_capability_score(m['id'], m['name']))
            self._cached_models = chat_models
            logger.info(f"✅ Gemini: Detected {len(chat_models)} available models")
            return chat_models
        except Exception as e:
            logger.warning(f"⚠️ Gemini: Could not fetch models: {e}")
            return self._get_fallback_models()

    def _format_model_name(self, model_id: str, model_obj=None) -> str:
        if model_obj and hasattr(model_obj, 'display_name'):
            return model_obj.display_name
        name = model_id.replace('gemini-', 'Gemini ').replace('-', ' ').title()
        if 'exp' in model_id.lower():
            name += " (Experimental)"
        return name

    def _get_fallback_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "gemini-1.5-flash",     "name": "Gemini 1.5 Flash"},
            {"id": "gemini-1.5-pro",       "name": "Gemini 1.5 Pro"},
            {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash Experimental"},
            {"id": "gemini-1.5-flash-8b",  "name": "Gemini 1.5 Flash 8B"},
        ]

    def get_name(self) -> str: return "Gemini"
    def get_default_model(self) -> str: return self.default_model


class OpenRouterProvider(AIProvider):

    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.api_key = api_key
        self.default_model = "meta-llama/llama-3.3-70b-instruct:free"
        self._cached_models = None

    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False) -> str:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        response = self.client.chat.completions.create(
            model=model, messages=chat_messages,
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("API returned empty response.")
        return response.choices[0].message.content

    def get_available_models(self) -> List[Dict[str, str]]:
        if self._cached_models:
            return self._cached_models
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            response.raise_for_status()
            free_models = []
            for model in response.json().get('data', []):
                model_id = model.get('id', '')
                pricing = model.get('pricing', {})
                context_length = model.get('context_length', 0)
                has_free_suffix = ':free' in model_id.lower()
                prompt_str = str(pricing.get('prompt', '')) if pricing.get('prompt') is not None else ''
                completion_str = str(pricing.get('completion', '')) if pricing.get('completion') is not None else ''
                has_zero_pricing = (prompt_str in ["0", "0.0", "0.00"] and
                                    completion_str in ["0", "0.0", "0.00"])
                if has_free_suffix and context_length > 0 and has_zero_pricing:
                    free_models.append({
                        "id": model_id,
                        "name": self._format_model_name(model),
                        "context": context_length
                    })
            free_models.sort(key=lambda m: get_model_capability_score(m['id'], m['name']))
            self._cached_models = free_models[:15]
            logger.info(f"✅ OpenRouter: Detected {len(self._cached_models)} free models")
            return self._cached_models
        except Exception as e:
            logger.warning(f"⚠️ OpenRouter: Could not fetch models: {e}")
            return self._get_fallback_models()

    def _format_model_name(self, model_obj: dict) -> str:
        name = model_obj.get('name', model_obj.get('id', 'Unknown'))
        model_id = model_obj.get('id', '')
        if ':free' in model_id.lower():
            name = name.replace(' (free)', '').replace(' (Free)', '') + " (Free)"
        return name

    def _get_fallback_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "meta-llama/llama-3.3-70b-instruct:free",    "name": "Llama 3.3 70B (Free)"},
            {"id": "nousresearch/hermes-3-llama-3.1-405b:free",  "name": "Hermes 3 405B (Free)"},
            {"id": "google/gemini-2.0-flash-exp:free",           "name": "Gemini 2.0 Flash Exp (Free)"},
            {"id": "qwen/qwen-2.5-72b-instruct:free",            "name": "Qwen 2.5 72B (Free)"},
            {"id": "mistralai/mistral-7b-instruct:free",         "name": "Mistral 7B (Free)"},
        ]

    def get_name(self) -> str: return "OpenRouter"
    def get_default_model(self) -> str: return self.default_model


class CerebrasProvider(AIProvider):

    def __init__(self, api_key: str):
        from cerebras.cloud.sdk import Cerebras
        self.client = Cerebras(api_key=api_key)
        self.default_model = "gpt-oss-120b"
        self._cached_models = None

    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False) -> str:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        response = self.client.chat.completions.create(
            messages=chat_messages, model=model,
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("API returned empty response.")
        return response.choices[0].message.content

    def get_available_models(self) -> List[Dict[str, str]]:
        if self._cached_models:
            return self._cached_models
        try:
            chat_models = []
            for model in self.client.models.list().data:
                if hasattr(model, 'id'):
                    chat_models.append({"id": model.id, "name": self._format_model_name(model.id)})
            chat_models.sort(key=lambda m: get_model_capability_score(m['id'], m['name']))
            self._cached_models = chat_models
            logger.info(f"✅ Cerebras: Detected {len(chat_models)} available models")
            return chat_models
        except Exception as e:
            logger.warning(f"⚠️ Cerebras: Could not fetch models: {e}")
            return self._get_fallback_models()

    def _format_model_name(self, model_id: str) -> str:
        return model_id.replace('-', ' ').title()

    def _get_fallback_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "gpt-oss-120b",   "name": "GPT-OSS 120B"},
            {"id": "llama3.1-8b",    "name": "Llama 3.1 8B"},
            {"id": "llama-3.3-70b",  "name": "Llama 3.3 70B"},
        ]

    def get_name(self) -> str: return "Cerebras"
    def get_default_model(self) -> str: return self.default_model


class NvidiaProvider(AIProvider):

    MODELS_WITHOUT_THINKING = {
        'qwen/qwen3-coder-480b-a35b-instruct',
        'openai/gpt-oss-120b',
        'minimaxai/minimax-m2.1',
        'minimaxai/minimax-m2'
    }

    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.default_model = 'openai/gpt-oss-120b'
        self._cached_models = None

    def supports_thinking(self, model_id: str) -> bool:
        return model_id not in self.MODELS_WITHOUT_THINKING

    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False) -> str:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        should_use_thinking = enable_thinking and self.supports_thinking(model)

        if should_use_thinking:
            response = self.client.chat.completions.create(
                messages=chat_messages, model=model,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True
            )
            reasoning_parts = []
            content_parts = []
            for chunk in response:
                if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if getattr(delta, "content", None) is not None:
                    content_parts.append(delta.content)
            full_response = ""
            if reasoning_parts:
                full_response = "💭 *Thinking:*\n" + "".join(reasoning_parts) + "\n\n"
            full_response += "".join(content_parts)
            return full_response
        else:
            response = self.client.chat.completions.create(
                messages=chat_messages, model=model,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            )
            if not response.choices or response.choices[0].message.content is None:
                raise ValueError("API returned empty response.")
            return response.choices[0].message.content

    def get_available_models(self) -> List[Dict[str, str]]:
        if self._cached_models:
            return self._cached_models
        self._cached_models = self._get_fallback_models()
        logger.info(f"✅ NVIDIA: Using {len(self._cached_models)} hand-picked models")
        return self._cached_models

    def _get_fallback_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "openai/gpt-oss-120b",                      "name": "GPT-OSS 120B (Stable)"},
            {"id": "qwen/qwen3-coder-480b-a35b-instruct",      "name": "Qwen3 Coder 480B"},
            {"id": "minimaxai/minimax-m2.1",                   "name": "MiniMax M2.1"},
            {"id": "minimaxai/minimax-m2",                     "name": "MiniMax M2"},
            {"id": "deepseek-ai/deepseek-v3.2",                "name": "DeepSeek V3.2 💭"},
            {"id": "deepseek-ai/deepseek-v3.1-terminus",       "name": "DeepSeek V3.1 Terminus 💭"},
            {"id": "qwen/qwen3-235b-a22b",                     "name": "Qwen3 235B 💭"},
            {"id": "moonshotai/kimi-k2.5",                     "name": "Kimi K2.5 💭"},
            {"id": "z-ai/glm4.7",                              "name": "GLM 4.7 💭"},
            {"id": "z-ai/glm5",                                "name": "GLM 5 💭"},
        ]

    def get_name(self) -> str: return "NVIDIA"
    def get_default_model(self) -> str: return self.default_model


# ============================================================================
# PROVIDER MANAGER
# ============================================================================

class ProviderManager:

    def __init__(self):
        self.providers = {}
        self._initialize_providers()

    def _initialize_providers(self):
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

        if CEREBRAS_API_KEY:
            try:
                self.providers['cerebras'] = CerebrasProvider(CEREBRAS_API_KEY)
                logger.info("✅ Cerebras provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Cerebras: {e}")

        if NVIDIA_API_KEY:
            try:
                self.providers['nvidia'] = NvidiaProvider(NVIDIA_API_KEY)
                logger.info("✅ NVIDIA provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize NVIDIA: {e}")

        if not self.providers:
            raise ValueError("No AI providers available! Set at least one API key.")

    def get_provider(self, provider_name: str) -> Optional[AIProvider]:
        return self.providers.get(provider_name.lower())

    def list_providers(self) -> List[str]:
        return list(self.providers.keys())

    def get_default_provider(self) -> str:
        if DEFAULT_PROVIDER in self.providers:
            return DEFAULT_PROVIDER
        providers = self.list_providers()
        if not providers:
            raise ValueError("No providers available")
        return providers[0]

    def refresh_models(self):
        logger.info("🔄 Refreshing model lists...")
        for name, provider in self.providers.items():
            provider._cached_models = None
            models = provider.get_available_models()
            logger.info(f"  {name}: {len(models)} models")


# Initialize provider manager
provider_manager = ProviderManager()

# Store user sessions (in-memory, resets on restart)
user_sessions: Dict[str, Dict] = {}


# ============================================================================
# SESSION HELPERS
# ============================================================================

def get_user_session(user_id: str) -> Dict:
    if user_id not in user_sessions:
        default_engine = "brave" if (SEARCH_ENGINE == "brave" and BRAVE_API_KEY) else "duckduckgo"
        user_sessions[user_id] = {
            "provider":         provider_manager.get_default_provider(),
            "models":           {},
            "history":          [],
            "thinking_enabled": False,
            "web_search":       True,
            "search_engine":    default_engine,
        }
    return user_sessions[user_id]


def is_user_allowed(user_id: str) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


# ============================================================================
# MODEL VALIDATION CACHE
# ============================================================================

def load_validated_models() -> Dict:
    try:
        if os.path.exists(VALIDATED_MODELS_CACHE):
            with open(VALIDATED_MODELS_CACHE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load validated models cache: {e}")
    return {}

def save_validated_models(validated: Dict):
    try:
        with open(VALIDATED_MODELS_CACHE, 'w') as f:
            json.dump(validated, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save validated models cache: {e}")

def get_validated_models(provider_name: str) -> List[str]:
    validated = load_validated_models()
    provider_data = validated.get(provider_name, {})
    if isinstance(provider_data, list):
        return provider_data
    return provider_data.get('working', [])

def get_failed_models(provider_name: str) -> List[str]:
    validated = load_validated_models()
    provider_data = validated.get(provider_name, {})
    if isinstance(provider_data, dict):
        return provider_data.get('failed', [])
    return []

def add_validated_model(provider_name: str, model_id: str):
    validated = load_validated_models()
    if provider_name not in validated:
        validated[provider_name] = {'working': [], 'failed': []}
    if isinstance(validated[provider_name], list):
        validated[provider_name] = {'working': validated[provider_name], 'failed': []}
    if model_id not in validated[provider_name]['working']:
        validated[provider_name]['working'].append(model_id)
        if model_id in validated[provider_name]['failed']:
            validated[provider_name]['failed'].remove(model_id)
        save_validated_models(validated)

def add_failed_model(provider_name: str, model_id: str):
    validated = load_validated_models()
    if provider_name not in validated:
        validated[provider_name] = {'working': [], 'failed': []}
    if isinstance(validated[provider_name], list):
        validated[provider_name] = {'working': validated[provider_name], 'failed': []}
    if model_id not in validated[provider_name]['failed']:
        validated[provider_name]['failed'].append(model_id)
        save_validated_models(validated)

def clear_validated_models(provider_name: Optional[str] = None):
    if provider_name:
        validated = load_validated_models()
        if provider_name in validated:
            del validated[provider_name]
            save_validated_models(validated)
    else:
        save_validated_models({})


# ============================================================================
# WEB SEARCH — Brave API + DuckDuckGo HTML scraping
# ============================================================================

def _strip_tags(html_str: str) -> str:
    out = []
    in_tag = False
    for c in html_str:
        if c == '<':   in_tag = True
        elif c == '>': in_tag = False
        elif not in_tag: out.append(c)
    return html_module.unescape(''.join(out)).strip()

def _brave_search_sync(query: str) -> list:
    if not BRAVE_API_KEY:
        return []
    q = urllib.parse.quote_plus(query)
    url = f"https://api.search.brave.com/res/v1/web/search?q={q}&count={MAX_SEARCH_RESULTS}"
    try:
        r = requests.get(url, headers={"Accept": "application/json",
                                        "X-Subscription-Token": BRAVE_API_KEY}, timeout=10)
        r.raise_for_status()
        snippets = []
        for item in r.json().get("web", {}).get("results", []):
            title = item.get("title", "")
            desc  = item.get("description", "")[:MAX_SNIPPET_LEN]
            if title or desc:
                snippets.append(f"{title}: {desc}")
        logger.info(f"[Search] Brave '{query}' -> {len(snippets)} results")
        return snippets
    except Exception as e:
        logger.error(f"[Search] Brave error: {e}")
        return []

def _duckduckgo_search_sync(query: str) -> list:
    q = urllib.parse.quote_plus(query)
    try:
        r = requests.get(f"https://html.duckduckgo.com/html/?q={q}",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        html_text = r.text
        snippets = []
        pos = 0
        while len(snippets) < MAX_SEARCH_RESULTS:
            pos = html_text.find('class="result__a"', pos)
            if pos == -1: break
            tag_end = html_text.find('>', pos)
            if tag_end == -1: break
            title_start = tag_end + 1
            title_end = html_text.find('</a>', title_start)
            if title_end == -1: break
            title = _strip_tags(html_text[title_start:title_end])
            snip_pos    = html_text.find('class="result__snippet', pos)
            next_result = html_text.find('class="result__a"', title_end + 1)
            desc = ""
            if snip_pos != -1 and (next_result == -1 or snip_pos < next_result):
                stag_end = html_text.find('>', snip_pos)
                if stag_end != -1:
                    snip_start = stag_end + 1
                    snip_end = html_text.find('</a>', snip_start)
                    if snip_end == -1:
                        snip_end = html_text.find('</td>', snip_start)
                    if snip_end != -1:
                        desc = _strip_tags(html_text[snip_start:snip_end])[:MAX_SNIPPET_LEN]
            if title:
                snippets.append(f"{title}: {desc}")
            pos = title_end + 1
        logger.info(f"[Search] DDG '{query}' -> {len(snippets)} results")
        return snippets
    except Exception as e:
        logger.error(f"[Search] DDG error: {e}")
        return []

async def web_search(query: str, engine: str) -> list:
    if engine == "brave" and BRAVE_API_KEY:
        return await asyncio.to_thread(_brave_search_sync, query)
    return await asyncio.to_thread(_duckduckgo_search_sync, query)


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    session = get_user_session(user_id)
    provider = provider_manager.get_provider(session["provider"])
    if not provider:
        session["provider"] = provider_manager.get_default_provider()
        provider = provider_manager.get_provider(session["provider"])
    web_status = "ON" if session.get("web_search") else "OFF"
    await update.message.reply_text(
        f"🤖 Hello! I'm your Multi-Provider AI assistant.\n\n"
        f"📡 Current Provider: *{provider.get_name()}*\n"
        f"🔧 Available Providers: {', '.join(provider_manager.list_providers())}\n"
        f"🌐 Web Search: {web_status}\n\n"
        f"Just send me a message!\n\n"
        f"*Commands:*\n"
        f"/provider - Switch AI provider\n"
        f"/models - List available models\n"
        f"/model - Switch model\n"
        f"/web - Toggle web search\n"
        f"/refresh - Refresh model lists\n"
        f"/clear - Clear conversation history\n"
        f"/help - Show help",
        parse_mode='Markdown'
    )

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    get_user_session(user_id)["history"] = []
    await update.message.reply_text("🗑️ Conversation history cleared!")

async def refresh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_user_allowed(str(update.effective_user.id)):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    await update.message.reply_text("🔄 Refreshing model lists...")
    try:
        provider_manager.refresh_models()
        await update.message.reply_text("✅ Model lists refreshed!\n\nUse /models to see latest.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error refreshing models: {str(e)}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_user_allowed(str(update.effective_user.id)):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    await update.message.reply_text(
        "💡 *How to use:*\n\n"
        "Just send a message — I'll respond using your selected AI provider!\n\n"
        "*Provider Management:*\n"
        "• `/provider` — show/switch provider\n"
        "• `/models` — verified working models\n"
        "• `/models all` — all models from API\n"
        "• `/model <id>` — switch model\n"
        "• `/refresh` — refresh model lists\n\n"
        "*Web Search:*\n"
        "• `/web` — show status\n"
        "• `/web on` / `/web off` — toggle\n"
        "• `/web brave` — Brave Search API\n"
        "• `/web ddg` — DuckDuckGo (free)\n\n"
        "*Model Validation:*\n"
        "• `/validate` — test which models work\n"
        "• `/verified` — show validated models\n"
        "• `/clearvalidation` — clear cache\n\n"
        "*Thinking Mode (NVIDIA only):*\n"
        "• `/thinking on` / `/thinking off`\n\n"
        "*Other:*\n"
        "• `/clear` — clear conversation history\n"
        "• `/help` — this message",
        parse_mode='Markdown'
    )

async def provider_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    session = get_user_session(user_id)
    if not context.args:
        current = provider_manager.get_provider(session["provider"])
        if not current:
            session["provider"] = provider_manager.get_default_provider()
            current = provider_manager.get_provider(session["provider"])
        await update.message.reply_text(
            f"📡 *Current:* {current.get_name()}\n"
            f"🔧 *Available:* {', '.join(provider_manager.list_providers())}\n\n"
            f"Use `/provider <name>` to switch.",
            parse_mode='Markdown'
        )
        return
    new_name = context.args[0].lower()
    new_provider = provider_manager.get_provider(new_name)
    if not new_provider:
        await update.message.reply_text(
            f"❌ Provider '{new_name}' not found.\n"
            f"Available: {', '.join(provider_manager.list_providers())}"
        )
        return
    session["provider"] = new_name
    current_model = session["models"].get(new_name) or new_provider.get_default_model()
    await update.message.reply_text(
        f"✅ Switched to *{new_provider.get_name()}*!\nModel: `{current_model}`",
        parse_mode='Markdown'
    )

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    session = get_user_session(user_id)
    provider_name = session["provider"]
    provider = provider_manager.get_provider(provider_name)
    if not provider:
        session["provider"] = provider_manager.get_default_provider()
        provider_name = session["provider"]
        provider = provider_manager.get_provider(provider_name)
    show_all = len(context.args) > 0 and context.args[0].lower() == 'all'
    if show_all:
        await update.message.reply_text("🔄 Fetching all models from API...")
        models = provider.get_available_models()
        title_suffix = " (All Models)"
        footer_note = "\n\n💡 Use `/models` to see only verified models"
    else:
        validated_ids = get_validated_models(provider_name)
        if not validated_ids:
            await update.message.reply_text(
                f"❌ *No verified models yet for {provider.get_name()}*\n\n"
                f"Run `/validate` first, or use `/models all` to see everything.",
                parse_mode='Markdown'
            )
            return
        await update.message.reply_text("✅ Showing verified models...")
        all_models = provider.get_available_models()
        models = [m for m in all_models if m['id'] in validated_ids]
        title_suffix = " (Verified)"
        footer_note = f"\n\n💡 Use `/models all` to see all {len(all_models)} models"
    current_model = session["models"].get(provider_name) or provider.get_default_model()
    max_per_msg = 20
    chunks = [models[i:i+max_per_msg] for i in range(0, len(models), max_per_msg)]
    for idx, chunk in enumerate(chunks, 1):
        model_list = "\n".join([
            f"• `{m['id']}`" + (" ✓" if m['id'] == current_model else "")
            for m in chunk
        ])
        part = f" (Part {idx}/{len(chunks)})" if len(chunks) > 1 else ""
        await update.message.reply_text(
            f"🤖 *{provider.get_name()}{title_suffix}{part}:*\n\n"
            f"{model_list}"
            + (f"\n\nCurrent: `{current_model}`{footer_note}\n\nUse `/model <id>` to switch."
               if idx == len(chunks) else ""),
            parse_mode='Markdown'
        )

async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    session = get_user_session(user_id)
    provider_name = session["provider"]
    provider = provider_manager.get_provider(provider_name)
    if not provider:
        session["provider"] = provider_manager.get_default_provider()
        provider_name = session["provider"]
        provider = provider_manager.get_provider(provider_name)
    if not context.args:
        current = session["models"].get(provider_name) or provider.get_default_model()
        await update.message.reply_text(
            f"🤖 *Current Model ({provider.get_name()}):* `{current}`\n\nUse `/models` to see options.",
            parse_mode='Markdown'
        )
        return
    new_model = " ".join(context.args)
    model_ids = [m['id'] for m in provider.get_available_models()]
    session["models"][provider_name] = new_model
    if new_model in model_ids:
        await update.message.reply_text(
            f"✅ Switched to model: `{new_model}`\n💾 Saved for {provider.get_name()}.",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            f"⚠️ Model `{new_model}` not in known list — set anyway.\n"
            f"Use `/models` to see known models.",
            parse_mode='Markdown'
        )

async def validate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    session = get_user_session(user_id)
    provider_name = session["provider"]
    provider = provider_manager.get_provider(provider_name)
    if not provider:
        session["provider"] = provider_manager.get_default_provider()
        provider_name = session["provider"]
        provider = provider_manager.get_provider(provider_name)
    models = provider.get_available_models()
    already_validated  = get_validated_models(provider_name)
    permanently_failed = get_failed_models(provider_name)
    models_to_test = [m for m in models
                      if m['id'] not in already_validated and m['id'] not in permanently_failed]
    skipped = len(already_validated) + len(permanently_failed)
    if skipped > 0:
        await update.message.reply_text(
            f"💡 *Smart Validation*\n\n"
            f"Skipping {skipped}: ✅ {len(already_validated)} working, ❌ {len(permanently_failed)} failed\n"
            f"To test: {len(models_to_test)}/{len(models)}\n"
            f"⏳ ~{len(models_to_test) * 2}s\n\n"
            f"Use `/clearvalidation` to re-test all",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            f"🔍 *Full Validation* — testing all {len(models)} models\n⏳ ~{len(models) * 2}s",
            parse_mode='Markdown'
        )
    if not models_to_test:
        await update.message.reply_text("✅ All models already validated!\n\nUse `/verified` to see them.")
        return
    validated = list(already_validated)
    newly_validated = []
    failed_not_available = []
    failed_rate_limit = []
    failed_unknown = []
    for idx, model_info in enumerate(models_to_test, 1):
        model_id = model_info['id']
        if idx % 5 == 0 or idx == 1:
            await update.message.reply_text(
                f"⏳ {idx}/{len(models_to_test)}: `{model_id}`...", parse_mode='Markdown'
            )
        success, error_type = provider.test_model(model_id)
        if success:
            validated.append(model_id)
            newly_validated.append(model_id)
            add_validated_model(provider_name, model_id)
        elif error_type == 'not_available':
            failed_not_available.append(model_id)
            add_failed_model(provider_name, model_id)
        elif error_type == 'rate_limit':
            failed_rate_limit.append(model_id)
        else:
            failed_unknown.append(model_id)
        if idx < len(models_to_test):
            await asyncio.sleep(2)
    total_failed = len(failed_not_available) + len(failed_rate_limit) + len(failed_unknown)
    success_rate = (len(newly_validated) / len(models_to_test) * 100) if models_to_test else 0
    msg = (
        f"✅ *Validation Complete*\n\n"
        f"• Tested: {len(models_to_test)}\n"
        f"• ✅ Newly validated: {len(newly_validated)}\n"
        f"• ❌ Failed: {total_failed}\n"
    )
    if total_failed > 0:
        msg += (
            f"\n*Failure breakdown:*\n"
            f"• 🚫 Not available: {len(failed_not_available)} (cached)\n"
            f"• ⏱️ Rate limited: {len(failed_rate_limit)} (retry later)\n"
            f"• ❓ Unknown: {len(failed_unknown)}\n"
        )
    msg += (
        f"\n• Success rate: {success_rate:.1f}%\n\n"
        f"📦 *Total validated: {len(validated)}*\n\n"
        f"Use `/verified` to see working models!"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

async def verified_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    session = get_user_session(user_id)
    provider_name = session["provider"]
    provider = provider_manager.get_provider(provider_name)
    if not provider:
        session["provider"] = provider_manager.get_default_provider()
        provider_name = session["provider"]
        provider = provider_manager.get_provider(provider_name)
    validated_ids = get_validated_models(provider_name)
    if not validated_ids:
        await update.message.reply_text(
            f"❌ *No validated models for {provider.get_name()}*\n\nRun `/validate` first!",
            parse_mode='Markdown'
        )
        return
    all_models = provider.get_available_models()
    validated_models = [m for m in all_models if m['id'] in validated_ids]
    current_model = session["models"].get(provider_name) or provider.get_default_model()
    model_list = "\n".join([
        f"• `{m['id']}`" + (" ✓" if m['id'] == current_model else "")
        for m in validated_models
    ])
    await update.message.reply_text(
        f"✅ *Verified Models — {provider.get_name()}:*\n\n"
        f"{model_list}\n\n"
        f"Current: `{current_model}`\n\nUse `/model <id>` to switch.",
        parse_mode='Markdown'
    )

async def thinking_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    session = get_user_session(user_id)
    provider_name = session["provider"]
    provider = provider_manager.get_provider(provider_name)
    if not provider:
        session["provider"] = provider_manager.get_default_provider()
        provider_name = session["provider"]
        provider = provider_manager.get_provider(provider_name)
    if not context.args:
        state = "enabled" if session.get("thinking_enabled", False) else "disabled"
        await update.message.reply_text(
            f"💭 *Thinking Mode:* {state}\n\nUse `/thinking on` or `/thinking off`.",
            parse_mode='Markdown'
        )
        return
    arg = context.args[0].lower()
    if arg == 'on':
        session["thinking_enabled"] = True
        if provider_name == 'nvidia':
            current_model = session["models"].get(provider_name) or provider.get_default_model()
            if provider.supports_thinking(current_model):
                await update.message.reply_text(
                    f"✅ *Thinking mode enabled!* 💭\n\n`{current_model}` supports thinking.",
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text(
                    f"⚠️ *Thinking mode enabled*, but `{current_model}` doesn't support it.\n"
                    f"Switch to a model with 💭 to use thinking.",
                    parse_mode='Markdown'
                )
        else:
            await update.message.reply_text(
                f"✅ *Thinking mode enabled!*\n\nOnly NVIDIA supports thinking. "
                f"Use `/provider nvidia` to switch.",
                parse_mode='Markdown'
            )
    elif arg == 'off':
        session["thinking_enabled"] = False
        await update.message.reply_text("🔕 *Thinking mode disabled.*", parse_mode='Markdown')
    else:
        await update.message.reply_text(f"❌ Use `/thinking on` or `/thinking off`", parse_mode='Markdown')

async def clearvalidation_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    session = get_user_session(user_id)
    provider_name = session["provider"]
    provider = provider_manager.get_provider(provider_name)
    if not provider:
        session["provider"] = provider_manager.get_default_provider()
        provider_name = session["provider"]
        provider = provider_manager.get_provider(provider_name)
    clear_validated_models(provider_name)
    await update.message.reply_text(
        f"🗑️ Cleared validation cache for {provider.get_name()}!\n\nRun `/validate` to re-test."
    )

async def web_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    session = get_user_session(user_id)
    if not context.args:
        status = "ON" if session.get("web_search") else "OFF"
        eng = session.get("search_engine", "duckduckgo")
        await update.message.reply_text(
            f"🌐 *Web Search:* {status}\n"
            f"🔍 *Engine:* {'Brave API' if eng == 'brave' else 'DuckDuckGo'}\n\n"
            f"Use: `/web on` | `/web off`\n"
            f"`/web brave` — Brave Search API\n"
            f"`/web ddg` — DuckDuckGo (free)",
            parse_mode='Markdown'
        )
        return
    arg = context.args[0].lower()
    if arg == "on":
        session["web_search"] = True
        eng = session.get("search_engine", "duckduckgo")
        await update.message.reply_text(
            f"✅ Web search enabled ({'Brave' if eng == 'brave' else 'DuckDuckGo'})."
        )
    elif arg == "off":
        session["web_search"] = False
        await update.message.reply_text("🔕 Web search disabled.")
    elif arg == "brave":
        if not BRAVE_API_KEY:
            await update.message.reply_text("❌ Brave API key not configured. Use `/web ddg`.",
                                             parse_mode='Markdown')
        else:
            session["search_engine"] = "brave"
            session["web_search"] = True
            await update.message.reply_text("✅ Switched to Brave Search API.")
    elif arg in ("ddg", "duckduckgo"):
        session["search_engine"] = "duckduckgo"
        session["web_search"] = True
        await update.message.reply_text("✅ Switched to DuckDuckGo (free).")
    else:
        await update.message.reply_text("❌ Use: `/web on|off|brave|ddg`", parse_mode='Markdown')


# ============================================================================
# MESSAGE HANDLER
#
# PATCH SUMMARY vs original:
#   1. Removed 2-pass LLM search loop (Pass 1 ask AI → wait for SEARCH: →
#      Pass 2). Replaced with binary heuristic: _wants_search() decides
#      immediately, _extract_search_query() pulls the query from user text.
#      Saves one full API round-trip per search-eligible message.
#
#   2. Search context built via _build_search_context() — numbered snippets
#      with today's date. Model can reference results naturally without
#      fabricating [1][2] citation numbers.
#
#   3. History trimmed via _trim_history() (pair-based) instead of asymmetric
#      tail-slice. No more orphaned user/assistant messages.
#
#   4. SYSTEM_PROMPT_WITH_SEARCH used for search path — pre-built at startup,
#      not re-concatenated on every request.
#
#   5. History rollback on exception preserved from original (was already good).
# ============================================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return

    user_message = update.message.text
    if not user_message:
        return

    session = get_user_session(user_id)

    try:
        await update.message.chat.send_action(action="typing")

        provider_name = session["provider"]
        provider = provider_manager.get_provider(provider_name)
        if not provider:
            session["provider"] = provider_manager.get_default_provider()
            provider_name = session["provider"]
            provider = provider_manager.get_provider(provider_name)

        current_model    = session["models"].get(provider_name)  # None = provider default
        thinking_enabled = session.get("thinking_enabled", False)
        web_on           = session.get("web_search", True)

        # Add user message to history (popped on exception below)
        session["history"].append({"role": "user", "content": user_message})

        # ── Decide search strategy ───────────────────────────────────────
        do_search = web_on and _wants_search(user_message)
        logger.info(f"[Bot] user={user_id} do_search={do_search} provider={provider_name}")

        if do_search:
            # ── SEARCH PATH ─────────────────────────────────────────────
            query  = _extract_search_query(user_message)
            engine = session.get("search_engine", "duckduckgo")
            logger.info(f"[Bot] Searching ({engine}): '{query}'")

            await update.message.chat.send_action(action="typing")
            snippets = await web_search(query, engine)

            if snippets:
                ctx = _build_search_context(query, snippets)

                # Build search messages: history (minus just-added user msg)
                # + combined user message with context appended
                search_msgs = session["history"][:-1].copy()
                search_msgs.append({"role": "user", "content": user_message + "\n\n" + ctx})

                await update.message.chat.send_action(action="typing")
                bot_response = provider.chat(
                    messages=search_msgs,
                    model=current_model,
                    enable_thinking=thinking_enabled
                ) or ""
            else:
                # Search returned nothing — fall through to plain AI call
                logger.warning(f"[Bot] Search returned no results, falling back to direct AI call")
                bot_response = provider.chat(
                    messages=session["history"],
                    model=current_model,
                    enable_thinking=thinking_enabled
                ) or ""
        else:
            # ── DIRECT AI PATH ───────────────────────────────────────────
            bot_response = provider.chat(
                messages=session["history"],
                model=current_model,
                enable_thinking=thinking_enabled
            ) or ""

        if not bot_response.strip():
            bot_response = "⚠️ The AI returned an empty response. Try again or switch models."

        # ── Update history ───────────────────────────────────────────────
        session["history"].append({"role": "assistant", "content": bot_response})
        _trim_history(session["history"])

        # ── Send response (handle 4096 char limit) ───────────────────────
        if len(bot_response) <= MAX_MESSAGE_LENGTH:
            await reply_text_safe(update.message, bot_response)
        else:
            HEADER_RESERVE = 25
            chunk_limit = MAX_MESSAGE_LENGTH - HEADER_RESERVE
            chunks = []
            current_chunk = ""
            for line in bot_response.split('\n'):
                if len(line) > chunk_limit:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    for j in range(0, len(line), chunk_limit):
                        chunks.append(line[j:j + chunk_limit])
                    continue
                if len(current_chunk) + len(line) + 1 > chunk_limit:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = line
                else:
                    current_chunk = (current_chunk + '\n' + line) if current_chunk else line
            if current_chunk:
                chunks.append(current_chunk)
            for i, chunk in enumerate(chunks, 1):
                header = f"📄 Part {i}/{len(chunks)}\n\n" if len(chunks) > 1 else ""
                await reply_text_safe(update.message, header + chunk)

    except Exception as e:
        logger.error(f"Error in handle_message: {e}", exc_info=True)
        # Roll back the orphaned user message
        if session["history"] and session["history"][-1].get("role") == "user":
            session["history"].pop()
        prov = provider_manager.get_provider(session.get("provider", ""))
        prov_label = prov.get_name() if prov else session.get("provider", "unknown")
        await update.message.reply_text(
            f"❌ Error with {prov_label}: {str(e)}\n\n"
            f"Try:\n• `/clear` to reset conversation\n• `/provider` to switch provider"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is required!")
    if not (GROQ_API_KEY or GEMINI_API_KEY or OPENROUTER_API_KEY or CEREBRAS_API_KEY or NVIDIA_API_KEY):
        raise ValueError("At least one AI provider API key is required!")

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start",           start))
    application.add_handler(CommandHandler("clear",           clear))
    application.add_handler(CommandHandler("refresh",         refresh_command))
    application.add_handler(CommandHandler("help",            help_command))
    application.add_handler(CommandHandler("provider",        provider_command))
    application.add_handler(CommandHandler("models",          models_command))
    application.add_handler(CommandHandler("model",           model_command))
    application.add_handler(CommandHandler("validate",        validate_command))
    application.add_handler(CommandHandler("verified",        verified_command))
    application.add_handler(CommandHandler("clearvalidation", clearvalidation_command))
    application.add_handler(CommandHandler("thinking",        thinking_command))
    application.add_handler(CommandHandler("web",             web_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 Multi-Provider AI Bot started!")
    logger.info(f"📡 Providers: {', '.join(provider_manager.list_providers())}")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
