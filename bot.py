import os
import re
import io
import gc
import base64
import time
import logging
import json
import asyncio
import datetime
import urllib.parse
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
try:
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '1024'))
except ValueError:
    MAX_TOKENS = 1024
try:
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
except ValueError:
    TEMPERATURE = 0.7
try:
    MAX_HISTORY_MESSAGES = int(os.getenv('MAX_HISTORY_MESSAGES', '20'))  # must be even
except ValueError:
    MAX_HISTORY_MESSAGES = 20
MAX_MESSAGE_LENGTH = 4096
MAX_INPUT_LENGTH   = 4000  # Reject user messages above this to protect memory and API costs

# Web Search Configuration
BRAVE_API_KEY  = os.getenv('BRAVE_API_KEY', '')
SEARXNG_URL    = os.getenv('SEARXNG_URL', '').rstrip('/')   # e.g. http://searxng.example.com
SEARCH_ENGINE  = os.getenv('SEARCH_ENGINE', 'duckduckgo').lower()
try:
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '5'))
except ValueError:
    MAX_SEARCH_RESULTS = 5
try:
    MAX_SNIPPET_LEN = int(os.getenv('MAX_SNIPPET_LEN', '300'))
except ValueError:
    MAX_SNIPPET_LEN = 300

# Provider API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
CEREBRAS_API_KEY = os.getenv('CEREBRAS_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

# Vision / OCR Configuration
NVIDIA_VISION_MODEL = os.getenv('OCR_VISION_MODEL', 'google/gemma-3-27b-it')
VISION_BASE_URL     = os.getenv('VISION_BASE_URL', 'https://integrate.api.nvidia.com/v1').rstrip('/')
# OCR_API_KEY takes priority; falls back to NVIDIA_API_KEY so existing setups need no change
OCR_API_KEY         = os.getenv('OCR_API_KEY') or NVIDIA_API_KEY
try:
    MAX_IMAGE_BYTES = int(os.getenv('MAX_IMAGE_BYTES', str(15 * 1024 * 1024)))  # 15 MB
except ValueError:
    MAX_IMAGE_BYTES = 15 * 1024 * 1024

# Model validation cache file
VALIDATED_MODELS_CACHE = os.path.join(os.path.dirname(__file__), 'validated_models.json')

# ============================================================================
# SYSTEM PROMPTS
#
# SYSTEM_PROMPT              — all direct AI calls (no search)
# SYSTEM_PROMPT_WITH_RESULTS — when search results are injected into context
# _make_search_decision_prompt() — standalone router prompt, built per-request
#                                  with today's date; used only for the
#                                  lightweight SEARCH/NOSEARCH decision call
# ============================================================================

_PROMPT_BASE = (
    "You are a helpful, concise AI assistant speaking through Telegram.\n\n"
    "FORMATTING RULES — follow strictly:\n"
    "- Use plain Telegram Markdown only: *bold*, _italic_, `code`, ```code blocks```\n"
    "- Never use ## headers — they do not render in Telegram\n"
    "- Never use LaTeX math notation\n"
    "- Use flat bullet lists (- item). Never nest lists\n"
    "- Keep responses concise. Avoid long preambles and sign-offs\n"
    "- For comparisons: use *bold item name* on its own line, then bullet points for attributes. "
    "Never use | pipe | tables — they do not render in Telegram\n\n"
    "BEHAVIOUR:\n"
    "- Be direct and practical\n"
    "- If you don't know something, say so rather than guessing\n"
    "- When code is requested, use fenced code blocks with the language name"
)

_PROMPT_SEARCH_DECISION = (
    "You are a search router. Decide if the user's latest message needs a "
    "live web search to answer accurately. Today's date is {date}.\n\n"
    "Respond with EXACTLY one line:\n"
    "  SEARCH: <2-6 word query>   — needs current/real-time info\n"
    "  NOSEARCH                   — can answer from training knowledge\n\n"
    "Use SEARCH for: current events, live prices, weather, sports scores, "
    "recent news, product availability, real-time data, people's current status, "
    "or any fact you are NOT fully confident about.\n\n"
    "Use NOSEARCH for: coding help, math, creative writing, definitions, "
    "explanations, greetings, opinions, general knowledge, or anything "
    "you can confidently answer from training data.\n\n"
    "CRITICAL: Output ONLY one line. No explanations. No apologies. "
    "No extra text after SEARCH: query. Just the decision."
)

_PROMPT_SEARCH_RESULTS = (
    "\n\nSEARCH RESULTS FORMAT:\n"
    "When web search results are provided, they appear as numbered snippets "
    "(1. Title: text). Use them to answer accurately. "
    "You may refer to a result naturally (e.g. 'according to X') but do NOT "
    "write citation numbers like [1] or [2] — there is no reference list."
)

# Pre-built at startup — never re-allocated per request
SYSTEM_PROMPT              = _PROMPT_BASE
SYSTEM_PROMPT_WITH_RESULTS = _PROMPT_BASE + _PROMPT_SEARCH_RESULTS


def _make_search_decision_prompt() -> str:
    """Build the standalone search-router prompt with today's date injected."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return _PROMPT_SEARCH_DECISION.format(date=today)


# ============================================================================
# SEARCH DECISION — QUICK FILTER
#
# Greetings and very short messages skip the AI decision call entirely (1 call).
# Everything else gets a lightweight AI call for SEARCH/NOSEARCH routing.
# ============================================================================

_GREETINGS = frozenset({
    'hi', 'hello', 'hey', 'hola', 'thanks', 'thank you', 'bye', 'goodbye',
    'ok', 'okay', 'yes', 'no', 'sure', 'lol', 'haha', 'nice', 'great', 'cool',
    'good morning', 'good night', 'good evening', 'good afternoon',
    'ty', 'thx', 'np', 'yw',
})


def _skip_search_decision(text: str) -> bool:
    """Return True for messages that obviously don't need a web search decision call."""
    if len(text) < 6:
        return True
    return text.lower().rstrip('!?.,: ') in _GREETINGS



def _parse_search_query(response: str) -> Optional[str]:
    """Extract SEARCH: query from AI search-decision response.

    Returns clean query string if AI wants to search, or None if it answered directly.

    Strategy: after extracting the first line following 'SEARCH:', keep only the
    leading portion that looks like a valid search query — alphanumeric, spaces,
    and common query punctuation (hyphen, apostrophe, straight quotes, dots, /).
    Anything else (*, apology text, leaked context, Markdown) is treated as a
    natural end-of-query boundary and discarded.  This replaces the old brittle
    _APOLOGY_PATTERNS list that had to be updated every time a new model quirk appeared.
    """
    stripped = response.strip()

    if stripped.upper().startswith('SEARCH:'):
        raw = stripped[7:].strip().split('\n')[0].strip()
    else:
        # Fallback: model added preamble before SEARCH: (e.g. "I'll look that up. SEARCH: query")
        m = re.search(r'(?i)\bSEARCH:\s*(.+)', stripped)
        if not m:
            return None
        raw = m.group(1).strip().split('\n')[0].strip()

    # Truncate at NOSEARCH keyword if model appends it on the same line
    # (e.g. Cerebras: "Taylor Swift latest song 2026NOSEARCHSEARCH")
    # Also recover a new SEARCH: query that follows NOSEARCH inline
    parts = re.split(r'(?i)NOSEARCH', raw, maxsplit=1)
    if len(parts) == 2 and re.match(r'(?i)\s*SEARCH:\s*', parts[1]):
        # Model wrote: "<junk>NOSEARCHSEARCH: real query" — use the part after NOSEARCH
        raw = re.sub(r'(?i)^SEARCH:\s*', '', parts[1]).strip()
    else:
        raw = parts[0].strip()

    # Strip repeated SEARCH: prefix (some models double/triple the entire output)
    while raw.upper().startswith('SEARCH:'):
        raw = raw[7:].strip()

    # Truncate at any remaining inline SEARCH: (model echoed query without newline separator)
    # e.g. "Taylor SwiftSEARCH: Taylor Swift" → "Taylor Swift"
    m_echo = re.search(r'(?i)SEARCH:', raw)
    if m_echo:
        raw = raw[:m_echo.start()].strip()

    # Strip surrounding quotes/backticks some models add
    for q in ('"', "'", '`'):
        if len(raw) > 2 and raw[0] == q and raw[-1] == q:
            raw = raw[1:-1].strip()

    # Keep only the leading valid-query portion.
    # Valid chars: letters, digits, spaces, - ' " . / & + % @ # ( )
    # First char outside this set is treated as end-of-query.
    m = re.match(r"^[\w\s\-'\".,/&+%@#()]+", raw, re.UNICODE)
    raw = m.group(0).strip() if m else ""

    # Cap at 10 words to prevent explanatory bleed-through
    # (prompt requests 2-6 words; 10 gives headroom for longer queries)
    words = raw.split()
    if len(words) > 10:
        raw = ' '.join(words[:10])

    raw = raw[:120]
    return raw if len(raw) >= 2 else None



def _build_search_context(query: str, snippets: list) -> str:
    """Format search results as numbered snippets with today's date for temporal grounding."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    parts = [f"Today is {today}. Web search results for '{query}':"]
    for i, snip in enumerate(snippets, 1):
        parts.append(f"{i}. {snip}")
    return '\n'.join(parts)


_SEARCH_DECISION_RETRIES = 3
_SEARCH_DECISION_RETRY_DELAY = 1.0  # seconds between retries

async def ai_decide_search(provider, model: Optional[str], messages: list) -> Optional[str]:
    """Lightweight AI call to decide SEARCH vs NOSEARCH.

    Uses a standalone router system prompt (not the main assistant prompt).
    Passes full conversation history so the AI has context for multi-turn exchanges.
    Retries up to _SEARCH_DECISION_RETRIES times on empty/failed responses.

    On total failure, optimistically searches using the raw user message rather than
    silently falling back to NOSEARCH.
    Returns a clean search query string, or None to answer directly.
    """
    decision_msgs = [{"role": "system", "content": _make_search_decision_prompt()}]
    decision_msgs += [m for m in messages if m.get("role") != "system"]

    last_error: Optional[Exception] = None
    for attempt in range(1, _SEARCH_DECISION_RETRIES + 1):
        try:
            response = await asyncio.to_thread(
                provider.chat,
                messages=decision_msgs,
                model=model,
                enable_thinking=False,
                max_tokens=100,
            ) or ""
            if response.strip():
                return _parse_search_query(response)
            logger.warning(f"[Bot] Search decision attempt {attempt}: empty response, retrying...")
        except Exception as e:
            last_error = e
            logger.warning(f"[Bot] Search decision attempt {attempt} error: {e}, retrying...")
        if attempt < _SEARCH_DECISION_RETRIES:
            await asyncio.sleep(_SEARCH_DECISION_RETRY_DELAY)

    fallback_query = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        None,
    )
    if fallback_query:
        fallback_query = fallback_query[:80].strip()
        logger.warning(
            f"[Bot] Search decision failed after {_SEARCH_DECISION_RETRIES} attempts "
            f"({last_error or 'empty response'}) — optimistic fallback SEARCH: '{fallback_query}'"
        )
        return fallback_query

    logger.warning(
        f"[Bot] Search decision failed after {_SEARCH_DECISION_RETRIES} attempts "
        f"({last_error or 'empty response'}) — no user message found, falling back to NOSEARCH"
    )
    return None


# ============================================================================
# TRANSIENT API RETRY
# ============================================================================

_TRANSIENT_ERROR_KEYWORDS = (
    'rate limit', 'too many requests', '429',
    'timeout', 'timed out',
    '503', '502', '500', '529',
    'overloaded', 'temporarily unavailable', 'service unavailable',
    'connection error', 'connection reset',
)
_CHAT_MAX_RETRIES    = 2
_CHAT_RETRY_BASE_DELAY = 3.0  # seconds; doubles each attempt (3s, 6s)


async def _chat_with_retry(
    provider,
    messages: list,
    model: Optional[str],
    enable_thinking: bool,
) -> str:
    """Call provider.chat() with automatic retry on transient errors.

    Retries up to _CHAT_MAX_RETRIES times with exponential backoff.
    Non-transient errors (bad model, auth failures, etc.) raise immediately.
    """
    for attempt in range(1, _CHAT_MAX_RETRIES + 2):
        try:
            return await asyncio.to_thread(
                provider.chat,
                messages=messages,
                model=model,
                enable_thinking=enable_thinking,
            ) or ""
        except Exception as e:
            if attempt > _CHAT_MAX_RETRIES:
                raise
            error_lower = str(e).lower()
            if not any(kw in error_lower for kw in _TRANSIENT_ERROR_KEYWORDS):
                raise  # Permanent error — don't retry
            delay = _CHAT_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                f"[Bot] Transient API error (attempt {attempt}/{_CHAT_MAX_RETRIES + 1}): "
                f"{e} — retrying in {delay:.0f}s"
            )
            await asyncio.sleep(delay)
    return ""  # unreachable, satisfies type checker


# ============================================================================
# HISTORY UTILITIES
# ============================================================================

def _trim_history(history: list) -> list:
    """Trim history to MAX_HISTORY_MESSAGES keeping complete user/assistant pairs.
    Always removes the oldest pair (2 messages) so the model never sees a
    dangling half-exchange."""
    while len(history) > MAX_HISTORY_MESSAGES:
        if len(history) >= 2:
            history.pop(0)
            history.pop(0)
        else:
            history.pop(0)
    return history


async def reply_text_safe(message, text: str):
    """Send Markdown first for better chat UI, fallback to plain text on parse error.

    Re-raises if both sends fail so callers can roll back history / notify the user.
    """
    try:
        await message.reply_text(text, parse_mode='Markdown')
    except Exception:
        try:
            await message.reply_text(text)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise


# ============================================================================
# FUTURE-PROOF MODEL RANKING UTILITIES
# ============================================================================

def extract_parameter_size(model_id: str) -> int:
    match = re.search(r'(\d+\.?\d*)b', model_id.lower())
    if match:
        size = float(match.group(1))
        return int(size) if size >= 1 else 0
    return 0

def get_model_capability_score(model_id: str) -> tuple:
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

    def __init__(self):
        import threading
        self._cached_models = None
        self._models_lock = threading.Lock()

    @abstractmethod
    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False, max_tokens: Optional[int] = None) -> str:
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
        super().__init__()
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.default_model = 'llama-3.3-70b-versatile'

    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False, max_tokens: Optional[int] = None) -> str:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        response = self.client.chat.completions.create(
            messages=chat_messages, model=model,
            temperature=TEMPERATURE, max_tokens=max_tokens or MAX_TOKENS,
        )
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("API returned empty response.")
        return response.choices[0].message.content

    def get_available_models(self) -> List[Dict[str, str]]:
        with self._models_lock:
            if self._cached_models is not None:
                return self._cached_models
            try:
                models_response = self.client.models.list()
                chat_models = []
                for model in models_response.data:
                    if model.active and hasattr(model, 'id'):
                        chat_models.append({"id": model.id, "name": model.id.replace('-', ' ').title()})
                chat_models.sort(key=lambda m: get_model_capability_score(m['id']))
                self._cached_models = chat_models
                logger.info(f"✅ Groq: Detected {len(chat_models)} available models")
                return chat_models
            except Exception as e:
                logger.warning(f"⚠️ Groq: Could not fetch models: {e}")
                return self._get_fallback_models()

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
        super().__init__()
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        self.default_model = "gemini-1.5-flash"

    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False, max_tokens: Optional[int] = None) -> str:
        if not messages:
            raise ValueError("Messages list cannot be empty")
        model_name = model or self.default_model
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
        gen_model = self.genai.GenerativeModel(
            model_name,
            generation_config={"temperature": TEMPERATURE, "max_output_tokens": max_tokens or MAX_TOKENS},
            system_instruction=system_msg
        )
        chat_history = []
        for msg in messages[:-1]:
            if msg["role"] == "system":
                continue
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [msg["content"]]})
        chat = gen_model.start_chat(history=chat_history)
        return chat.send_message(messages[-1]["content"]).text

    def get_available_models(self) -> List[Dict[str, str]]:
        with self._models_lock:
            if self._cached_models is not None:
                return self._cached_models
            try:
                chat_models = []
                for model in self.genai.list_models():
                    if 'generateContent' not in model.supported_generation_methods:
                        continue
                    model_id = model.name.replace('models/', '')
                    if any(x in model_id.lower() for x in ['vision', 'embedding', 'aqa']):
                        continue
                    name = model.display_name if hasattr(model, 'display_name') else model_id.replace('-', ' ').title()
                    chat_models.append({"id": model_id, "name": name})
                chat_models.sort(key=lambda m: get_model_capability_score(m['id']))
                self._cached_models = chat_models
                logger.info(f"✅ Gemini: Detected {len(chat_models)} available models")
                return chat_models
            except Exception as e:
                logger.warning(f"⚠️ Gemini: Could not fetch models: {e}")
                return self._get_fallback_models()

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
        super().__init__()
        from openai import OpenAI
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.api_key = api_key
        self.default_model = "openrouter/free"

    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False, max_tokens: Optional[int] = None) -> str:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        response = self.client.chat.completions.create(
            model=model, messages=chat_messages,
            temperature=TEMPERATURE, max_tokens=max_tokens or MAX_TOKENS,
        )
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("API returned empty response.")
        return response.choices[0].message.content

    def get_available_models(self) -> List[Dict[str, str]]:
        with self._models_lock:
            if self._cached_models is not None:
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
                    if ':free' not in model_id.lower() or context_length <= 0:
                        continue
                    try:
                        prompt_price = float(pricing.get('prompt') or 0)
                        compl_price  = float(pricing.get('completion') or 0)
                    except (TypeError, ValueError):
                        continue
                    if prompt_price != 0.0 or compl_price != 0.0:
                        continue
                    name = model.get('name', model_id)
                    name = name.replace(' (free)', '').replace(' (Free)', '') + " (Free)"
                    free_models.append({"id": model_id, "name": name, "context": context_length})
                free_models.sort(key=lambda m: get_model_capability_score(m['id']))
                self._cached_models = free_models[:15]
                logger.info(f"✅ OpenRouter: Detected {len(self._cached_models)} free models")
                return self._cached_models
            except Exception as e:
                logger.warning(f"⚠️ OpenRouter: Could not fetch models: {e}")
                return self._get_fallback_models()

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
        super().__init__()
        from cerebras.cloud.sdk import Cerebras
        self.client = Cerebras(api_key=api_key)
        self.default_model = "gpt-oss-120b"

    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False, max_tokens: Optional[int] = None) -> str:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        response = self.client.chat.completions.create(
            messages=chat_messages, model=model,
            temperature=TEMPERATURE, max_tokens=max_tokens or MAX_TOKENS,
        )
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("API returned empty response.")
        return response.choices[0].message.content

    def get_available_models(self) -> List[Dict[str, str]]:
        with self._models_lock:
            if self._cached_models is not None:
                return self._cached_models
            try:
                chat_models = []
                for model in self.client.models.list().data:
                    if hasattr(model, 'id'):
                        chat_models.append({"id": model.id, "name": model.id.replace('-', ' ').title()})
                chat_models.sort(key=lambda m: get_model_capability_score(m['id']))
                self._cached_models = chat_models
                logger.info(f"✅ Cerebras: Detected {len(chat_models)} available models")
                return chat_models
            except Exception as e:
                logger.warning(f"⚠️ Cerebras: Could not fetch models: {e}")
                return self._get_fallback_models()

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
        super().__init__()
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.default_model = 'openai/gpt-oss-120b'

    def supports_thinking(self, model_id: str) -> bool:
        return model_id not in self.MODELS_WITHOUT_THINKING

    def chat(self, messages: List[Dict], model: Optional[str] = None,
             enable_thinking: bool = False, max_tokens: Optional[int] = None) -> str:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        if enable_thinking and self.supports_thinking(model):
            response = self.client.chat.completions.create(
                messages=chat_messages, model=model,
                temperature=TEMPERATURE, max_tokens=max_tokens or MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True
            )
            reasoning_parts, content_parts = [], []
            for chunk in response:
                if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "reasoning_content", None):
                    reasoning_parts.append(delta.reasoning_content)
                if getattr(delta, "content", None) is not None:
                    content_parts.append(delta.content)
            full = ""
            if reasoning_parts:
                full = "💭 *Thinking:*\n" + "".join(reasoning_parts) + "\n\n"
            return full + "".join(content_parts)
        else:
            # Always stream — NVIDIA's non-streaming path returns content=None
            # for certain models (including gpt-oss-120b) on some responses.
            # Streaming accumulates delta.content chunks and never yields None.
            extra_body = (
                {"chat_template_kwargs": {"thinking": False}}
                if self.supports_thinking(model) else {}
            )
            response = self.client.chat.completions.create(
                messages=chat_messages, model=model,
                temperature=TEMPERATURE, max_tokens=max_tokens or MAX_TOKENS,
                **({"extra_body": extra_body} if extra_body else {}),
                stream=True,
            )
            content_parts = []
            chunk_count = 0
            # #region agent log
            import traceback as _tb
            # #endregion
            try:
                for chunk in response:
                    chunk_count += 1
                    if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                        continue
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None) is not None:
                        content_parts.append(delta.content)
            except Exception as _stream_err:
                # #region agent log
                logger.error(
                    "[DBG ab8c77] NVIDIA stream error | hyp=A-D | chunks_before_error=%d | exc_type=%s | exc_module=%s | msg=%s | tb=%s",
                    chunk_count, type(_stream_err).__name__, type(_stream_err).__module__,
                    str(_stream_err), _tb.format_exc()[-600:].replace("\n", "↵")
                )
                # #endregion
                raise
            # #region agent log
            logger.info(
                "[DBG ab8c77] NVIDIA stream OK | hyp=E | chunks=%d | content_len=%d",
                chunk_count, len("".join(content_parts))
            )
            # #endregion
            result = "".join(content_parts)
            if not result:
                raise ValueError("API returned empty response.")
            return result

    def get_available_models(self) -> List[Dict[str, str]]:
        with self._models_lock:
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


provider_manager = ProviderManager()
user_sessions: Dict[str, Dict] = {}

_SESSION_TTL      = 86400   # evict sessions inactive for >24 h
_CLEANUP_INTERVAL = 3600   # run cleanup at most once per hour
_last_session_cleanup: float = 0.0
_cleanup_in_progress: bool = False


# ============================================================================
# SESSION HELPERS
# ============================================================================

def get_user_session(user_id: str) -> Dict:
    global _last_session_cleanup, _cleanup_in_progress
    now = time.time()

    # Lazy hourly eviction of inactive sessions — guard prevents double-entry
    if not _cleanup_in_progress and now - _last_session_cleanup > _CLEANUP_INTERVAL:
        _cleanup_in_progress = True
        try:
            stale = [uid for uid, s in list(user_sessions.items())
                     if now - s.get("last_seen", 0) > _SESSION_TTL]
            for uid in stale:
                user_sessions.pop(uid, None)
            if stale:
                logger.info(f"[Session] Evicted {len(stale)} stale session(s)")
        finally:
            _last_session_cleanup = now
            _cleanup_in_progress = False

    if user_id not in user_sessions:
        if SEARCH_ENGINE == "searxng" and SEARXNG_URL:
            default_engine = "searxng"
        elif SEARCH_ENGINE == "brave" and BRAVE_API_KEY:
            default_engine = "brave"
        else:
            default_engine = "duckduckgo"
        user_sessions[user_id] = {
            "provider":         provider_manager.get_default_provider(),
            "models":           {},
            "history":          [],
            "thinking_enabled": False,
            "web_search":       True,
            "search_engine":    default_engine,
            "last_seen":        now,
        }
    else:
        user_sessions[user_id]["last_seen"] = now
    return user_sessions[user_id]


def is_user_allowed(user_id: str) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def _resolve_provider(session: Dict):
    """Return (provider_name, provider) — auto-corrects stale provider."""
    name = session["provider"]
    prov = provider_manager.get_provider(name)
    if not prov:
        name = provider_manager.get_default_provider()
        session["provider"] = name
        prov = provider_manager.get_provider(name)
    return name, prov


# ============================================================================
# MODEL VALIDATION CACHE
# ============================================================================


# In-memory cache — single source of truth; disk is persistence only.
# All add/get operations mutate this dict directly, eliminating the
# load→modify→save race that occurs when two concurrent /validate runs interleave.
_validated_cache: Optional[Dict] = None

def load_validated_models() -> Dict:
    global _validated_cache
    if _validated_cache is not None:
        return _validated_cache
    try:
        if os.path.exists(VALIDATED_MODELS_CACHE):
            with open(VALIDATED_MODELS_CACHE, 'r') as f:
                _validated_cache = json.load(f)
                return _validated_cache
    except Exception as e:
        logger.warning(f"Could not load validated models cache: {e}")
    _validated_cache = {}
    return _validated_cache

def save_validated_models(validated: Dict):
    global _validated_cache
    _validated_cache = validated
    try:
        with open(VALIDATED_MODELS_CACHE, 'w') as f:
            json.dump(validated, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save validated models cache: {e}")

def _ensure_provider_entry(validated: Dict, provider_name: str) -> Dict:
    if provider_name not in validated:
        validated[provider_name] = {'working': [], 'failed': []}
    if isinstance(validated[provider_name], list):
        validated[provider_name] = {'working': validated[provider_name], 'failed': []}
    return validated

def get_validated_models(provider_name: str) -> List[str]:
    d = load_validated_models().get(provider_name, {})
    return d.get('working', []) if isinstance(d, dict) else d

def get_failed_models(provider_name: str) -> List[str]:
    d = load_validated_models().get(provider_name, {})
    return d.get('failed', []) if isinstance(d, dict) else []

def add_validated_model(provider_name: str, model_id: str):
    validated = _ensure_provider_entry(load_validated_models(), provider_name)
    if model_id not in validated[provider_name]['working']:
        validated[provider_name]['working'].append(model_id)
        if model_id in validated[provider_name]['failed']:
            validated[provider_name]['failed'].remove(model_id)
        save_validated_models(validated)

def add_failed_model(provider_name: str, model_id: str):
    validated = _ensure_provider_entry(load_validated_models(), provider_name)
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
# WEB SEARCH — Brave API + ddgs (multi-backend) + SearXNG
# ============================================================================

def _brave_search_sync(query: str) -> list:
    if not BRAVE_API_KEY:
        return []
    q = urllib.parse.quote_plus(query)
    try:
        r = requests.get(
            f"https://api.search.brave.com/res/v1/web/search?q={q}&count={MAX_SEARCH_RESULTS}",
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
            timeout=10)
        r.raise_for_status()
        snippets = []
        for item in r.json().get("web", {}).get("results", []):
            title = item.get("title", "").strip()
            desc  = item.get("description", "").strip()[:MAX_SNIPPET_LEN]
            if title and len(desc) >= 15:
                snippets.append(f"{title}: {desc}")
        logger.info(f"[Search] Brave '{query}' -> {len(snippets)} results")
        return snippets
    except Exception as e:
        logger.error(f"[Search] Brave error: {e}")
        return []

def _duckduckgo_search_sync(query: str) -> list:
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=MAX_SEARCH_RESULTS, backend="auto")
        snippets = []
        for r in results:
            title = r.get("title", "").strip()
            body  = r.get("body", "").strip()[:MAX_SNIPPET_LEN]
            if title and len(body) >= 15:
                snippets.append(f"{title}: {body}")
        logger.info(f"[Search] DDG '{query}' -> {len(snippets)} results")
        return snippets
    except Exception as e:
        logger.error(f"[Search] DDG error: {e}")
        return []

def _searxng_search_sync(query: str) -> list:
    if not SEARXNG_URL:
        return []
    try:
        r = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "count": MAX_SEARCH_RESULTS},
            headers={"Accept": "application/json"},
            timeout=10)
        r.raise_for_status()
        snippets = []
        for item in r.json().get("results", [])[:MAX_SEARCH_RESULTS]:
            title   = item.get("title", "").strip()
            content = item.get("content", "").strip()[:MAX_SNIPPET_LEN]
            if title and len(content) >= 15:
                snippets.append(f"{title}: {content}")
        logger.info(f"[Search] SearXNG '{query}' -> {len(snippets)} results")
        return snippets
    except Exception as e:
        logger.error(f"[Search] SearXNG error: {e}")
        return []

async def web_search(query: str, engine: str) -> list:
    if engine == "searxng" and SEARXNG_URL:
        results = await asyncio.to_thread(_searxng_search_sync, query)
        if results:
            return results
        logger.warning(f"[Search] SearXNG returned no results for '{query}' — falling back to DuckDuckGo")
    if engine == "brave" and BRAVE_API_KEY:
        results = await asyncio.to_thread(_brave_search_sync, query)
        if results:
            return results
        logger.warning(f"[Search] Brave returned no results for '{query}' — falling back to DuckDuckGo")
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
    _, provider = _resolve_provider(session)
    web_status = "ON" if session.get("web_search") else "OFF"
    await update.message.reply_text(
        f"🤖 Hello! I'm your Multi-Provider AI assistant.\n\n"
        f"📡 Current Provider: *{provider.get_name()}*\n"
        f"🔧 Available Providers: {', '.join(provider_manager.list_providers())}\n"
        f"🌐 Web Search: {web_status}\n\n"
        f"Just send me a message or a photo!\n\n"
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
        await asyncio.to_thread(provider_manager.refresh_models)
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
        "• `/web searxng` — SearXNG (self-hosted)\n"
        "• `/web ddg` — DuckDuckGo (free)\n\n"
        "*Model Validation:*\n"
        "• `/validate` — test which models work\n"
        "• `/verified` — show validated models\n"
        "• `/clearvalidation` — clear cache\n\n"
        "*Thinking Mode (NVIDIA only):*\n"
        "• `/thinking on` / `/thinking off`\n\n"
        "*Image OCR:*\n"
        "• Send any photo — text is extracted via NVIDIA vision\n"
        "• Add a caption to ask a specific question about the image\n"
        "• Requires `NVIDIA_API_KEY` to be set\n\n"
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
        _, current = _resolve_provider(session)
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
    session["history"] = []  # Clear history — context from old provider is incompatible
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
    provider_name, provider = _resolve_provider(session)
    show_all = len(context.args) > 0 and context.args[0].lower() == 'all'
    if show_all:
        await update.message.reply_text("🔄 Fetching all models from API...")
        models = await asyncio.to_thread(provider.get_available_models)
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
        all_models = await asyncio.to_thread(provider.get_available_models)
        models = [m for m in all_models if m['id'] in validated_ids]
        if not models:
            await update.message.reply_text(
                f"⚠️ *Validated models are no longer in {provider.get_name()}'s model list.*\n\n"
                f"Run `/clearvalidation` then `/validate` to refresh.",
                parse_mode='Markdown'
            )
            return
        title_suffix = " (Verified)"
        footer_note = f"\n\n💡 Use `/models all` to see all {len(all_models)} models"
    current_model = session["models"].get(provider_name) or provider.get_default_model()
    chunks = [models[i:i+20] for i in range(0, len(models), 20)]
    for idx, chunk in enumerate(chunks, 1):
        model_list = "\n".join([
            f"• `{m['id']}`" + (" ✓" if m['id'] == current_model else "")
            for m in chunk
        ])
        part = f" (Part {idx}/{len(chunks)})" if len(chunks) > 1 else ""
        await update.message.reply_text(
            f"🤖 *{provider.get_name()}{title_suffix}{part}:*\n\n{model_list}"
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
    provider_name, provider = _resolve_provider(session)
    if not context.args:
        current = session["models"].get(provider_name) or provider.get_default_model()
        await update.message.reply_text(
            f"🤖 *Current Model ({provider.get_name()}):* `{current}`\n\nUse `/models` to see options.",
            parse_mode='Markdown'
        )
        return
    new_model = " ".join(context.args)
    model_ids = [m['id'] for m in await asyncio.to_thread(provider.get_available_models)]
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
    provider_name, provider = _resolve_provider(session)
    models = await asyncio.to_thread(provider.get_available_models)
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
    newly_validated, failed_na, failed_rl, failed_unk = [], [], [], []
    for idx, model_info in enumerate(models_to_test, 1):
        model_id = model_info['id']
        if idx % 5 == 0 or idx == 1:
            await update.message.reply_text(
                f"⏳ {idx}/{len(models_to_test)}: `{model_id}`...", parse_mode='Markdown'
            )
        success, error_type = await asyncio.to_thread(provider.test_model, model_id)
        if success:
            validated.append(model_id); newly_validated.append(model_id)
            add_validated_model(provider_name, model_id)
        elif error_type == 'not_available':
            failed_na.append(model_id); add_failed_model(provider_name, model_id)
        elif error_type == 'rate_limit':
            failed_rl.append(model_id)
        else:
            failed_unk.append(model_id)
        if idx < len(models_to_test):
            await asyncio.sleep(2)
    total_failed = len(failed_na) + len(failed_rl) + len(failed_unk)
    rate = (len(newly_validated) / len(models_to_test) * 100) if models_to_test else 0
    msg = (
        f"✅ *Validation Complete*\n\n"
        f"• Tested: {len(models_to_test)}\n"
        f"• ✅ Newly validated: {len(newly_validated)}\n"
        f"• ❌ Failed: {total_failed}\n"
    )
    if total_failed > 0:
        msg += (
            f"\n*Failure breakdown:*\n"
            f"• 🚫 Not available: {len(failed_na)} (cached)\n"
            f"• ⏱️ Rate limited: {len(failed_rl)} (retry later)\n"
            f"• ❓ Unknown: {len(failed_unk)}\n"
        )
    msg += f"\n• Success rate: {rate:.1f}%\n\n📦 *Total validated: {len(validated)}*\n\nUse `/verified` to see working models!"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def verified_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    session = get_user_session(user_id)
    provider_name, provider = _resolve_provider(session)
    validated_ids = get_validated_models(provider_name)
    if not validated_ids:
        await update.message.reply_text(
            f"❌ *No validated models for {provider.get_name()}*\n\nRun `/validate` first!",
            parse_mode='Markdown'
        )
        return
    all_models = await asyncio.to_thread(provider.get_available_models)
    validated_models = [m for m in all_models if m['id'] in validated_ids]
    if not validated_models:
        await update.message.reply_text(
            f"⚠️ *Validated models are no longer in {provider.get_name()}'s model list.*\n\n"
            f"Run `/clearvalidation` then `/validate` to refresh.",
            parse_mode='Markdown'
        )
        return
    current_model = session["models"].get(provider_name) or provider.get_default_model()
    model_list = "\n".join([
        f"• `{m['id']}`" + (" ✓" if m['id'] == current_model else "")
        for m in validated_models
    ])
    await update.message.reply_text(
        f"✅ *Verified Models — {provider.get_name()}:*\n\n{model_list}\n\n"
        f"Current: `{current_model}`\n\nUse `/model <id>` to switch.",
        parse_mode='Markdown'
    )

async def thinking_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    session = get_user_session(user_id)
    provider_name, provider = _resolve_provider(session)
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
            ok = provider.supports_thinking(current_model)
            await update.message.reply_text(
                f"✅ *Thinking mode enabled!* 💭\n\n"
                f"`{current_model}` {'supports' if ok else 'does NOT support'} thinking.",
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
    provider_name, provider = _resolve_provider(session)
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
        eng_label = {"brave": "Brave API", "searxng": "SearXNG", "duckduckgo": "DuckDuckGo"}.get(eng, eng)
        searxng_line = f"`/web searxng` — SearXNG (self-hosted)\n" if SEARXNG_URL else ""
        await update.message.reply_text(
            f"🌐 *Web Search:* {status}\n"
            f"🔍 *Engine:* {eng_label}\n\n"
            f"Use: `/web on` | `/web off`\n"
            f"`/web brave` — Brave Search API\n"
            f"{searxng_line}"
            f"`/web ddg` — DuckDuckGo (free)",
            parse_mode='Markdown'
        )
        return
    arg = context.args[0].lower()
    if arg == "on":
        session["web_search"] = True
        eng = session.get("search_engine", "duckduckgo")
        eng_label = {"brave": "Brave", "searxng": "SearXNG", "duckduckgo": "DuckDuckGo"}.get(eng, eng)
        await update.message.reply_text(f"✅ Web search enabled ({eng_label}).")
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
    elif arg == "searxng":
        if not SEARXNG_URL:
            await update.message.reply_text("❌ SEARXNG_URL not configured.", parse_mode='Markdown')
        else:
            session["search_engine"] = "searxng"
            session["web_search"] = True
            await update.message.reply_text("✅ Switched to SearXNG.")
    else:
        await update.message.reply_text("❌ Use: `/web on|off|brave|searxng|ddg`", parse_mode='Markdown')


# ============================================================================
# MESSAGE HANDLER — AI-routed search (v4)
#
# FLOW:
#
#   Quick filter (greeting / very short):
#     → 1 AI call, no search overhead
#
#   Everything else:
#     → 1 lightweight AI decision call (SEARCH/NOSEARCH)
#       NOSEARCH → 1 AI call with full system prompt → reply
#       SEARCH   → web_search → 1 AI call with results + full system prompt → reply
#
# WHY BETTER THAN 3-TIER HEURISTICS:
#   - "playlist" no longer hits "list" keyword → no false no-search
#   - "barcode" no longer hits "code" → no false no-search
#   - "today's code review" no longer triggers direct search
#   - Answers ALWAYS from full SYSTEM_PROMPT, never from decision prompt
#   - AI crafts optimised search queries vs dumb prefix stripping
#   - Cost: +1 small API call (max_tokens implicit, temp=0) per non-trivial message
# ============================================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return

    user_message = update.message.text
    if not user_message:
        return
    if len(user_message) > MAX_INPUT_LENGTH:
        await update.message.reply_text(
            f"❌ Message too long ({len(user_message):,} chars). "
            f"Please keep it under {MAX_INPUT_LENGTH:,} characters."
        )
        return

    session = get_user_session(user_id)
    assistant_appended = False

    try:
        try:
            await update.message.chat.send_action(action="typing")
        except Exception:
            pass

        provider_name, provider = _resolve_provider(session)
        current_model    = session["models"].get(provider_name)   # None = provider default
        thinking_enabled = session.get("thinking_enabled", False)
        web_on           = session.get("web_search", True)

        # Append user message (popped on error below)
        session["history"].append({"role": "user", "content": user_message})

        # ── AI SEARCH DECISION ─────────────────────────────────────────────
        skip = not web_on or _skip_search_decision(user_message)
        search_query = None

        if not skip:
            logger.info(f"[Bot] user={user_id} provider={provider_name} → asking AI for search decision")
            try:
                await update.message.chat.send_action(action="typing")
            except Exception:
                pass
            search_query = await ai_decide_search(provider, current_model, session["history"])
            logger.info(f"[Bot] Decision: {'SEARCH: ' + search_query if search_query else 'NOSEARCH'}")

        bot_response = ""

        # ── SEARCH PATH ───────────────────────────────────────────────────
        if search_query:
            engine = session.get("search_engine", "duckduckgo")
            logger.info(f"[Bot] Searching ({engine}): '{search_query}'")

            try:
                await update.message.chat.send_action(action="typing")
            except Exception:
                pass
            snippets = await web_search(search_query, engine)

            if snippets:
                ctx = _build_search_context(search_query, snippets)
                search_msgs = session["history"][:-1].copy()
                search_msgs.append({"role": "user", "content": user_message + "\n\n" + ctx})
                search_msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT_WITH_RESULTS})

                try:
                    await update.message.chat.send_action(action="typing")
                except Exception:
                    pass
                bot_response = await _chat_with_retry(
                    provider,
                    messages=search_msgs,
                    model=current_model,
                    enable_thinking=thinking_enabled,
                )
            else:
                logger.warning("[Bot] Search returned no results — falling back to direct AI")
                bot_response = await _chat_with_retry(
                    provider,
                    messages=session["history"],
                    model=current_model,
                    enable_thinking=thinking_enabled,
                )

        # ── NO-SEARCH PATH ────────────────────────────────────────────────
        else:
            if skip:
                logger.info(f"[Bot] user={user_id} provider={provider_name} → quick-filter skip, direct AI call")
            bot_response = await _chat_with_retry(
                provider,
                messages=session["history"],
                model=current_model,
                enable_thinking=thinking_enabled,
            )

        # ── Fallback for empty response ────────────────────────────────────
        if not bot_response.strip():
            bot_response = "⚠️ The AI returned an empty response. Try again or use `/model` to switch models."

        # ── Update history ─────────────────────────────────────────────────
        session["history"].append({"role": "assistant", "content": bot_response})
        assistant_appended = True
        _trim_history(session["history"])

        # ── Send response (handle 4096 char limit) ─────────────────────────
        if len(bot_response) <= MAX_MESSAGE_LENGTH:
            await reply_text_safe(update.message, bot_response)
        else:
            HEADER_RESERVE = 25
            chunk_limit = MAX_MESSAGE_LENGTH - HEADER_RESERVE
            chunks, current_chunk = [], ""
            for line in bot_response.split('\n'):
                if len(line) > chunk_limit:
                    if current_chunk: chunks.append(current_chunk); current_chunk = ""
                    for j in range(0, len(line), chunk_limit):
                        chunks.append(line[j:j + chunk_limit])
                    continue
                if len(current_chunk) + len(line) + 1 > chunk_limit:
                    if current_chunk: chunks.append(current_chunk)
                    current_chunk = line
                else:
                    current_chunk = (current_chunk + '\n' + line) if current_chunk else line
            if current_chunk: chunks.append(current_chunk)
            for i, chunk in enumerate(chunks, 1):
                header = f"📄 Part {i}/{len(chunks)}\n\n" if len(chunks) > 1 else ""
                await reply_text_safe(update.message, header + chunk)

    except Exception as e:
        logger.error(f"Error in handle_message: {e}", exc_info=True)
        # Roll back the incomplete exchange so history stays consistent.
        # Assistant message first (innermost), then user message.
        if assistant_appended and session["history"] and session["history"][-1].get("role") == "assistant":
            session["history"].pop()
        if session["history"] and session["history"][-1].get("role") == "user":
            session["history"].pop()
        _, prov = _resolve_provider(session)
        try:
            await update.message.reply_text(
                f"❌ Error with {prov.get_name()}: {str(e)}\n\n"
                f"Try:\n• `/clear` to reset conversation\n• `/provider` to switch provider"
            )
        except Exception:
            pass


# ============================================================================
# IMAGE OCR — NVIDIA vision API, fully in-memory (no disk)
#
# SINGLE PHOTO FLOW:
#   Telegram photo → BytesIO (RAM) → base64 → NVIDIA streaming API → reply
#
# ALBUM (multi-photo) FLOW:
#   Telegram sends each photo as a separate Update sharing a media_group_id.
#   Each update is buffered for _MEDIA_GROUP_WAIT seconds. Once the window
#   closes, all photos are processed sequentially and one combined reply is
#   sent. RAM is freed after every individual photo — only one photo lives
#   in memory at a time even for large albums.
# ============================================================================

_NVIDIA_VISION_URL  = f"{VISION_BASE_URL}/chat/completions"
_MEDIA_GROUP_WAIT   = 1.5   # seconds to collect all photos in an album
_OCR_MAX_RETRIES    = 2     # retry the NVIDIA API call up to 2 extra times
_OCR_RETRY_BASE_DELAY = 3.0 # seconds; doubles each attempt (3s, 6s)
_DEFAULT_OCR_PROMPT = (
    "Extract and transcribe ALL text visible in this image exactly as written. "
    "If there is no text, describe the image content concisely."
)

# {media_group_id: {"photos": [...], "message": msg, "prompt": str,
#                   "user_id": str, "session": dict}}
_media_group_buffer: Dict[str, dict] = {}
# {media_group_id: asyncio.Task}  — one flush task per in-flight album
_media_group_tasks:  Dict[str, "asyncio.Task[None]"] = {}


def _nvidia_vision_sync(b64_data: str, prompt: str) -> str:
    """Blocking NVIDIA vision call. Runs inside asyncio.to_thread — never call directly."""
    headers = {
        "Authorization": f"Bearer {OCR_API_KEY}",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": NVIDIA_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.20,
        "top_p": 0.70,
        "stream": True,
    }
    response = requests.post(_NVIDIA_VISION_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    parts = []
    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if line_str.startswith("data: "):
            line_str = line_str[6:]
        if line_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(line_str)
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                parts.append(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    return "".join(parts)


async def _ocr_one_photo(context, photo_obj, prompt: str, user_id: str) -> str:
    """Download a single Telegram photo to RAM, OCR it, return result text.

    The NVIDIA API call is retried up to _OCR_MAX_RETRIES times on transient
    errors (rate limit, timeout, 5xx). The base64 string is kept in memory
    across retries so the image is never re-downloaded. Both raw bytes and
    the base64 string are deleted in the finally block regardless of outcome.
    """
    buf: Optional[io.BytesIO] = None
    b64_str: Optional[str] = None
    try:
        # ── Download to RAM ───────────────────────────────────────────────
        tg_file = await context.bot.get_file(photo_obj.file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(buf)
        logger.info(f"[OCR] user={user_id} size={buf.tell()/1024:.1f} KB model={NVIDIA_VISION_MODEL}")

        # ── Encode once; free raw bytes immediately ───────────────────────
        b64_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        del buf
        buf = None

        # ── API call with retry ───────────────────────────────────────────
        last_error: Optional[Exception] = None
        for attempt in range(1, _OCR_MAX_RETRIES + 2):
            try:
                result = await asyncio.to_thread(_nvidia_vision_sync, b64_str, prompt)
                if result.strip():
                    return result.strip()
                # Empty response — treat as transient and retry
                logger.warning(f"[OCR] attempt {attempt}: empty response from model, retrying...")
            except Exception as e:
                last_error = e
                error_lower = str(e).lower()
                if not any(kw in error_lower for kw in _TRANSIENT_ERROR_KEYWORDS):
                    raise  # Permanent error (bad key, invalid model, etc.) — fail immediately
                logger.warning(f"[OCR] attempt {attempt} transient error: {e}")

            if attempt <= _OCR_MAX_RETRIES:
                delay = _OCR_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"[OCR] retrying in {delay:.0f}s...")
                await asyncio.sleep(delay)

        if last_error:
            raise last_error
        return "⚠️ The model returned an empty response after retries. Try a clearer image."
    finally:
        if buf is not None:
            try:
                buf.close()
            except Exception:
                pass
            del buf
        if b64_str is not None:
            del b64_str
        gc.collect()


async def _send_ocr_reply(message, text: str):
    """Send OCR text, splitting at Telegram's 4096-char limit if needed."""
    if len(text) <= MAX_MESSAGE_LENGTH:
        await reply_text_safe(message, text)
        return
    chunk_limit = MAX_MESSAGE_LENGTH - 25
    chunks, current_chunk = [], ""
    for line in text.split('\n'):
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
        await reply_text_safe(message, header + chunk)


async def _flush_media_group(group_id: str, context):
    """Wait for the album collection window, then OCR all photos and send one reply."""
    await asyncio.sleep(_MEDIA_GROUP_WAIT)

    entry = _media_group_buffer.pop(group_id, None)
    _media_group_tasks.pop(group_id, None)
    if not entry:
        return

    photos   = entry["photos"]
    message  = entry["message"]
    prompt   = entry["prompt"]
    user_id  = entry["user_id"]
    session  = entry["session"]
    total    = len(photos)

    logger.info(f"[OCR] album {group_id}: processing {total} photo(s) for user={user_id}")

    try:
        await message.chat.send_action(action="typing")
    except Exception:
        pass

    results = []
    for idx, photo in enumerate(photos, 1):
        try:
            await message.chat.send_action(action="typing")
        except Exception:
            pass
        logger.info(f"[OCR] album photo {idx}/{total}")
        try:
            result = await _ocr_one_photo(context, photo, prompt, user_id)
        except Exception as e:
            logger.error(f"[OCR] album photo {idx} failed: {e}", exc_info=True)
            result = f"⚠️ Failed to process image {idx}: {e}"
        results.append(result)

    # Build combined reply — single photo gets no wrapper, album gets numbered sections
    if total == 1:
        combined = results[0]
    else:
        sections = [f"*Image {i}/{total}*\n{r}" for i, r in enumerate(results, 1)]
        combined = "\n\n---\n\n".join(sections)

    # Store a text-only placeholder in history (no base64 ever enters history)
    session["history"].append({"role": "user", "content": f"[{total} image(s)] {prompt}"})
    session["history"].append({"role": "assistant", "content": combined})
    _trim_history(session["history"])

    try:
        await _send_ocr_reply(message, combined)
    except Exception as e:
        logger.error(f"[OCR] Failed to send album reply: {e}", exc_info=True)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return

    if not OCR_API_KEY:
        await update.message.reply_text(
            "❌ Image OCR requires `NVIDIA_API_KEY` or `OCR_API_KEY` to be configured.",
            parse_mode='Markdown'
        )
        return

    photo = update.message.photo[-1]  # largest resolution variant
    prompt = (update.message.caption.strip() if update.message.caption else _DEFAULT_OCR_PROMPT)

    if photo.file_size and photo.file_size > MAX_IMAGE_BYTES:
        await update.message.reply_text(
            f"❌ Image too large ({photo.file_size // 1024:,} KB). "
            f"Max allowed: {MAX_IMAGE_BYTES // 1024 // 1024} MB."
        )
        return

    session   = get_user_session(user_id)
    group_id  = update.message.media_group_id  # None for single photos

    # ── ALBUM: buffer and schedule a flush task ───────────────────────────
    if group_id is not None:
        if group_id not in _media_group_buffer:
            _media_group_buffer[group_id] = {
                "photos":  [],
                "message": update.message,
                "prompt":  prompt,
                "user_id": user_id,
                "session": session,
            }
            _media_group_tasks[group_id] = asyncio.create_task(
                _flush_media_group(group_id, context)
            )
        # Telegram only attaches the caption to the first photo in a group;
        # update prompt if this message carries one.
        if update.message.caption:
            _media_group_buffer[group_id]["prompt"] = prompt
        _media_group_buffer[group_id]["photos"].append(photo)
        logger.info(
            f"[OCR] buffered photo {len(_media_group_buffer[group_id]['photos'])} "
            f"for album {group_id} user={user_id}"
        )
        return

    # ── SINGLE PHOTO: process immediately ────────────────────────────────
    try:
        await update.message.chat.send_action(action="typing")
    except Exception:
        pass

    try:
        result = await _ocr_one_photo(context, photo, prompt, user_id)

        session["history"].append({"role": "user", "content": f"[image] {prompt}"})
        session["history"].append({"role": "assistant", "content": result})
        _trim_history(session["history"])

        await _send_ocr_reply(update.message, result)

    except Exception as e:
        logger.error(f"[OCR] Error for user={user_id}: {e}", exc_info=True)
        try:
            await update.message.reply_text(
                f"❌ Failed to process image: {str(e)}\n\n"
                f"Make sure `NVIDIA_API_KEY` is valid and model `{NVIDIA_VISION_MODEL}` is accessible."
            )
        except Exception:
            pass


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
    application.add_handler(MessageHandler(filters.PHOTO,                   handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 Multi-Provider AI Bot started!")
    logger.info(f"📡 Providers: {', '.join(provider_manager.list_providers())}")
    if OCR_API_KEY:
        logger.info(f"🖼️  Image OCR enabled — model: {NVIDIA_VISION_MODEL} endpoint: {VISION_BASE_URL}")
    else:
        logger.info("🖼️  Image OCR disabled — set NVIDIA_API_KEY or OCR_API_KEY to enable")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
