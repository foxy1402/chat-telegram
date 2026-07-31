import asyncio
import base64
import datetime
import gc
import io
import json
import logging
import os
import re
import threading
import time
import urllib.parse
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import requests
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER_IDS = [
    uid.strip() for uid in os.getenv("ALLOWED_USER_IDS", "").split(",") if uid.strip()
]
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "groq")
try:
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
except ValueError:
    MAX_TOKENS = 1024
try:
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
except ValueError:
    TEMPERATURE = 0.7
try:
    MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))  # must be even
except ValueError:
    MAX_HISTORY_MESSAGES = 20
if MAX_HISTORY_MESSAGES % 2:
    MAX_HISTORY_MESSAGES += 1
    logger.warning(
        f"MAX_HISTORY_MESSAGES must be even — adjusted to {MAX_HISTORY_MESSAGES}"
    )
if MAX_HISTORY_MESSAGES < 2:
    MAX_HISTORY_MESSAGES = 2
    logger.warning("MAX_HISTORY_MESSAGES must be >= 2 — adjusted to 2")

# ── SMART COMPACTION (Claude-style rolling memory) ─────────────────────────
try:
    COMPACT_THRESHOLD = int(os.getenv("COMPACT_THRESHOLD", str(MAX_HISTORY_MESSAGES)))
except ValueError:
    COMPACT_THRESHOLD = MAX_HISTORY_MESSAGES
try:
    COMPACT_KEEP_RECENT = int(os.getenv("COMPACT_KEEP_RECENT", "8"))
except ValueError:
    COMPACT_KEEP_RECENT = 8
if COMPACT_KEEP_RECENT < 2:
    COMPACT_KEEP_RECENT = 2
if COMPACT_KEEP_RECENT % 2:
    COMPACT_KEEP_RECENT += 1
try:
    MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "4000"))
except ValueError:
    MAX_SUMMARY_CHARS = 4000
try:
    COMPACT_MAX_TOKENS = int(os.getenv("COMPACT_MAX_TOKENS", "1500"))
except ValueError:
    COMPACT_MAX_TOKENS = 1500
COMPACT_TIMEOUT = 60.0

MAX_MESSAGE_LENGTH = 4096
MAX_INPUT_LENGTH = 4000

# Web Search Configuration
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
SEARXNG_URL = os.getenv("SEARXNG_URL", "").rstrip("/")
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "brave").lower()
try:
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
except ValueError:
    MAX_SEARCH_RESULTS = 5
try:
    MAX_SNIPPET_LEN = int(os.getenv("MAX_SNIPPET_LEN", "500"))
except ValueError:
    MAX_SNIPPET_LEN = 500

# Provider API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
VERCEL_API_KEY = os.getenv("VERCEL_API_KEY")

# Custom provider (OpenAI-compatible endpoint)
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY", "")
CUSTOM_BASE_URL = os.getenv("CUSTOM_BASE_URL", "").rstrip("/")
CUSTOM_DEFAULT_MODEL = os.getenv("CUSTOM_DEFAULT_MODEL", "")
OPENAI_SDK_USER_AGENT = os.getenv("OPENAI_SDK_USER_AGENT", "curl/8.7.1")

# Vision / OCR Configuration
NVIDIA_VISION_MODEL = os.getenv("OCR_VISION_MODEL", "gemini-flash-lite-latest")
VISION_BASE_URL = os.getenv(
    "VISION_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
).rstrip("/")
OCR_API_KEY = os.getenv("OCR_API_KEY") or GEMINI_API_KEY or NVIDIA_API_KEY
if (
    not os.getenv("OCR_API_KEY")
    and not GEMINI_API_KEY
    and NVIDIA_API_KEY
    and "generativelanguage.googleapis.com" in VISION_BASE_URL
):
    logger.warning(
        "⚠️ Only NVIDIA_API_KEY is set, but VISION_BASE_URL points at the Gemini "
        "endpoint — the NVIDIA key will likely be rejected there. "
        "Set OCR_API_KEY to a Gemini key, GEMINI_API_KEY, or an NVIDIA-compatible VISION_BASE_URL."
    )
try:
    MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
except ValueError:
    MAX_IMAGE_BYTES = 15 * 1024 * 1024

# Model validation cache file
VALIDATED_MODELS_CACHE = os.path.join(
    os.path.dirname(__file__), "validated_models.json"
)

# ============================================================================
# SYSTEM PROMPTS
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
    "BEHAVIOUR (in priority order — earlier wins over later):\n"
    "1. If you're not sure, say so explicitly and early. Never state uncertainty "
    "as if it were fact. It is ALWAYS better to say 'I don't know' or "
    "'I'm not certain, but...' than to state something wrong with confidence. "
    "This is the single most important rule.\n"
    "2. Be direct and practical\n"
    "3. When code is requested, use fenced code blocks with the language name\n\n"
    "WEB SEARCH TOOL:\n"
    "You have access to a `web_search` tool. Call it automatically whenever the user's "
    "question requires current, real-time, or time-sensitive information such as: "
    "news, prices, sports scores, weather, product availability, recent events, "
    "people's current status, or any fact you are not fully confident about. "
    "Do NOT call the tool for: coding help, math, creative writing, definitions, "
    "greetings, opinions, or general knowledge you can answer confidently. "
    "When search results are provided to you, base your answer ONLY on those results. "
    "Do NOT use your training data to fill in facts absent from the results — "
    "if the results lack enough info, say so clearly. "
    "If the search returns no usable results at all, say plainly 'I couldn't find "
    "current information on that' and stop — do not fall back to memory. "
    "Today's date is {date}."
)

_PROMPT_COMPACT_SUMMARIZER = (
    "You are a conversation memory compactor. Your job is to produce a "
    "dense, structured rolling memory of an ongoing chat so the assistant "
    "can keep helping the user without re-reading old messages.\n\n"
    "You will receive:\n"
    "  1. (Optional) the PREVIOUS rolling memory from earlier compactions.\n"
    "  2. NEW transcript chunk to fold in.\n\n"
    "Produce ONE updated memory document using EXACTLY these sections "
    "(omit a section entirely if it has zero content):\n\n"
    "USER PROFILE:\n"
    "- Name, role, location, language preference, tooling, skill level, persona — anything stable about the user.\n\n"
    "PREFERENCES & STYLE:\n"
    "- How they want responses (length, tone, format, code style, response language).\n\n"
    "ONGOING PROJECTS & TOPICS:\n"
    "- Projects, repos, files, products, or domains being worked on.\n\n"
    "KEY FACTS ESTABLISHED:\n"
    "- Concrete facts the assistant must remember: dates, names, numbers, IDs, URLs, configurations, versions.\n\n"
    "DECISIONS & CONCLUSIONS:\n"
    "- What was agreed/chosen/ruled out and WHY.\n\n"
    "OPEN QUESTIONS / PENDING:\n"
    "- Anything unresolved, promised follow-ups, things the user is still considering.\n\n"
    "IMPORTANT SNIPPETS:\n"
    "- Critical code, values, error messages, identifiers that may need to be quoted verbatim later.\n\n"
    "RULES:\n"
    "- Write in compact bullet points. No fluff, no preamble, no apology.\n"
    "- Preserve EXACT identifiers (names, paths, versions, error codes, numbers).\n"
    "- MERGE new info into the previous memory — don't just append. Update outdated facts. Drop facts the user has corrected or retracted.\n"
    "- Drop trivia: greetings, small talk, retries, transient errors that were resolved.\n"
    "- If a section has nothing, omit it entirely (no empty headers).\n"
    "- Keep total length under {max_chars} characters.\n"
    "- Output ONLY the memory document. No 'Here is the summary' wrapper, no markdown fences."
)

# Pre-built at startup
SYSTEM_PROMPT = _PROMPT_BASE  # date injected per-request via _build_system_prompt()


def _make_compact_summarizer_prompt() -> str:
    return _PROMPT_COMPACT_SUMMARIZER.format(max_chars=MAX_SUMMARY_CHARS)


def _build_system_prompt(session: Dict) -> str:
    """Return the system prompt with today's date and rolling memory appended (if any)."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    base = _PROMPT_BASE.format(date=today)
    summary = (session.get("summary") or "").strip()
    if not summary:
        return base
    return (
        f"{base}\n\n"
        "ROLLING CONVERSATION MEMORY (compacted from earlier messages — "
        "treat as authoritative background knowledge about this user and chat):\n"
        f"{summary}"
    )


def _serialize_history_for_summary(messages: List[Dict]) -> str:
    role_label = {"user": "USER", "assistant": "ASSISTANT", "system": "SYSTEM"}
    lines = []
    for msg in messages:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        label = role_label.get(msg.get("role", "?"), msg.get("role", "?").upper())
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines)


# ============================================================================
# FUNCTION CALLING — WEB SEARCH TOOL DEFINITION
#
# This single tool definition is passed to every provider that supports
# OpenAI-compatible tool/function calling. The model decides autonomously
# whether to call it; no separate SEARCH/NOSEARCH router call is needed.
# ============================================================================

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, real-time, or time-sensitive information. "
            "Call this when the user asks about recent news, live prices, sports scores, "
            "weather, product availability, people's current status, recent events, "
            "or any fact you are not fully confident about from training data. "
            "Do NOT call for coding help, math, definitions, greetings, or stable general knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A specific, self-contained web search query. "
                        "Include the subject noun and, for time-sensitive topics, the current year. "
                        "Aim for 4-8 words. Must be fully formed and searchable."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

# ============================================================================
# TRANSIENT API RETRY
# ============================================================================

_TRANSIENT_ERROR_KEYWORDS = (
    "rate limit",
    "too many requests",
    "429",
    "timeout",
    "timed out",
    "503",
    "502",
    "500",
    "529",
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
    "connection error",
    "connection reset",
)
_CHAT_MAX_RETRIES = 2
_CHAT_RETRY_BASE_DELAY = 3.0
_CHAT_ATTEMPT_TIMEOUT = 120.0


async def _chat_with_retry(
    provider,
    messages: list,
    model: Optional[str],
    enable_thinking: bool,
    tools: Optional[list] = None,
    cancel_event: Optional["asyncio.Event"] = None,
) -> any:
    """Call provider.chat() with automatic retry on transient errors.

    When tools is provided and the provider supports function calling,
    returns a ChatResult namedtuple with (.content, .tool_calls).
    Otherwise returns a plain string (backward compat for compaction calls).
    """
    e_for_delay: Optional[BaseException] = None
    for attempt in range(1, _CHAT_MAX_RETRIES + 2):
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Cancelled by /restart")
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    provider.chat,
                    messages=messages,
                    model=model,
                    enable_thinking=enable_thinking,
                    tools=tools,
                ),
                timeout=_CHAT_ATTEMPT_TIMEOUT,
            )
            return result if result is not None else ""
        except asyncio.TimeoutError:
            if attempt > _CHAT_MAX_RETRIES:
                raise
            logger.warning(
                f"[Bot] API call timed out after {_CHAT_ATTEMPT_TIMEOUT:.0f}s "
                f"(attempt {attempt}/{_CHAT_MAX_RETRIES + 1}) — retrying"
            )
            e_for_delay = asyncio.TimeoutError()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if attempt > _CHAT_MAX_RETRIES:
                raise
            error_lower = str(e).lower()
            if not any(kw in error_lower for kw in _TRANSIENT_ERROR_KEYWORDS):
                raise
            e_for_delay = e
        else:
            break

        delay = _CHAT_RETRY_BASE_DELAY * (2 ** (attempt - 1))
        logger.warning(
            f"[Bot] Transient API error (attempt {attempt}/{_CHAT_MAX_RETRIES + 1}): "
            f"{e_for_delay} — retrying in {delay:.0f}s"
        )
        if cancel_event:
            try:
                await asyncio.wait_for(cancel_event.wait(), timeout=delay)
                raise asyncio.CancelledError("Cancelled by /restart")
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(delay)
    return ""


# ============================================================================
# HISTORY UTILITIES
# ============================================================================


def _trim_history(history: list) -> list:
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history
    excess = len(history) - MAX_HISTORY_MESSAGES
    pairs_to_remove = (excess + 1) // 2
    messages_to_remove = pairs_to_remove * 2
    del history[:messages_to_remove]
    return history


async def _compact_history(
    session: Dict,
    provider,
    model: Optional[str],
    *,
    force: bool = False,
) -> bool:
    lock: Optional[asyncio.Lock] = session.get("compact_lock")
    if lock is None:
        lock = asyncio.Lock()
        session["compact_lock"] = lock

    async with lock:
        original_history = session["history"]
        history = original_history

        if not force and len(history) <= COMPACT_THRESHOLD:
            return False
        if len(history) < COMPACT_KEEP_RECENT + 2:
            return False

        split_at = len(history) - COMPACT_KEEP_RECENT
        if split_at % 2:
            split_at -= 1
        if split_at < 2:
            return False

        to_summarize = history[:split_at]
        previous_summary = (session.get("summary") or "").strip()

        sys_prompt = _make_compact_summarizer_prompt()
        user_parts = []
        if previous_summary:
            user_parts.append("PREVIOUS ROLLING MEMORY:\n" + previous_summary)
        user_parts.append(
            "NEW TRANSCRIPT CHUNK TO FOLD IN:\n"
            + _serialize_history_for_summary(to_summarize)
        )
        user_parts.append(
            "Produce the updated rolling memory now, following the format and rules above."
        )
        summary_msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

        logger.info(
            f"[Compact] Compressing {len(to_summarize)} old msgs → memory, "
            f"keeping ≥{len(history) - split_at} recent verbatim (force={force}, "
            f"prev_summary={len(previous_summary)} chars)"
        )

        try:
            # Compaction calls always use plain text — no tools
            raw = await asyncio.wait_for(
                asyncio.to_thread(
                    provider.chat,
                    messages=summary_msgs,
                    model=model,
                    enable_thinking=False,
                    tools=None,
                    max_tokens=COMPACT_MAX_TOKENS,
                ),
                timeout=COMPACT_TIMEOUT,
            )
            # provider.chat may return a ChatResult or a plain string here
            new_summary = raw.content if hasattr(raw, "content") else (raw or "")
        except asyncio.CancelledError:
            logger.info("[Compact] Cancelled by user action.")
            raise
        except Exception as e:
            logger.warning(
                f"[Compact] Summariser call failed: {e}. Falling back to hard trim."
            )
            if session["history"] is original_history:
                _trim_history(original_history)
            return False

        if session["history"] is not original_history:
            logger.info(
                "[Compact] Session was reset during compaction; discarding stale summary."
            )
            return False

        new_summary = strip_thinking_tags(
            new_summary or "", keep_thinking=False
        ).strip()
        if not new_summary:
            logger.warning(
                "[Compact] Empty summary returned. Falling back to hard trim."
            )
            _trim_history(original_history)
            return False

        if len(new_summary) > MAX_SUMMARY_CHARS:
            new_summary = new_summary[:MAX_SUMMARY_CHARS].rstrip() + "\n…[truncated]"

        del original_history[:split_at]
        session["summary"] = new_summary
        logger.info(
            f"[Compact] Done. summary={len(new_summary)} chars, "
            f"history={len(original_history)} msgs verbatim"
        )
        return True


async def _maybe_compact(session: Dict, provider, model: Optional[str]) -> None:
    try:
        if len(session["history"]) > COMPACT_THRESHOLD:
            await _compact_history(session, provider, model)
        else:
            _trim_history(session["history"])
    except Exception as e:
        logger.warning(f"[Compact] _maybe_compact error: {e}")
        _trim_history(session["history"])


def _schedule_compact(session: Dict, provider, model: Optional[str]) -> None:
    try:
        task = asyncio.create_task(_maybe_compact(session, provider, model))
    except RuntimeError:
        _trim_history(session["history"])
        return
    bg: set = session.setdefault("_bg_tasks", set())
    bg.add(task)
    task.add_done_callback(bg.discard)


def _cancel_bg_tasks(session: Dict) -> int:
    bg = session.get("_bg_tasks")
    if not bg:
        return 0
    cancelled = 0
    for task in list(bg):
        if not task.done():
            task.cancel()
            cancelled += 1
    if cancelled:
        logger.info(f"[Compact] Cancelled {cancelled} in-flight background task(s).")
    return cancelled


async def reply_text_safe(message, text: str):
    try:
        await message.reply_text(text, parse_mode="Markdown")
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
    match = re.search(r"(\d+\.?\d*)b", model_id.lower())
    if match:
        size = float(match.group(1))
        return int(size) if size >= 1 else 0
    return 0


def get_model_capability_score(model_id: str) -> tuple:
    model_lower = model_id.lower()
    param_size = extract_parameter_size(model_id)

    flagship_patterns = [
        "gpt-4",
        "claude-3-opus",
        "claude-3.5",
        "claude-4",
        "gemini-2.0",
        "gemini-pro",
        "llama-3.3",
        "llama-3.2",
        "qwen-2.5-72b",
        "hermes-3-llama-3.1-405b",
        "gpt-oss",
    ]
    if any(pattern in model_lower for pattern in flagship_patterns):
        return (0, -param_size if param_size > 0 else 0, model_id)

    if param_size >= 100:
        return (1, -param_size, model_id)
    if param_size >= 50:
        return (2, -param_size, model_id)
    if param_size >= 20:
        return (3, -param_size, model_id)
    if param_size >= 10:
        return (4, -param_size, model_id)
    if param_size >= 5:
        return (5, -param_size, model_id)
    if param_size >= 1:
        return (6, -param_size, model_id)

    if any(x in model_lower for x in ["exp", "experimental", "preview", "beta"]):
        version_match = re.search(r"(\d+\.\d+)", model_id)
        if version_match and float(version_match.group(1)) >= 2.0:
            return (0, 0, model_id)
        return (7, 0, model_id)

    return (8, 0, model_id)


# ============================================================================
# PROVIDER ABSTRACTION LAYER
# ============================================================================


def strip_thinking_tags(text: str, keep_thinking: bool = False) -> str:
    patterns = [
        (r"<think>(.*?)</think>", "thinking"),
        (r"<thinking>(.*?)</thinking>", "thinking"),
        (r"<reasoning>(.*?)</reasoning>", "reasoning"),
        (r"<context>(.*?)</context>", "context"),
    ]

    if keep_thinking:
        result = text
        for pattern, label in patterns:
            matches = re.finditer(pattern, result, re.DOTALL | re.IGNORECASE)
            for match in reversed(list(matches)):
                thinking_content = match.group(1).strip()
                formatted = f"💭 *{label.title()}:*\n{thinking_content}\n\n"
                result = result[: match.start()] + formatted + result[match.end() :]
        return result.strip()
    else:
        result = text
        for pattern, _ in patterns:
            result = re.sub(pattern, "", result, flags=re.DOTALL | re.IGNORECASE)
        return result.strip()


class ChatResult:
    """Unified return type from provider.chat() when tools are involved.

    .content    — the assistant's text reply (may be empty if tool_calls fired)
    .tool_calls — list of dicts: [{"name": str, "arguments": dict, "id": str}]
                  empty list when the model answered directly
    """

    __slots__ = ("content", "tool_calls")

    def __init__(self, content: str = "", tool_calls: Optional[list] = None):
        self.content = content
        self.tool_calls = tool_calls or []

    def __bool__(self):
        return bool(self.content or self.tool_calls)


class AIProvider(ABC):
    def __init__(self):
        self._models_lock = threading.Lock()
        self._cached_models = None
        self._last_refresh = 0

    @abstractmethod
    def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
    ) -> "ChatResult | str":
        """Generate a chat response from the AI.

        If tools is provided and the provider supports function calling,
        return a ChatResult.  Providers that do NOT support tools must
        ignore the parameter and return a plain string (or ChatResult with
        no tool_calls) so the caller falls back gracefully.
        """
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

    def supports_function_calling(self) -> bool:
        """Override to True in providers whose API supports OpenAI-style tools."""
        return False

    def test_model(self, model_id: str) -> Tuple[bool, str]:
        try:
            response = self.chat([{"role": "user", "content": "Hi"}], model=model_id)
            content = response.content if isinstance(response, ChatResult) else response
            if content is not None and len(content) > 0:
                return (True, "success")
            return (False, "unknown")
        except Exception as e:
            error_str = str(e).lower()
            logger.debug(f"Model {model_id} validation failed: {e}")
            if any(
                k in error_str
                for k in ["rate limit", "too many requests", "429", "quota"]
            ):
                return (False, "rate_limit")
            elif any(
                k in error_str
                for k in [
                    "not found",
                    "404",
                    "does not exist",
                    "invalid model",
                    "not available",
                    "not supported",
                    "no access",
                ]
            ):
                return (False, "not_available")
            else:
                return (False, "unknown")


# ── Helper: parse OpenAI-SDK tool_calls into our ChatResult format ──────────

def _parse_openai_tool_calls(response) -> List[Dict]:
    """Extract tool calls from an OpenAI-SDK completion response."""
    calls = []
    msg = response.choices[0].message if response.choices else None
    if msg is None:
        return calls
    raw_calls = getattr(msg, "tool_calls", None) or []
    for tc in raw_calls:
        try:
            arguments = json.loads(tc.function.arguments or "{}")
        except (json.JSONDecodeError, AttributeError):
            arguments = {}
        calls.append({
            "id": getattr(tc, "id", None) or "call_0",
            "name": tc.function.name,
            "arguments": arguments,
        })
    return calls


def _openai_tool_result_message(tool_call: Dict, result_text: str) -> Dict:
    """Build the tool result message for OpenAI-compatible providers."""
    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": result_text,
    }


def _openai_assistant_tool_call_message(tool_calls_raw) -> Dict:
    """Reconstruct the assistant message that requested tool calls (for multi-turn)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls_raw,
    }


class GroqProvider(AIProvider):
    def __init__(self, api_key: str):
        super().__init__()
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.default_model = "openai/gpt-oss-120b"

    def supports_function_calling(self) -> bool:
        return True

    def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
    ) -> ChatResult:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get("role") == "system" for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        kwargs = dict(
            messages=chat_messages,
            model=model,
            temperature=TEMPERATURE,
            max_tokens=max_tokens or MAX_TOKENS,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        tool_calls = _parse_openai_tool_calls(response) if tools else []
        content = response.choices[0].message.content if response.choices else ""
        content = strip_thinking_tags(content or "", keep_thinking=enable_thinking)
        return ChatResult(content=content, tool_calls=tool_calls)

    def get_available_models(self) -> List[Dict[str, str]]:
        with self._models_lock:
            if self._cached_models is not None:
                return self._cached_models
            try:
                models_response = self.client.models.list()
                chat_models = []
                for model in models_response.data:
                    if model.active and hasattr(model, "id"):
                        chat_models.append(
                            {"id": model.id, "name": model.id.replace("-", " ").title()}
                        )
                chat_models.sort(key=lambda m: get_model_capability_score(m["id"]))
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
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B 32K"},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant"},
            {"id": "gemma2-9b-it", "name": "Gemma 2 9B IT"},
        ]

    def get_name(self) -> str:
        return "Groq"

    def get_default_model(self) -> str:
        return self.default_model

    def supports_thinking(self, model_id: str) -> bool:
        reasoning_keywords = ["reasoning", "think", "deepseek", "qwq", "r1"]
        return any(keyword in model_id.lower() for keyword in reasoning_keywords)


class GeminiProvider(AIProvider):
    """Gemini via its native SDK.

    The native SDK does not expose an easy OpenAI-compatible tool_calls flow,
    so we fall back to system-prompt-based search context injection for this
    provider.  Function calling is handled at the OpenAI-compat layer if the
    user switches to Gemini via OpenRouter or a custom endpoint instead.
    """

    def __init__(self, api_key: str):
        super().__init__()
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.genai = genai
        self.default_model = "gemini-flash-lite-latest"

    def supports_function_calling(self) -> bool:
        # Native SDK path does not use our OpenAI tool-call format
        return False

    def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
    ) -> ChatResult:
        if not messages:
            raise ValueError("Messages list cannot be empty")
        model_name = model or self.default_model
        system_msg = next(
            (m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT
        )
        gen_model = self.genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": TEMPERATURE,
                "max_output_tokens": max_tokens or MAX_TOKENS,
            },
            system_instruction=system_msg,
        )
        chat_history = []
        for msg in messages[:-1]:
            if msg["role"] == "system":
                continue
            # Skip tool result messages — not native-SDK-compatible
            if msg.get("role") == "tool":
                continue
            role = "user" if msg["role"] == "user" else "model"
            content = msg.get("content") or ""
            if content:
                chat_history.append({"role": role, "parts": [content]})

        last_content = messages[-1].get("content") or ""
        chat = gen_model.start_chat(history=chat_history)
        response_text = chat.send_message(last_content).text if last_content else ""
        return ChatResult(
            content=strip_thinking_tags(response_text, keep_thinking=enable_thinking),
            tool_calls=[],
        )

    def get_available_models(self) -> List[Dict[str, str]]:
        with self._models_lock:
            if self._cached_models is not None:
                return self._cached_models
            try:
                chat_models = []
                for model in self.genai.list_models():
                    if "generateContent" not in model.supported_generation_methods:
                        continue
                    model_id = model.name.replace("models/", "")
                    if any(
                        x in model_id.lower() for x in ["vision", "embedding", "aqa"]
                    ):
                        continue
                    name = (
                        model.display_name
                        if hasattr(model, "display_name")
                        else model_id.replace("-", " ").title()
                    )
                    chat_models.append({"id": model_id, "name": name})
                chat_models.sort(key=lambda m: get_model_capability_score(m["id"]))
                self._cached_models = chat_models
                logger.info(f"✅ Gemini: Detected {len(chat_models)} available models")
                return chat_models
            except Exception as e:
                logger.warning(f"⚠️ Gemini: Could not fetch models: {e}")
                return self._get_fallback_models()

    def _get_fallback_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "gemini-flash-lite-latest", "name": "Gemini Flash Lite (Latest)"},
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
            {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash Experimental"},
        ]

    def get_name(self) -> str:
        return "Gemini"

    def get_default_model(self) -> str:
        return self.default_model

    def supports_thinking(self, model_id: str) -> bool:
        reasoning_keywords = ["reasoning", "think", "deepseek", "qwq", "r1", "gemini-2"]
        return any(keyword in model_id.lower() for keyword in reasoning_keywords)


class OpenRouterProvider(AIProvider):
    def __init__(self, api_key: str):
        super().__init__()
        from openai import OpenAI

        client_kwargs = {}
        if OPENAI_SDK_USER_AGENT:
            client_kwargs["default_headers"] = {"User-Agent": OPENAI_SDK_USER_AGENT}
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=api_key, **client_kwargs
        )
        self.api_key = api_key
        self.default_model = "openrouter/free"

    def supports_function_calling(self) -> bool:
        return True

    def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
    ) -> ChatResult:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get("role") == "system" for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        kwargs = dict(
            model=model,
            messages=chat_messages,
            temperature=TEMPERATURE,
            max_tokens=max_tokens or MAX_TOKENS,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        tool_calls = _parse_openai_tool_calls(response) if tools else []
        content = response.choices[0].message.content if response.choices else ""
        content = strip_thinking_tags(content or "", keep_thinking=enable_thinking)
        return ChatResult(content=content, tool_calls=tool_calls)

    def get_available_models(self) -> List[Dict[str, str]]:
        with self._models_lock:
            if self._cached_models is not None:
                return self._cached_models
            try:
                response = requests.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5,
                )
                response.raise_for_status()
                free_models = []
                for model in response.json().get("data", []):
                    model_id = model.get("id", "")
                    pricing = model.get("pricing", {})
                    context_length = model.get("context_length", 0)
                    if ":free" not in model_id.lower() or context_length <= 0:
                        continue
                    try:
                        prompt_price = float(pricing.get("prompt") or 0)
                        compl_price = float(pricing.get("completion") or 0)
                    except (TypeError, ValueError):
                        continue
                    if prompt_price != 0.0 or compl_price != 0.0:
                        continue
                    name = model.get("name", model_id)
                    name = (
                        name.replace(" (free)", "").replace(" (Free)", "") + " (Free)"
                    )
                    free_models.append(
                        {"id": model_id, "name": name, "context": context_length}
                    )
                free_models.sort(key=lambda m: get_model_capability_score(m["id"]))
                self._cached_models = free_models[:15]
                logger.info(
                    f"✅ OpenRouter: Detected {len(self._cached_models)} free models"
                )
                return self._cached_models
            except Exception as e:
                logger.warning(f"⚠️ OpenRouter: Could not fetch models: {e}")
                return self._get_fallback_models()

    def _get_fallback_models(self) -> List[Dict[str, str]]:
        return [
            {
                "id": "meta-llama/llama-3.3-70b-instruct:free",
                "name": "Llama 3.3 70B (Free)",
            },
            {
                "id": "nousresearch/hermes-3-llama-3.1-405b:free",
                "name": "Hermes 3 405B (Free)",
            },
            {
                "id": "google/gemini-2.0-flash-exp:free",
                "name": "Gemini 2.0 Flash Exp (Free)",
            },
            {"id": "qwen/qwen-2.5-72b-instruct:free", "name": "Qwen 2.5 72B (Free)"},
            {"id": "mistralai/mistral-7b-instruct:free", "name": "Mistral 7B (Free)"},
        ]

    def get_name(self) -> str:
        return "OpenRouter"

    def get_default_model(self) -> str:
        return self.default_model

    def supports_thinking(self, model_id: str) -> bool:
        reasoning_keywords = ["reasoning", "think", "deepseek", "qwq", "r1"]
        return any(keyword in model_id.lower() for keyword in reasoning_keywords)


class CerebrasProvider(AIProvider):
    def __init__(self, api_key: str):
        super().__init__()
        from cerebras.cloud.sdk import Cerebras

        self.client = Cerebras(api_key=api_key)
        self.default_model = "llama3.1-8b"

    def supports_function_calling(self) -> bool:
        return True

    def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
    ) -> ChatResult:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get("role") == "system" for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        kwargs = dict(
            messages=chat_messages,
            model=model,
            temperature=TEMPERATURE,
            max_tokens=max_tokens or MAX_TOKENS,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        tool_calls = _parse_openai_tool_calls(response) if tools else []
        content = response.choices[0].message.content if response.choices else ""
        content = strip_thinking_tags(content or "", keep_thinking=enable_thinking)
        return ChatResult(content=content, tool_calls=tool_calls)

    def get_available_models(self) -> List[Dict[str, str]]:
        with self._models_lock:
            if self._cached_models is not None:
                return self._cached_models
            try:
                chat_models = []
                for model in self.client.models.list().data:
                    if hasattr(model, "id"):
                        chat_models.append(
                            {"id": model.id, "name": model.id.replace("-", " ").title()}
                        )
                chat_models.sort(key=lambda m: get_model_capability_score(m["id"]))
                self._cached_models = chat_models
                logger.info(
                    f"✅ Cerebras: Detected {len(chat_models)} available models"
                )
                return chat_models
            except Exception as e:
                logger.warning(f"⚠️ Cerebras: Could not fetch models: {e}")
                return self._get_fallback_models()

    def _get_fallback_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "gpt-oss-120b", "name": "GPT-OSS 120B"},
            {"id": "llama3.1-8b", "name": "Llama 3.1 8B"},
            {"id": "llama-3.3-70b", "name": "Llama 3.3 70B"},
        ]

    def get_name(self) -> str:
        return "Cerebras"

    def get_default_model(self) -> str:
        return self.default_model

    def supports_thinking(self, model_id: str) -> bool:
        reasoning_keywords = ["reasoning", "think", "deepseek", "qwq", "r1"]
        return any(keyword in model_id.lower() for keyword in reasoning_keywords)


class NvidiaProvider(AIProvider):
    MODELS_WITHOUT_THINKING = {
        "qwen/qwen3-coder-480b-a35b-instruct",
        "openai/gpt-oss-120b",
        "minimaxai/minimax-m2.1",
        "minimaxai/minimax-m2",
    }

    def __init__(self, api_key: str):
        super().__init__()
        from openai import OpenAI

        client_kwargs = {}
        if OPENAI_SDK_USER_AGENT:
            client_kwargs["default_headers"] = {"User-Agent": OPENAI_SDK_USER_AGENT}
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            **client_kwargs,
        )
        self.default_model = "nvidia/nemotron-3-super-120b-a12b"

    def supports_thinking(self, model_id: str) -> bool:
        return model_id not in self.MODELS_WITHOUT_THINKING

    def supports_function_calling(self) -> bool:
        return True

    def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
    ) -> ChatResult:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get("role") == "system" for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        if enable_thinking and self.supports_thinking(model) and not tools:
            # Streaming thinking path — incompatible with tool calls
            response = self.client.chat.completions.create(
                messages=chat_messages,
                model=model,
                temperature=TEMPERATURE,
                max_tokens=max_tokens or MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True,
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
            return ChatResult(content=full + "".join(content_parts), tool_calls=[])
        else:
            extra_body = (
                {"chat_template_kwargs": {"thinking": False}}
                if self.supports_thinking(model)
                else {}
            )
            kwargs = dict(
                messages=chat_messages,
                model=model,
                temperature=TEMPERATURE,
                max_tokens=max_tokens or MAX_TOKENS,
                stream=False,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            tool_calls = _parse_openai_tool_calls(response) if tools else []
            content = response.choices[0].message.content if response.choices else ""
            if not content and not tool_calls:
                raise ValueError("API returned empty response.")
            return ChatResult(content=content or "", tool_calls=tool_calls)

    def get_available_models(self) -> List[Dict[str, str]]:
        with self._models_lock:
            if self._cached_models:
                return self._cached_models
            self._cached_models = self._get_fallback_models()
            logger.info(
                f"✅ NVIDIA: Using {len(self._cached_models)} hand-picked models"
            )
            return self._cached_models

    def _get_fallback_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B (Stable)"},
            {"id": "qwen/qwen3-coder-480b-a35b-instruct", "name": "Qwen3 Coder 480B"},
            {"id": "minimaxai/minimax-m2.1", "name": "MiniMax M2.1"},
            {"id": "minimaxai/minimax-m2", "name": "MiniMax M2"},
            {
                "id": "nvidia/nemotron-3-super-120b-a12b",
                "name": "NVIDIA-Nemotron-3-Super-120B-A12B 💭",
            },
            {"id": "deepseek-ai/deepseek-v3.2", "name": "DeepSeek V3.2 💭"},
            {
                "id": "deepseek-ai/deepseek-v3.1-terminus",
                "name": "DeepSeek V3.1 Terminus 💭",
            },
            {"id": "qwen/qwen3-235b-a22b", "name": "Qwen3 235B 💭"},
            {"id": "moonshotai/kimi-k2.5", "name": "Kimi K2.5 💭"},
            {"id": "z-ai/glm4.7", "name": "GLM 4.7 💭"},
            {"id": "z-ai/glm5", "name": "GLM 5 💭"},
        ]

    def get_name(self) -> str:
        return "NVIDIA"

    def get_default_model(self) -> str:
        return self.default_model


class CustomProvider(AIProvider):
    def __init__(self, api_key: str, base_url: str, default_model: str):
        super().__init__()
        from openai import OpenAI

        user_agent = OPENAI_SDK_USER_AGENT
        client_kwargs = {}
        if user_agent:
            client_kwargs["default_headers"] = {"User-Agent": user_agent}
        self.client = OpenAI(base_url=base_url, api_key=api_key, **client_kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model

    def supports_thinking(self, model_id: str) -> bool:
        return True

    def supports_function_calling(self) -> bool:
        return True

    def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
    ) -> ChatResult:
        model = model or self.default_model
        chat_messages = messages.copy()
        if not any(msg.get("role") == "system" for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        kwargs = dict(
            model=model,
            messages=chat_messages,
            temperature=TEMPERATURE,
            max_tokens=max_tokens or MAX_TOKENS,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        tool_calls = _parse_openai_tool_calls(response) if tools else []
        content = response.choices[0].message.content if response.choices else ""
        return ChatResult(
            content=strip_thinking_tags(content or "", keep_thinking=enable_thinking),
            tool_calls=tool_calls,
        )

    def get_available_models(self) -> List[Dict[str, str]]:
        with self._models_lock:
            if self._cached_models is not None:
                return self._cached_models
            try:
                models_response = self.client.models.list()
                chat_models = [
                    {"id": m.id, "name": m.id.replace("-", " ").title()}
                    for m in models_response.data
                    if hasattr(m, "id")
                ]
                if chat_models:
                    chat_models.sort(key=lambda m: get_model_capability_score(m["id"]))
                    self._cached_models = chat_models
                    logger.info(
                        f"✅ Custom: Detected {len(chat_models)} available models"
                    )
                    return self._cached_models
            except Exception as e:
                logger.warning(
                    f"⚠️ Custom: Could not fetch model list from {self.base_url}: {e}"
                )
            self._cached_models = [
                {
                    "id": self.default_model,
                    "name": self.default_model.replace("-", " ").title(),
                }
            ]
            logger.info("✅ Custom: Using configured default model as fallback")
            return self._cached_models

    def get_name(self) -> str:
        return "Custom"

    def get_default_model(self) -> str:
        return self.default_model


class VercelProvider(CustomProvider):
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://ai-gateway.vercel.sh/v1",
            default_model="perplexity/sonar",
        )

    def get_name(self) -> str:
        return "Vercel"

    def supports_thinking(self, model_id: str) -> bool:
        # Default gateway model (Perplexity Sonar) emits no thinking tags;
        # CustomProvider returns True unconditionally, which would be misleading.
        return False


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
                self.providers["groq"] = GroqProvider(GROQ_API_KEY)
                logger.info("✅ Groq provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq: {e}")
        if GEMINI_API_KEY:
            try:
                self.providers["gemini"] = GeminiProvider(GEMINI_API_KEY)
                logger.info("✅ Gemini provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini: {e}")
        if OPENROUTER_API_KEY:
            try:
                self.providers["openrouter"] = OpenRouterProvider(OPENROUTER_API_KEY)
                logger.info("✅ OpenRouter provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenRouter: {e}")
        if CEREBRAS_API_KEY:
            try:
                self.providers["cerebras"] = CerebrasProvider(CEREBRAS_API_KEY)
                logger.info("✅ Cerebras provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Cerebras: {e}")
        if NVIDIA_API_KEY:
            try:
                self.providers["nvidia"] = NvidiaProvider(NVIDIA_API_KEY)
                logger.info("✅ NVIDIA provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize NVIDIA: {e}")
        if VERCEL_API_KEY:
            try:
                self.providers["vercel"] = VercelProvider(VERCEL_API_KEY)
                logger.info("✅ Vercel provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Vercel: {e}")
        if CUSTOM_API_KEY and CUSTOM_BASE_URL and CUSTOM_DEFAULT_MODEL:
            try:
                self.providers["custom"] = CustomProvider(
                    CUSTOM_API_KEY, CUSTOM_BASE_URL, CUSTOM_DEFAULT_MODEL
                )
                logger.info("✅ Custom provider initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Custom: {e}")
        if not self.providers:
            raise ValueError("No AI providers available! Set at least one API key.")

        if DEFAULT_PROVIDER not in self.providers:
            logger.warning(
                f"⚠️ DEFAULT_PROVIDER='{DEFAULT_PROVIDER}' is not available — "
                f"will fall back to '{self.list_providers()[0]}'"
            )

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


# ============================================================================
# SESSION HELPERS
# ============================================================================


def get_user_session(user_id: str) -> Dict:
    now = time.time()

    if user_id not in user_sessions:
        if SEARCH_ENGINE == "searxng" and SEARXNG_URL:
            default_engine = "searxng"
        elif SEARCH_ENGINE == "brave" and BRAVE_API_KEY:
            default_engine = "brave"
        else:
            default_engine = "duckduckgo"
            if SEARCH_ENGINE not in ("duckduckgo", "ddg", ""):
                logger.info(
                    f"[Session] SEARCH_ENGINE='{SEARCH_ENGINE}' unavailable "
                    "(missing key/URL) — defaulting to DuckDuckGo. "
                    "The user can change it with /web."
                )
        user_sessions[user_id] = {
            "provider": provider_manager.get_default_provider(),
            "models": {},
            "history": [],
            "summary": "",
            "thinking_enabled": False,
            "web_search": True,
            "search_engine": default_engine,
            "last_seen": now,
            "cancel_event": asyncio.Event(),
            "compact_lock": asyncio.Lock(),
        }
    else:
        user_sessions[user_id]["last_seen"] = now
    return user_sessions[user_id]


def is_user_allowed(user_id: str) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def _resolve_provider(session: Dict):
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

_validated_cache: Optional[Dict] = None


def load_validated_models() -> Dict:
    global _validated_cache
    if _validated_cache is not None:
        return _validated_cache
    try:
        if os.path.exists(VALIDATED_MODELS_CACHE):
            with open(VALIDATED_MODELS_CACHE, "r") as f:
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
        with open(VALIDATED_MODELS_CACHE, "w") as f:
            json.dump(validated, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save validated models cache: {e}")


def _ensure_provider_entry(validated: Dict, provider_name: str) -> Dict:
    if provider_name not in validated:
        validated[provider_name] = {"working": [], "failed": []}
    if isinstance(validated[provider_name], list):
        validated[provider_name] = {"working": validated[provider_name], "failed": []}
    return validated


def get_validated_models(provider_name: str) -> List[str]:
    d = load_validated_models().get(provider_name, {})
    return d.get("working", []) if isinstance(d, dict) else d


def get_failed_models(provider_name: str) -> List[str]:
    d = load_validated_models().get(provider_name, {})
    return d.get("failed", []) if isinstance(d, dict) else []


def add_validated_model(provider_name: str, model_id: str):
    validated = _ensure_provider_entry(load_validated_models(), provider_name)
    if model_id not in validated[provider_name]["working"]:
        validated[provider_name]["working"].append(model_id)
        if model_id in validated[provider_name]["failed"]:
            validated[provider_name]["failed"].remove(model_id)
        save_validated_models(validated)


def add_failed_model(provider_name: str, model_id: str):
    validated = _ensure_provider_entry(load_validated_models(), provider_name)
    if model_id not in validated[provider_name]["failed"]:
        validated[provider_name]["failed"].append(model_id)
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
# WEB SEARCH BACKENDS — Brave, SearXNG, DuckDuckGo
# (These functions are unchanged — function calling only changes HOW the
#  model triggers them, not how the actual HTTP requests work.)
# ============================================================================


def _brave_search_sync(query: str) -> list:
    if not BRAVE_API_KEY:
        return []
    q = urllib.parse.quote_plus(query)
    try:
        r = requests.get(
            f"https://api.search.brave.com/res/v1/web/search?q={q}&count={MAX_SEARCH_RESULTS}",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            timeout=10,
        )
        r.raise_for_status()
        snippets = []
        for item in r.json().get("web", {}).get("results", []):
            title = item.get("title", "").strip()
            desc = item.get("description", "").strip()[:MAX_SNIPPET_LEN]
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
            body = r.get("body", "").strip()[:MAX_SNIPPET_LEN]
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
            timeout=10,
        )
        r.raise_for_status()
        snippets = []
        for item in r.json().get("results", [])[:MAX_SEARCH_RESULTS]:
            title = item.get("title", "").strip()
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
        logger.warning(
            f"[Search] SearXNG returned no results for '{query}' — falling back to DuckDuckGo"
        )
    if engine == "brave" and BRAVE_API_KEY:
        results = await asyncio.to_thread(_brave_search_sync, query)
        if results:
            return results
        logger.warning(
            f"[Search] Brave returned no results for '{query}' — falling back to DuckDuckGo"
        )
    return await asyncio.to_thread(_duckduckgo_search_sync, query)


def _format_search_results(query: str, snippets: list) -> str:
    """Format search snippets as a tool result string."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    parts = [f"Today is {today}. Web search results for '{query}':"]
    for i, snip in enumerate(snippets, 1):
        parts.append(f"{i}. {snip}")
    parts.append(
        "\nAnswer using ONLY the information above. If it does not answer the "
        "question, say so plainly instead of filling gaps from memory."
    )
    return "\n".join(parts)


# ============================================================================
# FUNCTION-CALLING SEARCH FLOW
#
# FLOW (providers that support function calling):
#
#   Call 1 — main call with WEB_SEARCH_TOOL attached:
#     • Model answers directly           → done in 1 call (no search needed)
#     • Model emits tool_calls           → execute search, inject result, Call 2
#
#   Call 2 — follow-up with tool result in messages:
#     • Model answers using search data  → done
#
# FALLBACK (providers without function calling, e.g. Gemini native SDK):
#   Same as before: single direct AI call. The system prompt already instructs
#   the model to mention if it's uncertain; the user can enable /web on to
#   append search results via a manual query when needed.
#
# ============================================================================


async def _execute_tool_call(tool_call: Dict, engine: str) -> str:
    """Run the tool the model requested and return the result as a string."""
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})

    if name == "web_search":
        query = args.get("query", "").strip()
        if not query:
            return "Error: web_search called with empty query."
        logger.info(f"[FuncCall] web_search('{query}') via {engine}")
        snippets = await web_search(query, engine)
        if not snippets:
            return (
                f"The web search for '{query}' returned no usable results. "
                "Tell the user plainly that you could not find current "
                "information on this topic, and do NOT answer from memory."
            )
        return _format_search_results(query, snippets)

    return f"Unknown tool: {name}"


async def _chat_with_function_calling(
    provider,
    messages: list,
    model: Optional[str],
    enable_thinking: bool,
    web_on: bool,
    engine: str,
    cancel_event: Optional["asyncio.Event"] = None,
) -> str:
    """Single-entry-point for the function-calling search flow.

    For providers that support function calling:
      1. Call model with WEB_SEARCH_TOOL; if it picks up tool_calls → execute
         search and make a second call with the result injected.
      2. If model answers directly in Call 1, return that immediately.

    For providers without function calling:
      Make one direct call without tools (same as the old no-search path).
      The system prompt still encourages the model to be honest about
      uncertainty, and web_on state is displayed in /status for transparency.

    Returns the final text response string.
    """
    use_tools = web_on and provider.supports_function_calling()
    tools = [WEB_SEARCH_TOOL] if use_tools else None

    # ── Call 1 ────────────────────────────────────────────────────────────────
    result = await _chat_with_retry(
        provider,
        messages=messages,
        model=model,
        enable_thinking=enable_thinking,
        tools=tools,
        cancel_event=cancel_event,
    )

    # Normalise: older providers may return a plain string
    if isinstance(result, str):
        return result or "⚠️ The AI returned an empty response."

    chat_result: ChatResult = result

    # ── No tool call — model answered directly ────────────────────────────────
    if not chat_result.tool_calls:
        return chat_result.content or "⚠️ The AI returned an empty response."

    # ── Tool call(s) — execute each, then Call 2 ─────────────────────────────
    if cancel_event and cancel_event.is_set():
        raise asyncio.CancelledError("Cancelled by /restart")

    # We only handle one tool call per turn (web_search). If the model somehow
    # emits multiple, we process the first and ignore the rest — this is safe
    # because web_search is the only registered tool.
    tool_call = chat_result.tool_calls[0]
    tool_result_text = await _execute_tool_call(tool_call, engine)

    if cancel_event and cancel_event.is_set():
        raise asyncio.CancelledError("Cancelled by /restart")

    # Reconstruct the conversation thread for Call 2.
    # OpenAI spec: after an assistant message with tool_calls, we must append
    # the assistant message (with the tool_calls list) and then the tool result.
    follow_up_messages = list(messages)

    # Rebuild the raw tool_calls structure the SDK expects
    raw_tool_calls = []
    for tc in chat_result.tool_calls:
        raw_tool_calls.append({
            "id": tc.get("id") or "call_0",
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc.get("arguments", {})),
            },
        })

    follow_up_messages.append({
        "role": "assistant",
        "content": chat_result.content or None,
        "tool_calls": raw_tool_calls,
    })
    follow_up_messages.append({
        "role": "tool",
        "tool_call_id": tool_call.get("id", "call_0"),
        "content": tool_result_text,
    })

    logger.info("[FuncCall] Tool result injected, making Call 2 for final answer")

    # ── Call 2 — final answer with search results ─────────────────────────────
    result2 = await _chat_with_retry(
        provider,
        messages=follow_up_messages,
        model=model,
        enable_thinking=enable_thinking,
        tools=None,  # No more tool calls in the follow-up
        cancel_event=cancel_event,
    )

    if isinstance(result2, str):
        return result2 or "⚠️ The AI returned an empty response."

    return result2.content or "⚠️ The AI returned an empty response."


# ============================================================================
# COMMAND HANDLERS
# ============================================================================


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    session = get_user_session(user_id)
    _, provider = _resolve_provider(session)
    web_status = "ON" if session.get("web_search") else "OFF"
    fc_status = "✅ function calling" if provider.supports_function_calling() else "⚠️ direct (no tool support)"
    await update.message.reply_text(
        f"🤖 Hello! I'm your Multi-Provider AI assistant.\n\n"
        f"📡 Current Provider: *{provider.get_name()}*\n"
        f"🔧 Available Providers: {', '.join(provider_manager.list_providers())}\n"
        f"🌐 Web Search: {web_status} ({fc_status})\n\n"
        f"Just send me a message or a photo!\n\n"
        f"*Commands:*\n"
        f"/status - Show current settings & stats\n"
        f"/provider - Switch AI provider\n"
        f"/models - List available models\n"
        f"/model - Switch model\n"
        f"/web - Toggle web search\n"
        f"/refresh - Refresh model lists\n"
        f"/clear - Clear conversation\n"
        f"/restart - Cancel any stuck/pending AI request\n"
        f"/help - Show help",
        parse_mode="Markdown",
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    session = get_user_session(user_id)
    _cancel_bg_tasks(session)
    session["history"] = []
    session["summary"] = ""
    await update.message.reply_text(
        "🗑️ Conversation history and rolling memory cleared!"
    )


async def refresh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_user_allowed(str(update.effective_user.id)):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    await update.message.reply_text("🔄 Refreshing model lists...")
    try:
        await asyncio.to_thread(provider_manager.refresh_models)
        await update.message.reply_text(
            "✅ Model lists refreshed!\n\nUse /models to see latest."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error refreshing models: {str(e)}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_user_allowed(str(update.effective_user.id)):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
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
        "*Web Search (function calling):*\n"
        "• `/web` — show status\n"
        "• `/web on` / `/web off` — toggle\n"
        "• `/web brave` — Brave Search API\n"
        "• `/web searxng` — SearXNG (self-hosted)\n"
        "• `/web ddg` — DuckDuckGo (free)\n\n"
        "_When web search is ON and your provider supports function calling, "
        "the model decides autonomously whether to search — no extra routing call needed._\n\n"
        "*Model Validation:*\n"
        "• `/validate` — test which models work\n"
        "• `/verified` — show validated models\n"
        "• `/clearvalidation` — clear cache\n\n"
        "*Thinking Mode (NVIDIA only):*\n"
        "• `/thinking on` / `/thinking off`\n\n"
        "*Image OCR:*\n"
        "• Send any photo — text is extracted via vision model\n"
        "• Add a caption to ask a specific question about the image\n"
        "• Requires `OCR_API_KEY` / `GEMINI_API_KEY` to be set\n\n"
        "*Other:*\n"
        "• `/status` — show provider, model, toggles & stats\n"
        "• `/clear` — clear conversation\n"
        "• `/restart` — cancel any stuck/pending AI request\n"
        "• `/help` — this message",
        parse_mode="Markdown",
    )


async def provider_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    session = get_user_session(user_id)
    if not context.args:
        _, current = _resolve_provider(session)
        await update.message.reply_text(
            f"📡 *Current:* {current.get_name()}\n"
            f"🔧 *Available:* {', '.join(provider_manager.list_providers())}\n\n"
            f"Use `/provider <name>` to switch.",
            parse_mode="Markdown",
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
    _cancel_bg_tasks(session)
    session["provider"] = new_name
    session["history"] = []
    session["summary"] = ""
    current_model = session["models"].get(new_name) or new_provider.get_default_model()
    fc = "✅ supports function calling" if new_provider.supports_function_calling() else "⚠️ no function calling"
    await update.message.reply_text(
        f"✅ Switched to *{new_provider.get_name()}*!\nModel: `{current_model}`\n_{fc}_",
        parse_mode="Markdown",
    )


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    session = get_user_session(user_id)
    provider_name, provider = _resolve_provider(session)
    show_all = len(context.args) > 0 and context.args[0].lower() == "all"
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
                parse_mode="Markdown",
            )
            return
        await update.message.reply_text("✅ Showing verified models...")
        all_models = await asyncio.to_thread(provider.get_available_models)
        models = [m for m in all_models if m["id"] in validated_ids]
        if not models:
            await update.message.reply_text(
                f"⚠️ *Validated models are no longer in {provider.get_name()}'s model list.*\n\n"
                f"Run `/clearvalidation` then `/validate` to refresh.",
                parse_mode="Markdown",
            )
            return
        title_suffix = " (Verified)"
        footer_note = f"\n\n💡 Use `/models all` to see all {len(all_models)} models"
    current_model = session["models"].get(provider_name) or provider.get_default_model()
    chunks = [models[i : i + 20] for i in range(0, len(models), 20)]
    for idx, chunk in enumerate(chunks, 1):
        model_list = "\n".join(
            [
                f"• `{m['id']}`" + (" ✓" if m["id"] == current_model else "")
                for m in chunk
            ]
        )
        part = f" (Part {idx}/{len(chunks)})" if len(chunks) > 1 else ""
        await update.message.reply_text(
            f"🤖 *{provider.get_name()}{title_suffix}{part}:*\n\n{model_list}"
            + (
                f"\n\nCurrent: `{current_model}`{footer_note}\n\nUse `/model <id>` to switch."
                if idx == len(chunks)
                else ""
            ),
            parse_mode="Markdown",
        )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    session = get_user_session(user_id)
    provider_name, provider = _resolve_provider(session)
    if not context.args:
        current = session["models"].get(provider_name) or provider.get_default_model()
        await update.message.reply_text(
            f"🤖 *Current Model ({provider.get_name()}):* `{current}`\n\nUse `/models` to see options.",
            parse_mode="Markdown",
        )
        return
    new_model = " ".join(context.args)
    try:
        model_ids = [
            m["id"] for m in await asyncio.to_thread(provider.get_available_models)
        ]
    except Exception as e:
        logger.error(f"[Model] Could not fetch model list for {provider_name}: {e}")
        await update.message.reply_text(
            "❌ Could not verify the model (model list fetch failed).\n"
            "No change made — try again later."
        )
        return
    session["models"][provider_name] = new_model
    if new_model in model_ids:
        await update.message.reply_text(
            f"✅ Switched to model: `{new_model}`\n💾 Saved for {provider.get_name()}.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            f"⚠️ Model `{new_model}` not in known list — set anyway.\n"
            f"Use `/models` to see known models.",
            parse_mode="Markdown",
        )


async def validate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    session = get_user_session(user_id)
    provider_name, provider = _resolve_provider(session)
    models = await asyncio.to_thread(provider.get_available_models)
    already_validated = get_validated_models(provider_name)
    permanently_failed = get_failed_models(provider_name)
    models_to_test = [
        m
        for m in models
        if m["id"] not in already_validated and m["id"] not in permanently_failed
    ]
    skipped = len(already_validated) + len(permanently_failed)
    if skipped > 0:
        await update.message.reply_text(
            f"💡 *Smart Validation*\n\n"
            f"Skipping {skipped}: ✅ {len(already_validated)} working, ❌ {len(permanently_failed)} failed\n"
            f"To test: {len(models_to_test)}/{len(models)}\n"
            f"⏳ ~{len(models_to_test) * 2}s\n\n"
            f"Use `/clearvalidation` to re-test all",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            f"🔍 *Full Validation* — testing all {len(models)} models\n⏳ ~{len(models) * 2}s",
            parse_mode="Markdown",
        )
    if not models_to_test:
        await update.message.reply_text(
            "✅ All models already validated!\n\nUse `/verified` to see them."
        )
        return
    cancel_event = asyncio.Event()
    session["cancel_event"] = cancel_event  # lets /restart abort the validation loop

    validated = list(already_validated)
    newly_validated, failed_na, failed_rl, failed_unk = [], [], [], []
    for idx, model_info in enumerate(models_to_test, 1):
        if cancel_event.is_set():
            break
        model_id = model_info["id"]
        if idx % 5 == 0 or idx == 1:
            await update.message.reply_text(
                f"⏳ {idx}/{len(models_to_test)}: `{model_id}`...",
                parse_mode="Markdown",
            )
        success, error_type = await asyncio.to_thread(provider.test_model, model_id)
        if success:
            validated.append(model_id)
            newly_validated.append(model_id)
            add_validated_model(provider_name, model_id)
        elif error_type == "not_available":
            failed_na.append(model_id)
            add_failed_model(provider_name, model_id)
        elif error_type == "rate_limit":
            failed_rl.append(model_id)
        else:
            failed_unk.append(model_id)
        if idx < len(models_to_test):
            try:
                await asyncio.wait_for(cancel_event.wait(), timeout=2)
                break
            except asyncio.TimeoutError:
                pass
    job_cancelled = cancel_event.is_set()
    total_tested = len(newly_validated) + len(failed_na) + len(failed_rl) + len(failed_unk)
    total_failed = len(failed_na) + len(failed_rl) + len(failed_unk)
    rate = (len(newly_validated) / total_tested * 100) if total_tested else 0
    msg = (
        f"{'⏹️ *Validation Stopped* (via /restart)' if job_cancelled else '✅ *Validation Complete*'}\n\n"
        f"• Tested: {total_tested}/{len(models_to_test)}\n"
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
    await update.message.reply_text(msg, parse_mode="Markdown")


async def verified_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    session = get_user_session(user_id)
    provider_name, provider = _resolve_provider(session)
    validated_ids = get_validated_models(provider_name)
    if not validated_ids:
        await update.message.reply_text(
            f"❌ *No validated models for {provider.get_name()}*\n\nRun `/validate` first!",
            parse_mode="Markdown",
        )
        return
    all_models = await asyncio.to_thread(provider.get_available_models)
    validated_models = [m for m in all_models if m["id"] in validated_ids]
    if not validated_models:
        await update.message.reply_text(
            f"⚠️ *Validated models are no longer in {provider.get_name()}'s model list.*\n\n"
            f"Run `/clearvalidation` then `/validate` to refresh.",
            parse_mode="Markdown",
        )
        return
    current_model = session["models"].get(provider_name) or provider.get_default_model()
    model_list = "\n".join(
        [
            f"• `{m['id']}`" + (" ✓" if m["id"] == current_model else "")
            for m in validated_models
        ]
    )
    await update.message.reply_text(
        f"✅ *Verified Models — {provider.get_name()}:*\n\n{model_list}\n\n"
        f"Current: `{current_model}`\n\nUse `/model <id>` to switch.",
        parse_mode="Markdown",
    )


async def thinking_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    session = get_user_session(user_id)
    provider_name, provider = _resolve_provider(session)

    if not context.args:
        state = "enabled" if session.get("thinking_enabled", False) else "disabled"
        await update.message.reply_text(
            f"💭 *Thinking Mode:* {state}\n\nUse `/thinking on` or `/thinking off`.",
            parse_mode="Markdown",
        )
        return

    arg = context.args[0].lower()
    if arg == "on":
        session["thinking_enabled"] = True
        current_model = (
            session["models"].get(provider_name) or provider.get_default_model()
        )
        supports = provider.supports_thinking(current_model)
        await update.message.reply_text(
            f"✅ *Thinking mode enabled!* 💭\n\n"
            f"Provider: `{provider_name}`\n"
            f"Model: `{current_model}`\n"
            f"Supports thinking: {'✅ Yes' if supports else '⚠️ No'}",
            parse_mode="Markdown",
        )
    elif arg == "off":
        session["thinking_enabled"] = False
        await update.message.reply_text(
            "🔕 *Thinking mode disabled.*", parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            "❌ Use `/thinking on` or `/thinking off`", parse_mode="Markdown"
        )


async def clearvalidation_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
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
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    session = get_user_session(user_id)
    _, provider = _resolve_provider(session)

    if not context.args:
        status = "ON" if session.get("web_search") else "OFF"
        eng = session.get("search_engine", "duckduckgo")
        eng_label = {
            "brave": "Brave API",
            "searxng": "SearXNG",
            "duckduckgo": "DuckDuckGo",
        }.get(eng, eng)
        fc_note = (
            "✅ Function calling active — model triggers search automatically."
            if provider.supports_function_calling()
            else "⚠️ Provider has no function calling — search unavailable for this provider."
        )
        searxng_line = (
            "`/web searxng` — SearXNG (self-hosted)\n" if SEARXNG_URL else ""
        )
        await update.message.reply_text(
            f"🌐 *Web Search:* {status}\n"
            f"🔍 *Engine:* {eng_label}\n"
            f"⚙️ {fc_note}\n\n"
            f"Use: `/web on` | `/web off`\n"
            f"`/web brave` — Brave Search API\n"
            f"{searxng_line}"
            f"`/web ddg` — DuckDuckGo (free)",
            parse_mode="Markdown",
        )
        return
    arg = context.args[0].lower()
    if arg == "on":
        session["web_search"] = True
        eng = session.get("search_engine", "duckduckgo")
        eng_label = {
            "brave": "Brave",
            "searxng": "SearXNG",
            "duckduckgo": "DuckDuckGo",
        }.get(eng, eng)
        await update.message.reply_text(f"✅ Web search enabled ({eng_label}).")
    elif arg == "off":
        session["web_search"] = False
        await update.message.reply_text("🔕 Web search disabled.")
    elif arg == "brave":
        if not BRAVE_API_KEY:
            await update.message.reply_text(
                "❌ Brave API key not configured. Use `/web ddg`.",
                parse_mode="Markdown",
            )
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
            await update.message.reply_text(
                "❌ SEARXNG_URL not configured.", parse_mode="Markdown"
            )
        else:
            session["search_engine"] = "searxng"
            session["web_search"] = True
            await update.message.reply_text("✅ Switched to SearXNG.")
    else:
        await update.message.reply_text(
            "❌ Use: `/web on|off|brave|searxng|ddg`", parse_mode="Markdown"
        )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return

    session = get_user_session(user_id)
    provider_name, provider = _resolve_provider(session)
    current_model = session["models"].get(provider_name) or provider.get_default_model()

    web_on = session.get("web_search", True)
    engine = session.get("search_engine", "duckduckgo")
    engine_label = {
        "brave": "Brave",
        "searxng": "SearXNG",
        "duckduckgo": "DuckDuckGo",
    }.get(engine, engine)

    if web_on and provider.supports_function_calling():
        web_line = f"ON ({engine_label}, function calling ✅)"
    elif web_on:
        web_line = f"ON ({engine_label}, ⚠️ provider lacks function calling)"
    else:
        web_line = "OFF"

    thinking_on = session.get("thinking_enabled", False)
    if thinking_on and provider_name != "nvidia":
        thinking_line = "ON (NVIDIA only — ignored here)"
    else:
        thinking_line = "ON" if thinking_on else "OFF"

    history_len = len(session.get("history", []))
    summary_chars = len((session.get("summary") or "").strip())
    memory_line = f"{summary_chars:,} chars" if summary_chars else "empty"

    await update.message.reply_text(
        f"📊 *Status*\n"
        f"📡 Provider: *{provider.get_name()}*\n"
        f"🧠 Model: `{current_model}`\n"
        f"🌐 Web: {web_line}\n"
        f"💭 Thinking: {thinking_line}\n"
        f"💬 History: {history_len} msgs\n"
        f"📜 Memory: {memory_line}",
        parse_mode="Markdown",
    )


async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return
    session = get_user_session(user_id)
    cancel_event: asyncio.Event = session["cancel_event"]
    cancel_event.set()  # aborts the in-flight request that still holds this event
    _cancel_bg_tasks(session)
    # The event is replaced lazily: the next incoming message installs a fresh
    # one, so pressing /restart again still aborts the same task until it exits.
    history = session["history"]
    if history and history[-1].get("role") == "user":
        history.pop()
    await update.message.reply_text(
        "🛑 *Restart requested.* You can send a new message now.",
        parse_mode="Markdown",
    )


# ============================================================================
# MESSAGE HANDLER — function-calling search
#
# FLOW:
#
#   Web ON  + provider supports tools  →  _chat_with_function_calling()
#     • Call 1: model gets WEB_SEARCH_TOOL definition
#       - Answers directly            → 1 total LLM call, zero search overhead
#       - Emits tool_calls            → execute search, Call 2 with results
#                                        → final answer using live data
#
#   Web OFF or provider has no tools  →  single direct AI call (no search)
#
# WHY BETTER THAN THE OLD TWO-CALL ROUTER:
#   - No dedicated SEARCH/NOSEARCH call; the decision happens inside Call 1
#   - No text parsing / regex; query arrives as structured JSON
#   - Model crafts the query in context, not in an isolated router prompt
#   - Greetings & trivial messages cost exactly 1 call (model doesn't fire tool)
#   - Same or fewer total LLM calls in every scenario
# ============================================================================


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
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
    # Any prior abort signal is consumed here when we install a fresh event;
    # an already-aborted AI call exits on its own, so the install is safe.
    cancel_event = asyncio.Event()
    session["cancel_event"] = cancel_event
    assistant_appended = False

    try:
        if cancel_event.is_set():
            return

        try:
            await update.message.chat.send_action(action="typing")
        except Exception:
            pass

        provider_name, provider = _resolve_provider(session)
        current_model = session["models"].get(provider_name)
        thinking_enabled = session.get("thinking_enabled", False)
        web_on = session.get("web_search", True)
        engine = session.get("search_engine", "duckduckgo")

        # Append user message (popped on error below)
        session["history"].append({"role": "user", "content": user_message})

        # Build the messages list for the LLM — system prompt always first
        chat_msgs = [
            {"role": "system", "content": _build_system_prompt(session)}
        ] + session["history"]

        if cancel_event.is_set():
            raise asyncio.CancelledError("Cancelled by /restart")

        try:
            await update.message.chat.send_action(action="typing")
        except Exception:
            pass

        bot_response = await _chat_with_function_calling(
            provider=provider,
            messages=chat_msgs,
            model=current_model,
            enable_thinking=thinking_enabled,
            web_on=web_on,
            engine=engine,
            cancel_event=cancel_event,
        )

        # ── /restart race guard ────────────────────────────────────────────
        if cancel_event.is_set():
            raise asyncio.CancelledError("Cancelled by /restart")

        if not bot_response.strip():
            bot_response = "⚠️ The AI returned an empty response. Try again or use `/model` to switch models."

        # ── Update history ─────────────────────────────────────────────────
        session["history"].append({"role": "assistant", "content": bot_response})
        assistant_appended = True

        # ── Send response (handle 4096 char limit) ─────────────────────────
        if len(bot_response) <= MAX_MESSAGE_LENGTH:
            await reply_text_safe(update.message, bot_response)
        else:
            HEADER_RESERVE = 25
            chunk_limit = MAX_MESSAGE_LENGTH - HEADER_RESERVE
            chunks, current_chunk = [], ""
            for line in bot_response.split("\n"):
                if len(line) > chunk_limit:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    for j in range(0, len(line), chunk_limit):
                        chunks.append(line[j : j + chunk_limit])
                    continue
                if len(current_chunk) + len(line) + 1 > chunk_limit:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = line
                else:
                    current_chunk = (
                        (current_chunk + "\n" + line) if current_chunk else line
                    )
            if current_chunk:
                chunks.append(current_chunk)
            for i, chunk in enumerate(chunks, 1):
                header = f"📄 Part {i}/{len(chunks)}\n\n" if len(chunks) > 1 else ""
                await reply_text_safe(update.message, header + chunk)

        _schedule_compact(session, provider, current_model)

    except asyncio.CancelledError:
        logger.info(f"[Bot] user={user_id} request cancelled by /restart")
        if (
            assistant_appended
            and session["history"]
            and session["history"][-1].get("role") == "assistant"
        ):
            session["history"].pop()
        if session["history"] and session["history"][-1].get("role") == "user":
            session["history"].pop()
    except Exception as e:
        logger.error(f"Error in handle_message: {e}", exc_info=True)
        if (
            assistant_appended
            and session["history"]
            and session["history"][-1].get("role") == "assistant"
        ):
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
# IMAGE OCR
# ============================================================================

_NVIDIA_VISION_URL = f"{VISION_BASE_URL}/chat/completions"
_MEDIA_GROUP_WAIT = 1.5
_OCR_MAX_RETRIES = 2
_OCR_RETRY_BASE_DELAY = 3.0
_DEFAULT_OCR_PROMPT = (
    "Extract and transcribe ALL text visible in this image exactly as written. "
    "If there is no text, describe the image content concisely."
)

_media_group_buffer: Dict[str, dict] = {}
_media_group_tasks: Dict[str, "asyncio.Task[None]"] = {}


def _nvidia_vision_sync(b64_data: str, prompt: str) -> str:
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
    response = requests.post(
        _NVIDIA_VISION_URL, headers=headers, json=payload, timeout=60
    )
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
    buf: Optional[io.BytesIO] = None
    b64_str: Optional[str] = None
    try:
        tg_file = await context.bot.get_file(photo_obj.file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(buf)
        # photo.file_size can be reported as None by some clients — enforce the
        # limit on the actual downloaded bytes too.
        actual_size = buf.tell()
        if actual_size > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image too large after download "
                f"({actual_size / 1024 / 1024:.1f} MB, "
                f"max {MAX_IMAGE_BYTES / 1024 / 1024:.0f} MB)."
            )
        logger.info(
            f"[OCR] user={user_id} size={buf.tell() / 1024:.1f} KB model={NVIDIA_VISION_MODEL}"
        )

        b64_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        del buf
        buf = None

        last_error: Optional[Exception] = None
        for attempt in range(1, _OCR_MAX_RETRIES + 2):
            try:
                result = await asyncio.to_thread(_nvidia_vision_sync, b64_str, prompt)
                if result.strip():
                    return result.strip()
                logger.warning(
                    f"[OCR] attempt {attempt}: empty response from model, retrying..."
                )
            except Exception as e:
                last_error = e
                error_lower = str(e).lower()
                if not any(kw in error_lower for kw in _TRANSIENT_ERROR_KEYWORDS):
                    raise
                logger.warning(f"[OCR] attempt {attempt} transient error: {e}")

            if attempt <= _OCR_MAX_RETRIES:
                delay = _OCR_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"[OCR] retrying in {delay:.0f}s...")
                await asyncio.sleep(delay)

        if last_error:
            raise last_error
        return (
            "⚠️ The model returned an empty response after retries. Try a clearer image."
        )
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
    if len(text) <= MAX_MESSAGE_LENGTH:
        await reply_text_safe(message, text)
        return
    chunk_limit = MAX_MESSAGE_LENGTH - 25
    chunks, current_chunk = [], ""
    for line in text.split("\n"):
        if len(line) > chunk_limit:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            for j in range(0, len(line), chunk_limit):
                chunks.append(line[j : j + chunk_limit])
            continue
        if len(current_chunk) + len(line) + 1 > chunk_limit:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk = (current_chunk + "\n" + line) if current_chunk else line
    if current_chunk:
        chunks.append(current_chunk)
    for i, chunk in enumerate(chunks, 1):
        header = f"📄 Part {i}/{len(chunks)}\n\n" if len(chunks) > 1 else ""
        await reply_text_safe(message, header + chunk)


async def _flush_media_group(group_id: str, context):
    await asyncio.sleep(_MEDIA_GROUP_WAIT)

    entry = _media_group_buffer.pop(group_id, None)
    _media_group_tasks.pop(group_id, None)
    if not entry:
        return

    photos = entry["photos"]
    user_id = entry["user_id"]
    total = len(photos)

    logger.info(
        f"[OCR] album {group_id}: processing {total} photo(s) for user={user_id}"
    )

    try:
        await _process_media_group(entry, group_id, context)
    except Exception:
        logger.error(f"[OCR] album {group_id} flush failed", exc_info=True)
    finally:
        # Guarantee cleanup even if an exception or task cancellation hits mid-way.
        _media_group_buffer.pop(group_id, None)
        _media_group_tasks.pop(group_id, None)


async def _process_media_group(entry: dict, group_id: str, context) -> None:
    photos = entry["photos"]
    message = entry["message"]
    prompt = entry["prompt"]
    user_id = entry["user_id"]
    session = entry["session"]
    total = len(photos)

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

    if total == 1:
        combined = results[0]
    else:
        sections = [f"*Image {i}/{total}*\n{r}" for i, r in enumerate(results, 1)]
        combined = "\n\n---\n\n".join(sections)

    session["history"].append(
        {"role": "user", "content": f"[{total} image(s)] {prompt}"}
    )
    session["history"].append({"role": "assistant", "content": combined})

    try:
        await _send_ocr_reply(message, combined)
    except Exception as e:
        logger.error(f"[OCR] Failed to send album reply: {e}", exc_info=True)

    try:
        provider_name, provider = _resolve_provider(session)
        current_model = session["models"].get(provider_name)
        _schedule_compact(session, provider, current_model)
    except Exception as e:
        logger.warning(f"[OCR] post-album compact skipped: {e}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not is_user_allowed(user_id):
        await update.message.reply_text(
            "⛔ Sorry, you're not authorized to use this bot."
        )
        return

    if not OCR_API_KEY:
        await update.message.reply_text(
            "❌ Image OCR requires `GEMINI_API_KEY`, `NVIDIA_API_KEY`, or `OCR_API_KEY` to be configured.",
            parse_mode="Markdown",
        )
        return

    photo = update.message.photo[-1]
    prompt = (
        update.message.caption.strip()
        if update.message.caption
        else _DEFAULT_OCR_PROMPT
    )

    if photo.file_size and photo.file_size > MAX_IMAGE_BYTES:
        await update.message.reply_text(
            f"❌ Image too large ({photo.file_size / 1024 / 1024:.1f} MB). "
            f"Max allowed: {MAX_IMAGE_BYTES / 1024 / 1024:.0f} MB."
        )
        return

    session = get_user_session(user_id)
    group_id = update.message.media_group_id

    if group_id is not None:
        if group_id not in _media_group_buffer:
            _media_group_buffer[group_id] = {
                "photos": [],
                "message": update.message,
                "prompt": prompt,
                "user_id": user_id,
                "session": session,
            }
            _media_group_tasks[group_id] = asyncio.create_task(
                _flush_media_group(group_id, context)
            )
        if update.message.caption:
            _media_group_buffer[group_id]["prompt"] = prompt
        _media_group_buffer[group_id]["photos"].append(photo)
        logger.info(
            f"[OCR] buffered photo {len(_media_group_buffer[group_id]['photos'])} "
            f"for album {group_id} user={user_id}"
        )
        return

    try:
        await update.message.chat.send_action(action="typing")
    except Exception:
        pass

    try:
        result = await _ocr_one_photo(context, photo, prompt, user_id)

        session["history"].append({"role": "user", "content": f"[image] {prompt}"})
        session["history"].append({"role": "assistant", "content": result})

        await _send_ocr_reply(update.message, result)

        try:
            provider_name, provider = _resolve_provider(session)
            current_model = session["models"].get(provider_name)
            _schedule_compact(session, provider, current_model)
        except Exception as e:
            logger.warning(f"[OCR] post-photo compact skipped: {e}")

    except Exception as e:
        logger.error(f"[OCR] Error for user={user_id}: {e}", exc_info=True)
        try:
            await update.message.reply_text(
                f"❌ Failed to process image: {str(e)}\n\n"
                f"Make sure `OCR_API_KEY` (or `GEMINI_API_KEY`) is valid and model `{NVIDIA_VISION_MODEL}` is accessible."
            )
        except Exception:
            pass


# ============================================================================
# MAIN
# ============================================================================


def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is required!")
    if not (
        GROQ_API_KEY
        or GEMINI_API_KEY
        or OPENROUTER_API_KEY
        or CEREBRAS_API_KEY
        or NVIDIA_API_KEY
        or VERCEL_API_KEY
        or (CUSTOM_API_KEY and CUSTOM_BASE_URL and CUSTOM_DEFAULT_MODEL)
    ):
        raise ValueError("At least one AI provider API key is required!")

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(CommandHandler("refresh", refresh_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("provider", provider_command))
    application.add_handler(CommandHandler("models", models_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("validate", validate_command))
    application.add_handler(CommandHandler("verified", verified_command))
    application.add_handler(CommandHandler("clearvalidation", clearvalidation_command))
    application.add_handler(CommandHandler("thinking", thinking_command))
    application.add_handler(CommandHandler("web", web_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("restart", restart_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("🚀 Multi-Provider AI Bot started!")
    logger.info(f"📡 Providers: {', '.join(provider_manager.list_providers())}")

    # Log function calling support per provider
    for name, prov in provider_manager.providers.items():
        fc = "✅ function calling" if prov.supports_function_calling() else "⚠️  no function calling"
        logger.info(f"   {name}: {fc}")

    if OCR_API_KEY:
        logger.info(
            f"🖼️  Image OCR enabled — model: {NVIDIA_VISION_MODEL} endpoint: {VISION_BASE_URL}"
        )
    else:
        logger.info(
            "🖼️  Image OCR disabled — set GEMINI_API_KEY, NVIDIA_API_KEY, or OCR_API_KEY to enable"
        )
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
