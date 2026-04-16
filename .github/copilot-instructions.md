# Copilot Instructions — Multi-Provider AI Telegram Bot

## Build and Run

### Python (Direct)
```bash
pip install -r requirements.txt
python bot.py
```

### Docker
```bash
docker build -t multi-ai-bot .
docker run -d --env-file .env multi-ai-bot
```

**No test suite or linting is configured for this project.**

---

## Architecture Overview

This is a **multi-provider Telegram bot** that routes chat requests to different AI APIs (Groq, Gemini, OpenRouter, Cerebras, NVIDIA). All providers implement the abstract `AIProvider` base class with a consistent interface:

### Provider Abstraction Layer

**Base class:** `AIProvider` (line 138 in `bot.py`)

All provider implementations must:
- Implement `chat(messages, model, enable_thinking)` → returns AI response string
- Implement `get_available_models()` → returns list of `{"id": str, "name": str}` dicts
- Implement `get_name()` → returns provider display name
- Implement `get_default_model()` → returns default model ID
- Optionally implement `supports_thinking(model_id)` → for reasoning trace support
- Optionally implement `test_model(model_id)` → validates if model works

**Existing providers:**
- `GroqProvider` (line 199)
- `GeminiProvider` (line 287)
- `OpenRouterProvider` (line 402)
- `CerebrasProvider` (line 539)
- `NvidiaProvider` (line 628)

### Model Ranking System

Models are **future-proofed** using `get_model_capability_score()` (line 78), which ranks models by:
1. **Tier** (flagship models → experimental → unknown)
2. **Parameter size** (extracted from model ID: e.g., "70b" → 70)
3. **Model ID** (lexicographic tiebreaker)

This ensures new models are automatically ranked correctly without hardcoded lists.

### Two-Pass Web Search

When web search is enabled (`/web on`):
1. **Pass 1:** AI receives user message + search prompt, decides if it needs search, responds with `SEARCH: <query>` or answers directly
2. **Search:** Bot fetches results from Brave API or DuckDuckGo
3. **Pass 2:** AI receives original message + search results, generates final answer

Search results are **not** added to conversation history to avoid context pollution.

**Search detection logic:** `get_search_prompt()` (line 36) injects date + search trigger keywords.

### Session Management

`user_sessions` dict stores per-user state:
```python
{
    "history": [],           # List[Dict] with role/content
    "provider": str,         # Current provider name
    "model": str,           # Current model ID
    "web_search": bool,     # Search enabled/disabled
    "search_engine": str,   # "brave" or "duckduckgo"
    "thinking": bool        # Reasoning mode (NVIDIA only)
}
```

**History limit:** `MAX_HISTORY_MESSAGES` (default: 20) — oldest messages are pruned.

**Access control:** `ALLOWED_USER_IDS` (comma-separated user IDs) — empty = public bot.

---

## Thinking Tag Filtering

### Universal Tag Stripping: `strip_thinking_tags()`

Located at **line 550**, this utility function handles inline thinking tags across all providers:

```python
def strip_thinking_tags(text: str, keep_thinking: bool = False) -> str
```

**Supported Tags:**
- `<think>...</think>`
- `<thinking>...</thinking>`
- `<reasoning>...</reasoning>`
- `<context>...</context>`

**Behavior:**
- `keep_thinking=False` → Removes all thinking content (default)
- `keep_thinking=True` → Formats thinking with emoji headers

**Integration:** Every provider's `chat()` method calls this before returning:
```python
content = response.choices[0].message.content
return strip_thinking_tags(content, keep_thinking=enable_thinking)
```

**NVIDIA Exception:** NVIDIA uses native streaming with `reasoning_content` field, bypasses tag filtering.

---

## Key Conventions

### Message Splitting

Telegram enforces a **4096 character limit** per message. Long responses are automatically chunked at newline boundaries (line 1800-1834). Each chunk gets a header like `📄 Part 1/3`.

### Model Validation Cache

`validated_models.json` stores tested models to avoid redundant API calls. Schema:
```json
{
    "groq": {
        "model-id": {"works": true, "error": "success", "timestamp": 1234567890}
    }
}
```

Commands:
- `/validate` — tests all models with real API calls
- `/clearvalidation` — deletes cache, forces re-validation

### System Prompt Injection

All providers inject `SYSTEM_PROMPT` automatically if not present in messages. Search-enabled requests use a **dynamic system prompt** (`SYSTEM_PROMPT + get_search_prompt()`).

### Thinking Mode (All Providers)

Models with reasoning support can expose thinking traces when `/thinking on` is enabled. The bot uses `strip_thinking_tags()` (line 550) to filter inline thinking tags like `<think>`, `<thinking>`, `<reasoning>`, and `<context>`.

**Provider-Specific Behavior:**
- **NVIDIA:** Native streaming API with separate `reasoning_content` field
- **All Others:** Inline XML-style tags filtered via regex

**Tag Filtering:** When `/thinking off` (default), all thinking content is removed. When `/thinking on`, thinking is formatted with `💭 *Thinking:*` headers.

**Model Detection:** Each provider implements `supports_thinking(model_id)` to detect reasoning-capable models by keywords like "reasoning", "think", "deepseek", "qwq", "r1".

---

## ESP32 Version

`esp32_bot.py` is a **MicroPython port** for ESP32-C3 boards with severe RAM constraints:
- **Zero disk writes** — all state is in-memory
- **Client-side search heuristics** — pre-filters messages to save API calls (line 91-140 in `esp32_bot.py`)
- **Aggressive limits:** `MAX_HISTORY=5`, `MAX_SESSIONS=3`, `MAX_RESPONSE_LEN=4000`
- **Watchdog timer** — auto-reboot on hangs

Flashed as `main.py` (auto-runs on boot). See `ESP32-GUIDE.md` for setup instructions.

---

## Environment Variables

### Required
- `TELEGRAM_TOKEN` — bot token from @BotFather

### Providers (at least one required)
- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- `CEREBRAS_API_KEY`
- `NVIDIA_API_KEY`

### Behavior
- `DEFAULT_PROVIDER` — `groq` | `gemini` | `openrouter` | `cerebras` | `nvidia`
- `MAX_TOKENS` — response length (default: 512)
- `TEMPERATURE` — 0.0-1.0 (default: 0.7)
- `MAX_HISTORY_MESSAGES` — context window (default: 20)
- `SYSTEM_PROMPT` — AI personality/instructions

### Web Search
- `BRAVE_API_KEY` — optional, falls back to DuckDuckGo if unset
- `SEARCH_ENGINE` — `brave` | `duckduckgo` (default: `brave`)
- `MAX_SEARCH_RESULTS` — results per query (default: 3)
- `MAX_SNIPPET_LEN` — chars per snippet (default: 300)

### Access Control
- `ALLOWED_USER_IDS` — comma-separated Telegram user IDs (empty = public)

---

## Adding a New Provider

1. **Create subclass** of `AIProvider`:
   ```python
   class NewProvider(AIProvider):
       def __init__(self, api_key: str):
           self.api_key = api_key
           self.default_model = "model-id"
       
       def chat(self, messages, model=None, enable_thinking=False, max_tokens=None):
           # Implement API call logic
           response = self.client.chat.completions.create(...)
           content = response.choices[0].message.content
           if not content:
               raise ValueError("API returned empty response.")
           # IMPORTANT: Filter thinking tags
           return strip_thinking_tags(content, keep_thinking=enable_thinking)
       
       def get_available_models(self):
           # Fetch from API or return hardcoded list
           pass
       
       def get_name(self):
           return "NewProvider"
       
       def get_default_model(self):
           return self.default_model
       
       def supports_thinking(self, model_id: str) -> bool:
           # Detect reasoning-capable models
           reasoning_keywords = ["reasoning", "think", "deepseek", "qwq", "r1"]
           return any(keyword in model_id.lower() for keyword in reasoning_keywords)
   ```

2. **Add to `ProviderManager`** (around line 800-900):
   - Add API key env var: `NEW_PROVIDER_API_KEY`
   - Register in `ProviderManager.__init__()` with conditional creation
   - Update initialization logic to include the new provider

3. **Update README.md** with provider details and free tier limits.

4. **Thinking Support:** Implement `supports_thinking()` for reasoning models:
   ```python
   def supports_thinking(self, model_id: str) -> bool:
       reasoning_keywords = ["reasoning", "think", "deepseek", "qwq", "r1"]
       return any(keyword in model_id.lower() for keyword in reasoning_keywords)
   ```
   
   **Note:** `strip_thinking_tags()` is called automatically in `chat()`, so no additional logic needed unless using custom streaming (like NVIDIA).

---

## Common Pitfalls

### Gemini Message Format
Gemini uses `chat.start_chat(history=...)` with a **separate** `system_instruction` parameter. Do not include system messages in history (line 322-327).

### Empty API Responses
Some providers return `choices[0].message.content = None` when overloaded or rate-limited. Always check for `None` and raise a clear error (line 228).

### Model ID Variations
Model IDs are inconsistent across providers:
- Groq: `llama-3.3-70b-versatile`
- NVIDIA: `openai/gpt-oss-120b`
- OpenRouter: `meta-llama/llama-3.3-70b-instruct`

Use the exact ID returned by the provider's API, not a simplified name.

---

## Project Structure

```
chat-telegram/
├── bot.py                   # Main bot (1862 lines)
│   ├── Model ranking (66-133)
│   ├── Provider abstractions (138-750)
│   ├── Search engines (750-850)
│   ├── ProviderManager (850-950)
│   ├── Model validation (950-1100)
│   ├── Command handlers (1100-1750)
│   └── Message handler (1750-1850)
├── esp32_bot.py             # MicroPython ESP32 version
├── clear_old_commands.py    # Utility to clean bot commands
├── requirements.txt         # Python deps
├── Dockerfile               # Docker build
└── .env.example             # Config template
```

---

## Deployment

**GHCR:** `ghcr.io/foxy1402/chat-telegram:latest` (automatically built on push to `main` via GitHub Actions — see `.github/workflows/docker-publish.yml`)

**Local Docker:** `docker build -t multi-ai-bot . && docker run -d --env-file .env multi-ai-bot`

**Cloud platforms:** Point to GHCR image + set env vars
