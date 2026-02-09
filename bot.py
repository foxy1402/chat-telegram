import os
import logging
import json
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from datetime import datetime
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
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '512'))  # Configurable token limit (lower = more concise)
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))  # Configurable temperature (0.0-1.0)
MAX_HISTORY_MESSAGES = int(os.getenv('MAX_HISTORY_MESSAGES', '20'))  # Configurable history length
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'You are a helpful AI assistant. Be concise and straight to the point. Avoid unnecessary explanations unless specifically asked.')

# Provider API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
CEREBRAS_API_KEY = os.getenv('CEREBRAS_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

# Model validation cache file
VALIDATED_MODELS_CACHE = os.path.join(os.path.dirname(__file__), 'validated_models.json')

# ============================================================================
# FUTURE-PROOF MODEL RANKING UTILITIES
# ============================================================================

import re

def extract_parameter_size(model_id: str) -> int:
    """
    Extract parameter size from model name (e.g., '70b' -> 70, '405b' -> 405)
    Returns 0 if no size found.
    """
    # Match patterns like: 70b, 405b, 8b, 1.5b, etc.
    match = re.search(r'(\d+\.?\d*)b', model_id.lower())
    if match:
        size = float(match.group(1))
        return int(size) if size >= 1 else 0
    return 0

def get_model_capability_score(model_id: str, model_name: str = "") -> tuple:
    """
    Calculate capability score for future-proof ranking.
    Returns (tier, -param_size, model_id) for sorting.
    Lower tier = more capable. Negative param_size so larger models rank first.
    """
    model_lower = model_id.lower()
    name_lower = model_name.lower()
    param_size = extract_parameter_size(model_id)
    
    # Tier 0: Flagship models (known high-performers)
    flagship_patterns = [
        'gpt-4', 'claude-3-opus', 'claude-3.5', 'claude-4',
        'gemini-2.0', 'gemini-pro', 'llama-3.3', 'llama-3.2',
        'qwen-2.5-72b', 'hermes-3-llama-3.1-405b', 'gpt-oss'
    ]
    if any(pattern in model_lower for pattern in flagship_patterns):
        return (0, -param_size if param_size > 0 else 0, model_id)  # Negative for reverse sort (larger first)
    
    # Tier 1: Very large models (100B+)
    if param_size >= 100:
        return (1, -param_size, model_id)  # 120B ranks before 100B
    
    # Tier 2: Large models (50B-99B)
    if param_size >= 50:
        return (2, -param_size, model_id)  # 70B ranks before 50B
    
    # Tier 3: Medium-large models (20B-49B)
    if param_size >= 20:
        return (3, -param_size, model_id)
    
    # Tier 4: Medium models (10B-19B)
    if param_size >= 10:
        return (4, -param_size, model_id)
    
    # Tier 5: Small-medium models (5B-9B)
    if param_size >= 5:
        return (5, -param_size, model_id)
    
    # Tier 6: Small models (1B-4B)
    if param_size >= 1:
        return (6, -param_size, model_id)
    
    # Tier 7: Experimental/latest (if no size but has 'exp' or version 2.0+)
    if any(x in model_lower for x in ['exp', 'experimental', 'preview', 'beta']):
        # Check for version numbers
        version_match = re.search(r'(\d+\.\d+)', model_id)
        if version_match:
            version = float(version_match.group(1))
            if version >= 2.0:
                return (0, 0, model_id)  # Treat as flagship
        return (7, 0, model_id)
    
    # Tier 8: Unknown models (no size info, no special markers)
    return (8, 0, model_id)

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
        """Get list of available models (dynamically from API when possible)"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get provider name"""
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get default model name"""
        pass
    
    def test_model(self, model_id: str) -> Tuple[bool, str]:
        """Test if a model actually works by sending a minimal request.
        Returns (success, error_type) where error_type is:
        - 'success' if model works
        - 'not_available' if model not in free tier / doesn't exist
        - 'rate_limit' if rate limited
        - 'unknown' for other errors
        """
        try:
            # Send minimal test message
            test_messages = [{"role": "user", "content": "Hi"}]
            response = self.chat(test_messages, model=model_id)
            # If we get any response without error, model works
            if response is not None and len(response) > 0:
                return (True, 'success')
            return (False, 'unknown')
        except Exception as e:
            error_str = str(e).lower()
            logger.debug(f"Model {model_id} validation failed: {e}")
            
            # Detect error type from error message
            if any(keyword in error_str for keyword in ['rate limit', 'too many requests', '429', 'quota']):
                return (False, 'rate_limit')
            elif any(keyword in error_str for keyword in ['not found', '404', 'does not exist', 'invalid model', 'not available', 'not supported', 'no access']):
                return (False, 'not_available')
            else:
                return (False, 'unknown')


class GroqProvider(AIProvider):
    """Groq AI Provider with Dynamic Model Detection"""
    
    def __init__(self, api_key: str):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        # Smart default: Use the smartest free model
        self.default_model = 'llama-3.3-70b-versatile'
        self._cached_models = None
    
    def chat(self, messages: List[Dict], model: Optional[str] = None) -> str:
        model = model or self.default_model
        # Inject system prompt if not already present
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        
        response = self.client.chat.completions.create(
            messages=chat_messages,
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Dynamically fetch available models from Groq API"""
        # Use cached models if available (refresh every bot restart)
        if self._cached_models:
            return self._cached_models
        
        try:
            # Fetch models from Groq API
            models_response = self.client.models.list()
            
            # Filter for chat models and sort by quality
            chat_models = []
            for model in models_response.data:
                # Only include active chat models
                if model.active and hasattr(model, 'id'):
                    model_info = {
                        "id": model.id,
                        "name": self._format_model_name(model.id)
                    }
                    chat_models.append(model_info)
            
            # Future-proof ranking: Use capability scoring
            chat_models.sort(key=lambda m: get_model_capability_score(m['id'], m['name']))
            self._cached_models = chat_models
            
            logger.info(f"✅ Groq: Detected {len(chat_models)} available models")
            return chat_models
            
        except Exception as e:
            logger.warning(f"⚠️ Groq: Could not fetch models dynamically: {e}")
            # Fallback to known models
            return self._get_fallback_models()
    
    def _format_model_name(self, model_id: str) -> str:
        """Format model ID into readable name"""
        # Convert model IDs like 'llama-3.3-70b-versatile' to 'Llama 3.3 70B Versatile'
        name = model_id.replace('-', ' ').title()
        return name
    
    def _get_fallback_models(self) -> List[Dict[str, str]]:
        """Fallback models if API detection fails (ranked smartest to least capable)"""
        return [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile (Smartest)"},
            {"id": "llama-3.1-70b-versatile", "name": "Llama 3.1 70B Versatile"},
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B 32K"},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant"},
            {"id": "gemma2-9b-it", "name": "Gemma 2 9B IT"},
        ]
    
    def get_name(self) -> str:
        return "Groq"
    
    def get_default_model(self) -> str:
        return self.default_model


class GeminiProvider(AIProvider):
    """Google Gemini Provider with Dynamic Model Detection"""
    
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        # Smart default: Use stable model (experimental models may not be available)
        self.default_model = "gemini-1.5-flash"
        self._cached_models = None
    
    def chat(self, messages: List[Dict], model: Optional[str] = None) -> str:
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        model_name = model or self.default_model
        # Configure model with system prompt and generation settings
        generation_config = {
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_TOKENS,
        }
        model = self.genai.GenerativeModel(
            model_name,
            generation_config=generation_config,
            system_instruction=SYSTEM_PROMPT
        )
        
        # Convert messages to Gemini format (skip system messages as they're in system_instruction)
        chat_history = []
        for msg in messages[:-1]:  # All except last
            if msg["role"] == "system":
                continue  # Skip system messages, already in system_instruction
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [msg["content"]]})
        
        # Start chat with history
        chat = model.start_chat(history=chat_history)
        
        # Send last message
        response = chat.send_message(messages[-1]["content"])
        return response.text
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Dynamically fetch available models from Gemini API"""
        if self._cached_models:
            return self._cached_models
        
        try:
            # Fetch all models from Gemini API
            models_response = self.genai.list_models()
            
            # Filter for generateContent-capable models
            chat_models = []
            for model in models_response:
                # Only include models that support generateContent
                if 'generateContent' in model.supported_generation_methods:
                    model_id = model.name.replace('models/', '')
                    
                    # Filter out non-chat models and paid-only models
                    if any(x in model_id.lower() for x in ['vision', 'embedding', 'aqa']):
                        continue
                    
                    # Only include free-tier models (all current Gemini models are free)
                    model_info = {
                        "id": model_id,
                        "name": self._format_model_name(model_id, model)
                    }
                    chat_models.append(model_info)
            
            # Future-proof ranking: Use capability scoring
            chat_models.sort(key=lambda m: get_model_capability_score(m['id'], m['name']))
            self._cached_models = chat_models
            
            logger.info(f"✅ Gemini: Detected {len(chat_models)} available models")
            return chat_models
            
        except Exception as e:
            logger.warning(f"⚠️ Gemini: Could not fetch models dynamically: {e}")
            return self._get_fallback_models()
    
    def _format_model_name(self, model_id: str, model_obj=None) -> str:
        """Format model ID into readable name"""
        # Use display name if available
        if model_obj and hasattr(model_obj, 'display_name'):
            return model_obj.display_name
        
        # Otherwise format the ID
        name = model_id.replace('gemini-', 'Gemini ').replace('-', ' ').title()
        if 'exp' in model_id.lower():
            name += " (Experimental)"
        return name
    
    def _get_fallback_models(self) -> List[Dict[str, str]]:
        """Fallback models if API detection fails (ranked by stability and capability)"""
        return [
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash (Default - Stable)"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro (Most Capable - Stable)"},
            {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash Experimental"},
            {"id": "gemini-1.5-flash-8b", "name": "Gemini 1.5 Flash 8B (Fastest)"},
        ]
    
    def get_name(self) -> str:
        return "Gemini"
    
    def get_default_model(self) -> str:
        return self.default_model


class OpenRouterProvider(AIProvider):
    """OpenRouter Provider with Dynamic Model Detection"""
    
    def __init__(self, api_key: str):
        from openai import OpenAI
        import requests
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.api_key = api_key
        self.requests = requests
        # Smart default: Use the smartest free model (no auto-select)
        self.default_model = "meta-llama/llama-3.3-70b-instruct:free"
        self._cached_models = None
    
    def chat(self, messages: List[Dict], model: Optional[str] = None) -> str:
        model = model or self.default_model
        # Inject system prompt if not already present
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        
        response = self.client.chat.completions.create(
            model=model,
            messages=chat_messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Dynamically fetch FREE models from OpenRouter API"""
        if self._cached_models:
            return self._cached_models
        
        try:
            # Fetch models from OpenRouter's public API
            response = self.requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            response.raise_for_status()
            models_data = response.json()
            
            # Filter for FREE models only
            free_models = []
            for model in models_data.get('data', []):
                model_id = model.get('id', '')
                pricing = model.get('pricing', {})
                context_length = model.get('context_length', 0)
                
                # STRICT free model detection to avoid false positives
                # Must have :free suffix AND zero pricing AND reasonable context length
                has_free_suffix = ':free' in model_id.lower()
                
                # Check pricing - must be explicitly "0" or 0
                prompt_price = pricing.get('prompt', '')
                completion_price = pricing.get('completion', '')
                
                # Convert to string for comparison
                prompt_str = str(prompt_price) if prompt_price is not None else ''
                completion_str = str(completion_price) if completion_price is not None else ''
                
                # Both must be exactly "0" or "0.0"
                has_zero_pricing = (
                    prompt_str in ["0", "0.0", "0.00"] and 
                    completion_str in ["0", "0.0", "0.00"]
                )
                
                # Must have reasonable context length (filters out broken models)
                has_context = context_length > 0
                
                # STRICT: Must have :free suffix OR (zero pricing AND context)
                # This filters out models that claim to be free but aren't
                is_truly_free = (
                    (has_free_suffix and has_context) or
                    (has_zero_pricing and has_context and has_free_suffix)
                )
                
                if is_truly_free:
                    model_info = {
                        "id": model_id,
                        "name": self._format_model_name(model),
                        "context": context_length  # Store for debugging
                    }
                    free_models.append(model_info)
            
            # Future-proof ranking: Use capability scoring
            free_models.sort(key=lambda m: get_model_capability_score(m['id'], m['name']))
            
            # Limit to top 15 free models (no auto-select)
            self._cached_models = free_models[:15]
            
            logger.info(f"✅ OpenRouter: Detected {len(self._cached_models)} free models")
            return self._cached_models
            
        except Exception as e:
            logger.warning(f"⚠️ OpenRouter: Could not fetch models dynamically: {e}")
            return self._get_fallback_models()
    
    def _format_model_name(self, model_obj: dict) -> str:
        """Format model object into readable name"""
        name = model_obj.get('name', model_obj.get('id', 'Unknown'))
        model_id = model_obj.get('id', '')
        
        # Add (Free) suffix
        if ':free' in model_id.lower():
            name = name.replace(' (free)', '').replace(' (Free)', '')
            name += " (Free)"
        
        return name
    
    def _get_fallback_models(self) -> List[Dict[str, str]]:
        """Fallback models if API detection fails (ranked smartest to least capable)"""
        return [
            {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3 70B Instruct (Smartest)"},
            {"id": "nousresearch/hermes-3-llama-3.1-405b:free", "name": "Hermes 3 405B"},
            {"id": "google/gemini-2.0-flash-exp:free", "name": "Gemini 2.0 Flash Experimental"},
            {"id": "qwen/qwen-2.5-72b-instruct:free", "name": "Qwen 2.5 72B Instruct"},
            {"id": "mistralai/mistral-7b-instruct:free", "name": "Mistral 7B Instruct"},
        ]
    
    def get_name(self) -> str:
        return "OpenRouter"
    
    def get_default_model(self) -> str:
        return self.default_model


class CerebrasProvider(AIProvider):
    """Cerebras AI Provider with Dynamic Model Detection"""
    
    def __init__(self, api_key: str):
        from cerebras.cloud.sdk import Cerebras
        self.client = Cerebras(api_key=api_key)
        # Smart default: Use the best free tier model
        self.default_model = "gpt-oss-120b"
        self._cached_models = None
    
    def chat(self, messages: List[Dict], model: Optional[str] = None) -> str:
        model = model or self.default_model
        # Inject system prompt if not already present
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        
        response = self.client.chat.completions.create(
            messages=chat_messages,
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Dynamically fetch available models from Cerebras API"""
        # Use cached models if available (refresh every bot restart)
        if self._cached_models:
            return self._cached_models
        
        try:
            # Fetch models from Cerebras API
            models_response = self.client.models.list()
            
            # Filter for chat models and sort by quality
            chat_models = []
            for model in models_response.data:
                # Only include active models
                if hasattr(model, 'id'):
                    model_info = {
                        "id": model.id,
                        "name": self._format_model_name(model.id)
                    }
                    chat_models.append(model_info)
            
            # Future-proof ranking: Use capability scoring
            chat_models.sort(key=lambda m: get_model_capability_score(m['id'], m['name']))
            self._cached_models = chat_models
            
            logger.info(f"✅ Cerebras: Detected {len(chat_models)} available models")
            return chat_models
            
        except Exception as e:
            logger.warning(f"⚠️ Cerebras: Could not fetch models dynamically: {e}")
            # Fallback to known models
            return self._get_fallback_models()
    
    def _format_model_name(self, model_id: str) -> str:
        """Format model ID into readable name"""
        # Convert model IDs like 'llama-3.3-70b' to 'Llama 3.3 70B'
        name = model_id.replace('-', ' ').title()
        return name
    
    def _get_fallback_models(self) -> List[Dict[str, str]]:
        """Fallback models if API detection fails (ranked smartest to least capable)"""
        return [
            # Free tier models (stable)
            {"id": "gpt-oss-120b", "name": "GPT-OSS 120B (Recommended)"},
            {"id": "llama3.1-8b", "name": "Llama 3.1 8B (Fast)"},
            # Preview models (trial - may be removed anytime)
            {"id": "qwen-3-235b-a22b-instruct-2507", "name": "Qwen 3 235B (Preview)"},
            {"id": "zai-glm-4.7", "name": "ZAI GLM 4.7 (Preview)"},
        ]
    
    def get_name(self) -> str:
        return "Cerebras"
    
    def get_default_model(self) -> str:
        return self.default_model


class NvidiaProvider(AIProvider):
    """NVIDIA AI Provider with Thinking/Reasoning Support"""
    
    # Models that DON'T support thinking
    MODELS_WITHOUT_THINKING = {
        'qwen/qwen3-coder-480b-a35b-instruct',
        'openai/gpt-oss-120b'
    }
    
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        # Smart default: Use stable free tier model
        self.default_model = 'openai/gpt-oss-120b'
        self._cached_models = None
    
    def supports_thinking(self, model_id: str) -> bool:
        """Check if a model supports thinking/reasoning"""
        return model_id not in self.MODELS_WITHOUT_THINKING
    
    def chat(self, messages: List[Dict], model: Optional[str] = None, enable_thinking: bool = False) -> str:
        model = model or self.default_model
        # Inject system prompt if not already present
        chat_messages = messages.copy()
        if not any(msg.get('role') == 'system' for msg in chat_messages):
            chat_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        
        # Only enable thinking if BOTH: model supports it AND user enabled it
        should_use_thinking = enable_thinking and self.supports_thinking(model)
        
        if should_use_thinking:
            # Use streaming to capture reasoning content
            response = self.client.chat.completions.create(
                messages=chat_messages,
                model=model,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True
            )
            
            # Collect reasoning and content separately
            reasoning_parts = []
            content_parts = []
            
            for chunk in response:
                if not getattr(chunk, "choices", None):
                    continue
                if len(chunk.choices) == 0:
                    continue
                
                delta = chunk.choices[0].delta
                
                # Extract reasoning content
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
                
                # Extract regular content
                if getattr(delta, "content", None) is not None:
                    content_parts.append(delta.content)
            
            # Combine reasoning and content
            full_response = ""
            if reasoning_parts:
                full_response = "💭 **Thinking:**\n" + "".join(reasoning_parts) + "\n\n"
            full_response += "".join(content_parts)
            
            return full_response
        else:
            # Regular non-streaming response
            response = self.client.chat.completions.create(
                messages=chat_messages,
                model=model,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Dynamically fetch available models from NVIDIA API"""
        # Use cached models if available (refresh every bot restart)
        if self._cached_models:
            return self._cached_models
        
        try:
            # Fetch models from NVIDIA API
            models_response = self.client.models.list()
            
            # Filter for chat models and sort by quality
            chat_models = []
            for model in models_response.data:
                # Only include active models
                if hasattr(model, 'id'):
                    model_info = {
                        "id": model.id,
                        "name": self._format_model_name(model.id)
                    }
                    chat_models.append(model_info)
            
            # Future-proof ranking: Use capability scoring
            chat_models.sort(key=lambda m: get_model_capability_score(m['id'], m['name']))
            self._cached_models = chat_models
            
            logger.info(f"✅ NVIDIA: Detected {len(chat_models)} available models")
            return chat_models
            
        except Exception as e:
            logger.warning(f"⚠️ NVIDIA: Could not fetch models dynamically: {e}")
            # Fallback to known models
            return self._get_fallback_models()
    
    def _format_model_name(self, model_id: str) -> str:
        """Format model ID into readable name"""
        # Extract the model name after the slash
        if '/' in model_id:
            name = model_id.split('/')[-1]
        else:
            name = model_id
        
        # Convert to title case and clean up
        name = name.replace('-', ' ').title()
        
        # Add thinking indicator
        if self.supports_thinking(model_id):
            name += " 💭"
        
        return name
    
    def _get_fallback_models(self) -> List[Dict[str, str]]:
        """Fallback models if API detection fails (ranked by stability and capability)"""
        return [
            # Free tier models (stable)
            {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B (Stable Free Tier)"},
            {"id": "qwen/qwen3-coder-480b-a35b-instruct", "name": "Qwen3 Coder 480B (Coding)"},
            # Premium models with thinking (may lose access)
            {"id": "deepseek-ai/deepseek-v3.2", "name": "DeepSeek V3.2 💭"},
            {"id": "deepseek-ai/deepseek-v3.1-terminus", "name": "DeepSeek V3.1 Terminus 💭"},
            {"id": "qwen/qwen3-235b-a22b", "name": "Qwen3 235B 💭"},
            {"id": "moonshotai/kimi-k2.5", "name": "Kimi K2.5 💭"},
            {"id": "z-ai/glm4.7", "name": "GLM 4.7 💭"},
        ]
    
    def get_name(self) -> str:
        return "NVIDIA"
    
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
        providers = self.list_providers()
        if not providers:
            raise ValueError("No providers available")
        return providers[0]  # Return first available
    
    def refresh_models(self):
        """Refresh model lists from all providers"""
        logger.info("🔄 Refreshing model lists from providers...")
        for name, provider in self.providers.items():
            provider._cached_models = None
            models = provider.get_available_models()
            logger.info(f"  {name}: {len(models)} models available")


# Initialize provider manager
provider_manager = ProviderManager()

# Store user sessions (in-memory, resets on restart)
user_sessions: Dict[str, Dict] = {}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_user_session(user_id: str) -> Dict:
    """Get or create user session"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "provider": provider_manager.get_default_provider(),
            "models": {},  # Store model per provider: {"groq": "model-id", "gemini": "model-id"}
            "history": [],
            "thinking_enabled": False  # For NVIDIA thinking models
        }
    return user_sessions[user_id]


def is_user_allowed(user_id: str) -> bool:
    """Check if user is allowed to use the bot."""
    # If ALLOWED_USER_IDS is empty, allow all users
    if not ALLOWED_USER_IDS:
        return True
    # Otherwise, check if user_id is in the allowed list
    return user_id in ALLOWED_USER_IDS


# ============================================================================
# MODEL VALIDATION CACHE
# ============================================================================

def load_validated_models() -> Dict[str, List[str]]:
    """Load validated models from cache file"""
    try:
        if os.path.exists(VALIDATED_MODELS_CACHE):
            with open(VALIDATED_MODELS_CACHE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load validated models cache: {e}")
    return {}

def save_validated_models(validated: Dict[str, List[str]]):
    """Save validated models to cache file"""
    try:
        with open(VALIDATED_MODELS_CACHE, 'w') as f:
            json.dump(validated, f, indent=2)
        logger.info(f"Saved validated models cache")
    except Exception as e:
        logger.error(f"Could not save validated models cache: {e}")

def get_validated_models(provider_name: str) -> List[str]:
    """Get list of validated model IDs for a provider"""
    validated = load_validated_models()
    provider_data = validated.get(provider_name, {})
    # Handle old format (list) and new format (dict)
    if isinstance(provider_data, list):
        return provider_data
    return provider_data.get('working', [])

def get_failed_models(provider_name: str) -> List[str]:
    """Get list of permanently failed model IDs for a provider"""
    validated = load_validated_models()
    provider_data = validated.get(provider_name, {})
    if isinstance(provider_data, dict):
        return provider_data.get('failed', [])
    return []

def add_validated_model(provider_name: str, model_id: str):
    """Mark a model as validated (working)"""
    validated = load_validated_models()
    if provider_name not in validated:
        validated[provider_name] = {'working': [], 'failed': []}
    # Migrate old format
    if isinstance(validated[provider_name], list):
        validated[provider_name] = {'working': validated[provider_name], 'failed': []}
    
    if model_id not in validated[provider_name]['working']:
        validated[provider_name]['working'].append(model_id)
        # Remove from failed if it was there
        if model_id in validated[provider_name]['failed']:
            validated[provider_name]['failed'].remove(model_id)
        save_validated_models(validated)

def add_failed_model(provider_name: str, model_id: str):
    """Mark a model as permanently failed (not available)"""
    validated = load_validated_models()
    if provider_name not in validated:
        validated[provider_name] = {'working': [], 'failed': []}
    # Migrate old format
    if isinstance(validated[provider_name], list):
        validated[provider_name] = {'working': validated[provider_name], 'failed': []}
    
    if model_id not in validated[provider_name]['failed']:
        validated[provider_name]['failed'].append(model_id)
        save_validated_models(validated)

def clear_validated_models(provider_name: Optional[str] = None):
    """Clear validated models cache (all or for specific provider)"""
    if provider_name:
        validated = load_validated_models()
        if provider_name in validated:
            del validated[provider_name]
            save_validated_models(validated)
    else:
        save_validated_models({})


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
        f"/models - List available models (auto-detected!)\n"
        f"/model - Switch model\n"
        f"/refresh - Refresh model lists from APIs\n"
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

async def refresh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Refresh model lists from all providers."""
    if not is_user_allowed(str(update.effective_user.id)):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    await update.message.reply_text("🔄 Refreshing model lists from all providers...")
    
    try:
        provider_manager.refresh_models()
        await update.message.reply_text(
            "✅ Model lists refreshed!\n\n"
            "Use /models to see the latest available models."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error refreshing models: {str(e)}")

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
        "• `/provider <name>` - Switch provider (groq/gemini/openrouter/cerebras/nvidia)\n"
        "• `/models` - List verified working models\n"
        "• `/models all` - List all available models from API\n"
        "• `/model <name>` - Switch to a specific model\n"
        "• `/refresh` - Refresh model lists from providers\n\n"
        "**Model Validation (NEW!):**\n"
        "• `/validate` - Test all models to see which actually work\n"
        "• `/verified` - Show only validated working models\n"
        "• `/clearvalidation` - Clear validation cache\n\n"
        "**Thinking Mode (NVIDIA only):**\n"
        "• `/thinking on` - Enable thinking/reasoning in responses 💭\n"
        "• `/thinking off` - Disable thinking for concise responses\n\n"
        "**Other Commands:**\n"
        "• `/clear` - Clear conversation history\n"
        "• `/help` - Show this help\n\n"
        "**Features:**\n"
        "✨ Auto-detects new models from providers\n"
        "✨ Validates models with real API tests\n"
        "✨ Shows only working models by default!",
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
    # Get model for new provider (or use default)
    current_model = session["models"].get(new_provider_name) or new_provider.get_default_model()
    await update.message.reply_text(
        f"✅ Switched to **{new_provider.get_name()}**!\n"
        f"Using model: `{current_model}`",
        parse_mode='Markdown'
    )

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List available models for current provider (verified only by default)."""
    user_id = str(update.effective_user.id)
    
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    session = get_user_session(user_id)
    provider = provider_manager.get_provider(session["provider"])
    provider_name = session["provider"]
    
    # Check if user wants to see all models
    show_all = len(context.args) > 0 and context.args[0].lower() == 'all'
    
    if show_all:
        await update.message.reply_text("🔄 Fetching all models from API...")
        models = provider.get_available_models()
        title_suffix = " (All Models)"
        footer_note = "\n\n💡 Use `/models` to see only verified models"
    else:
        # Show only verified models
        validated_ids = get_validated_models(provider_name)
        
        if not validated_ids:
            await update.message.reply_text(
                f"❌ **No verified models yet for {provider.get_name()}**\n\n"
                f"Run `/validate` first to test which models work!\n\n"
                f"Or use `/models all` to see all available models.",
                parse_mode='Markdown'
            )
            return
        
        await update.message.reply_text("✅ Showing verified models...")
        all_models = provider.get_available_models()
        models = [m for m in all_models if m['id'] in validated_ids]
        title_suffix = " (Verified Only)"
        footer_note = f"\n\n💡 Use `/models all` to see all {len(all_models)} models"
    
    # Get current model for this provider
    current_model = session["models"].get(provider_name) or provider.get_default_model()
    
    # Split into chunks if too many models (Telegram message limit)
    max_models_per_message = 20
    
    if len(models) > max_models_per_message:
        model_chunks = [models[i:i+max_models_per_message] for i in range(0, len(models), max_models_per_message)]
        for idx, chunk in enumerate(model_chunks, 1):
            model_list = "\n".join([
                f"• `{m['id']}`" + (f"\n  {m['name']}" if m['name'] != m['id'] else "") + (" ✓" if m['id'] == current_model else "")
                for m in chunk
            ])
            
            await update.message.reply_text(
                f"🤖 **{provider.get_name()}{title_suffix}** (Part {idx}/{len(model_chunks)}):\n\n"
                f"{model_list}",
                parse_mode='Markdown'
            )
    else:
        model_list = "\n".join([
            f"• `{m['id']}`" + (f"\n  {m['name']}" if m['name'] != m['id'] else "") + (" ✓" if m['id'] == current_model else "")
            for m in models
        ])
        
        await update.message.reply_text(
            f"🤖 **{provider.get_name()}{title_suffix}:**\n\n"
            f"{model_list}\n\n"
            f"Current: `{current_model}`{footer_note}\n\n"
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
    provider_name = session["provider"]
    provider = provider_manager.get_provider(provider_name)
    
    # If no argument, show current model
    if not context.args:
        current_model = session["models"].get(provider_name) or provider.get_default_model()
        await update.message.reply_text(
            f"🤖 **Current Model for {provider.get_name()}:** `{current_model}`\n\n"
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
    
    # Save model for this provider
    session["models"][provider_name] = new_model
    await update.message.reply_text(
        f"✅ Switched to model: `{new_model}`\n\n"
        f"💾 Model saved for {provider.get_name()}!\n"
        f"It will be remembered when you switch back.",
        parse_mode='Markdown'
    )

async def validate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Validate all models for current provider by testing them"""
    user_id = str(update.effective_user.id)
    
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    session = get_user_session(user_id)
    provider = provider_manager.get_provider(session["provider"])
    provider_name = session["provider"]
    
    # Check if user wants to retry only failed models
    retry_only = len(context.args) > 0 and context.args[0].lower() == 'retry'
    
    # Get all available models
    models = provider.get_available_models()
    total = len(models)
    
    # Get already validated and permanently failed models
    already_validated = get_validated_models(provider_name)
    permanently_failed = get_failed_models(provider_name)
    
    # Filter models to test (skip validated AND permanently failed)
    models_to_test = [m for m in models if m['id'] not in already_validated and m['id'] not in permanently_failed]
    skipped_validated = len(already_validated)
    skipped_failed = len(permanently_failed)
    
    # Show smart validation message
    if retry_only:
        await update.message.reply_text(
            f"🔄 **Retry Mode**\n\n"
            f"• Total: {total}\n"
            f"• Already validated: {skipped_validated}\n"
            f"• Permanently failed: {skipped_failed}\n"
            f"• To test: {len(models_to_test)}\n\n"
            f"⏳ ~{len(models_to_test) * 2}s (2s delay per model)",
            parse_mode='Markdown'
        )
    elif (skipped_validated + skipped_failed) > 0:
        await update.message.reply_text(
            f"💡 **Smart Validation**\n\n"
            f"Skipping {skipped_validated + skipped_failed} models:\n"
            f"• ✅ Working: {skipped_validated}\n"
            f"• ❌ Permanently failed: {skipped_failed}\n\n"
            f"• To test: {len(models_to_test)}/{total}\n"
            f"⏳ ~{len(models_to_test) * 2}s\n\n"
            f"Use `/clearvalidation` to re-test all",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            f"🔍 **Full Validation**\n\n"
            f"Testing all {total} models\n"
            f"⏳ ~{total * 2}s (2s delay to avoid rate limits)",
            parse_mode='Markdown'
        )
    
    if len(models_to_test) == 0:
        await update.message.reply_text(
            "✅ All models already validated!\n\n"
            "Use `/verified` to see them",
            parse_mode='Markdown'
        )
        return
    
    validated = list(already_validated)
    failed_not_available = []
    failed_rate_limit = []
    failed_unknown = []
    newly_validated = []
    
    # Test each model with rate limit protection
    for idx, model_info in enumerate(models_to_test, 1):
        model_id = model_info['id']
        
        # Send progress update every 5 models
        if idx % 5 == 0 or idx == 1:
            await update.message.reply_text(
                f"⏳ {idx}/{len(models_to_test)}: `{model_id}`...",
                parse_mode='Markdown'
            )
        
        # Test the model
        logger.info(f"Testing model: {model_id}")
        success, error_type = provider.test_model(model_id)
        
        if success:
            validated.append(model_id)
            newly_validated.append(model_id)
            add_validated_model(provider_name, model_id)
            logger.info(f"✅ Model {model_id} validated")
        else:
            # Categorize failure
            if error_type == 'not_available':
                failed_not_available.append(model_id)
                add_failed_model(provider_name, model_id)  # Cache permanently
                logger.info(f"❌ Model {model_id} not available (cached)")
            elif error_type == 'rate_limit':
                failed_rate_limit.append(model_id)
                logger.info(f"⏱️ Model {model_id} rate limited (will retry)")
            else:
                failed_unknown.append(model_id)
                logger.info(f"❓ Model {model_id} failed (unknown error)")
        
        # Rate limit protection: 2 second delay
        if idx < len(models_to_test):
            await asyncio.sleep(2)
    
    # Send results
    total_failed = len(failed_not_available) + len(failed_rate_limit) + len(failed_unknown)
    success_rate = (len(newly_validated) / len(models_to_test) * 100) if len(models_to_test) > 0 else 0
    
    result_message = (
        f"✅ **Validation Complete**\n\n"
        f"📊 **Results:**\n"
        f"• Tested: {len(models_to_test)}\n"
        f"• ✅ Newly validated: {len(newly_validated)}\n"
        f"• ❌ Failed: {total_failed}\n"
    )
    
    # Show failure breakdown if there are failures
    if total_failed > 0:
        result_message += (
            f"\n**Failure breakdown:**\n"
            f"• 🚫 Not available: {len(failed_not_available)} (cached)\n"
            f"• ⏱️ Rate limited: {len(failed_rate_limit)} (retry later)\n"
            f"• ❓ Unknown: {len(failed_unknown)}\n"
        )
    
    result_message += (
        f"\n• Success rate: {success_rate:.1f}%\n\n"
        f"📦 **Total validated: {len(validated)}**\n"
        f"🚫 **Permanently failed: {len(permanently_failed) + len(failed_not_available)}**\n\n"
        f"Use `/verified` to see working models!"
    )
    
    await update.message.reply_text(result_message, parse_mode='Markdown')

async def verified_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show only verified/validated models for current provider"""
    user_id = str(update.effective_user.id)
    
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    session = get_user_session(user_id)
    provider = provider_manager.get_provider(session["provider"])
    provider_name = session["provider"]
    
    # Get validated models
    validated_ids = get_validated_models(provider_name)
    
    if not validated_ids:
        await update.message.reply_text(
            f"❌ **No validated models for {provider.get_name()}**\n\n"
            f"Run `/validate` first to test which models actually work!",
            parse_mode='Markdown'
        )
        return
    
    # Get full model info
    all_models = provider.get_available_models()
    validated_models = [m for m in all_models if m['id'] in validated_ids]
    
    # Get current model
    current_model = session["models"].get(provider_name) or provider.get_default_model()
    
    # Build model list
    model_list = "\n".join([
        f"• `{m['id']}`" + (f"\n  {m['name']}" if m['name'] != m['id'] else "") + (" ✓" if m['id'] == current_model else "")
        for m in validated_models
    ])
    
    await update.message.reply_text(
        f"✅ **Verified Models for {provider.get_name()}:**\n\n"
        f"{model_list}\n\n"
        f"Current: `{current_model}`\n\n"
        f"💡 These models have been tested and confirmed working!\n"
        f"Use `/model <id>` to switch.",
        parse_mode='Markdown'
    )

async def thinking_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle thinking mode on/off for NVIDIA models"""
    user_id = str(update.effective_user.id)
    
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    session = get_user_session(user_id)
    provider_name = session["provider"]
    provider = provider_manager.get_provider(provider_name)
    
    # Check if no argument provided
    if not context.args:
        current_state = "enabled" if session.get("thinking_enabled", False) else "disabled"
        await update.message.reply_text(
            f"💭 **Thinking Mode:** {current_state}\n\n"
            f"Use `/thinking on` or `/thinking off` to toggle.",
            parse_mode='Markdown'
        )
        return
    
    # Parse argument
    arg = context.args[0].lower()
    
    if arg == 'on':
        session["thinking_enabled"] = True
        
        # Check if current model supports thinking (only for NVIDIA)
        if provider_name == 'nvidia':
            current_model = session["models"].get(provider_name) or provider.get_default_model()
            if hasattr(provider, 'supports_thinking') and provider.supports_thinking(current_model):
                await update.message.reply_text(
                    f"✅ **Thinking mode enabled!** 💭\n\n"
                    f"Model `{current_model}` supports thinking.\n"
                    f"You'll see reasoning content in responses.",
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text(
                    f"⚠️ **Thinking mode enabled**, but...\n\n"
                    f"Model `{current_model}` doesn't support thinking.\n"
                    f"Switch to a model with 💭 icon for thinking support.\n\n"
                    f"Use `/models all` to see available models.",
                    parse_mode='Markdown'
                )
        else:
            await update.message.reply_text(
                f"✅ **Thinking mode enabled!**\n\n"
                f"Note: Only NVIDIA provider supports thinking.\n"
                f"Switch to NVIDIA with `/provider nvidia` to use it.",
                parse_mode='Markdown'
            )
    
    elif arg == 'off':
        session["thinking_enabled"] = False
        await update.message.reply_text(
            f"🔕 **Thinking mode disabled!**\n\n"
            f"Responses will be more concise.",
            parse_mode='Markdown'
        )
    
    else:
        await update.message.reply_text(
            f"❌ Invalid argument: `{arg}`\n\n"
            f"Use `/thinking on` or `/thinking off`",
            parse_mode='Markdown'
        )

async def clearvalidation_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear validation cache for current provider"""
    user_id = str(update.effective_user.id)
    
    if not is_user_allowed(user_id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    session = get_user_session(user_id)
    provider_name = session["provider"]
    provider = provider_manager.get_provider(provider_name)
    
    clear_validated_models(provider_name)
    
    await update.message.reply_text(
        f"🗑️ Cleared validation cache for {provider.get_name()}!\n\n"
        f"Run `/validate` to re-test models.",
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
    
    # Keep only last N messages to avoid token limits (configurable)
    if len(session["history"]) > MAX_HISTORY_MESSAGES:
        session["history"] = session["history"][-MAX_HISTORY_MESSAGES:]
    
    try:
        # Send typing indicator
        await update.message.chat.send_action(action="typing")
        
        # Get current provider and model
        provider_name = session["provider"]
        provider = provider_manager.get_provider(provider_name)
        current_model = session["models"].get(provider_name)  # None = use provider default
        
        # Call AI provider (pass thinking flag for NVIDIA)
        thinking_enabled = session.get("thinking_enabled", False)
        bot_response = provider.chat(
            messages=session["history"],
            model=current_model,
            enable_thinking=thinking_enabled
        )
        
        # Add bot response to history
        session["history"].append({
            "role": "assistant",
            "content": bot_response
        })
        
        # Send response (split if too long for Telegram's 4096 char limit)
        MAX_MESSAGE_LENGTH = 4096
        if len(bot_response) <= MAX_MESSAGE_LENGTH:
            await update.message.reply_text(bot_response)
        else:
            # Split message into chunks at newlines to preserve formatting
            chunks = []
            current_chunk = ""
            
            for line in bot_response.split('\n'):
                # If adding this line would exceed limit, save current chunk and start new one
                if len(current_chunk) + len(line) + 1 > MAX_MESSAGE_LENGTH:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = line
                else:
                    if current_chunk:
                        current_chunk += '\n' + line
                    else:
                        current_chunk = line
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            # Send all chunks
            for i, chunk in enumerate(chunks, 1):
                if len(chunks) > 1:
                    # Add part indicator for multi-part messages
                    header = f"📄 Part {i}/{len(chunks)}\n\n"
                    await update.message.reply_text(header + chunk)
                else:
                    await update.message.reply_text(chunk)
        
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
    if not (GROQ_API_KEY or GEMINI_API_KEY or OPENROUTER_API_KEY or CEREBRAS_API_KEY or NVIDIA_API_KEY):
        raise ValueError(
            "At least one AI provider API key is required!\n"
            "Set one of: GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, CEREBRAS_API_KEY, or NVIDIA_API_KEY"
        )
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Register handlers
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
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    logger.info("🚀 Multi-Provider AI Bot started!")
    logger.info(f"📡 Available providers: {', '.join(provider_manager.list_providers())}")
    logger.info("🔄 Model lists will be fetched dynamically from APIs")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()