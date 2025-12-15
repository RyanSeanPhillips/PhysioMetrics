"""
AI Client - Unified interface for LLM APIs (Claude, GPT, Gemini, Ollama).

This module provides a simple interface for integrating AI capabilities
into PhysioMetrics. Users bring their own API keys (or use Ollama locally for free).

Supported Providers:
    - Claude (Anthropic) - requires API key
    - OpenAI (GPT) - requires API key
    - Gemini (Google) - requires API key
    - Ollama (Local) - FREE, runs locally, no API key needed
      Download: https://ollama.com/download

Usage:
    from core.ai_client import AIClient

    # Cloud providers (require API keys)
    client = AIClient(provider='claude', api_key='sk-ant-...')
    response = client.complete("Analyze this respiratory data...")

    # Ollama (free, local - no API key needed!)
    client = AIClient(provider='ollama', model='llama3.2')
    response = client.complete("Analyze this respiratory data...")

    # With image (for vision-capable models)
    response = client.complete("What do you see?", image_path="trace.png")
"""

import os
import json
import base64
from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
from enum import Enum


class AIProvider(Enum):
    """Supported AI providers."""
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"  # Free, local - https://ollama.com/download


@dataclass
class AIResponse:
    """Standardized response from AI models."""
    content: str
    provider: str
    model: str
    usage: Dict  # Token counts
    raw_response: Dict  # Full API response for debugging


@dataclass
class ToolResponse:
    """Response from AI models with tool use support."""
    content: str  # Text content (may be empty if only tool calls)
    tool_calls: List[Dict]  # List of {id, name, input} dicts
    stop_reason: str  # "tool_use", "end_turn", etc.
    provider: str
    model: str
    usage: Dict
    raw_response: Dict

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class AIClient:
    """
    Unified AI client supporting multiple providers.

    Provides a consistent interface regardless of which AI provider is used.
    """

    # Default models for each provider
    DEFAULT_MODELS = {
        AIProvider.CLAUDE: "claude-sonnet-4-20250514",
        AIProvider.OPENAI: "gpt-4o",
        AIProvider.GEMINI: "gemini-1.5-pro",
        AIProvider.OLLAMA: "llama3.2",  # Good balance of speed/quality, or use "llama3.2:1b" for faster
    }

    # Ollama default URL (can be overridden)
    OLLAMA_BASE_URL = "http://localhost:11434"

    def __init__(self, provider: str = "claude", api_key: str = None, model: str = None,
                 ollama_base_url: str = None):
        """
        Initialize AI client.

        Args:
            provider: One of 'claude', 'openai', 'gemini', 'ollama'
            api_key: API key for the provider (or set via environment variable)
                     Not required for Ollama (runs locally for free!)
            model: Specific model to use (or uses default for provider)
            ollama_base_url: Custom Ollama server URL (default: http://localhost:11434)
        """
        # Normalize provider
        provider_lower = provider.lower()
        if provider_lower == "claude" or provider_lower == "anthropic":
            self.provider = AIProvider.CLAUDE
        elif provider_lower == "openai" or provider_lower == "gpt":
            self.provider = AIProvider.OPENAI
        elif provider_lower == "gemini" or provider_lower == "google":
            self.provider = AIProvider.GEMINI
        elif provider_lower == "ollama" or provider_lower == "local":
            self.provider = AIProvider.OLLAMA
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude', 'openai', 'gemini', or 'ollama'")

        # Ollama doesn't require API key (runs locally for free!)
        if self.provider == AIProvider.OLLAMA:
            self.api_key = None
            self.ollama_base_url = ollama_base_url or self.OLLAMA_BASE_URL
        else:
            # Get API key from parameter or environment
            self.api_key = api_key or self._get_api_key_from_env()
            if not self.api_key:
                raise ValueError(f"No API key provided for {self.provider.value}. "
                               f"Set {self._get_env_var_name()} or pass api_key parameter.")

        # Set model
        self.model = model or self.DEFAULT_MODELS[self.provider]

        # Initialize provider-specific client
        self._client = None
        self._init_client()

    def _get_env_var_name(self) -> str:
        """Get environment variable name for API key."""
        return {
            AIProvider.CLAUDE: "ANTHROPIC_API_KEY",
            AIProvider.OPENAI: "OPENAI_API_KEY",
            AIProvider.GEMINI: "GOOGLE_API_KEY",
        }[self.provider]

    def _get_api_key_from_env(self) -> Optional[str]:
        """Try to get API key from environment variable."""
        return os.environ.get(self._get_env_var_name())

    def _init_client(self):
        """Initialize the provider-specific client library."""
        if self.provider == AIProvider.CLAUDE:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")

        elif self.provider == AIProvider.OPENAI:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")

        elif self.provider == AIProvider.GEMINI:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("google-generativeai package not installed. "
                                "Run: pip install google-generativeai")

        elif self.provider == AIProvider.OLLAMA:
            # Ollama uses OpenAI-compatible API, so we use the openai client
            # pointed at the local Ollama server
            try:
                import openai
                self._client = openai.OpenAI(
                    base_url=f"{self.ollama_base_url}/v1",
                    api_key="ollama"  # Ollama doesn't check this, but openai client requires it
                )
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai\n"
                                "Note: Ollama uses the OpenAI-compatible API.")

    def complete(self,
                 prompt: str,
                 system_prompt: str = None,
                 image_path: Union[str, Path] = None,
                 image_data: bytes = None,
                 max_tokens: int = 4096,
                 temperature: float = 0.7) -> AIResponse:
        """
        Send a completion request to the AI model.

        Args:
            prompt: The user's message/question
            system_prompt: Optional system instructions
            image_path: Path to an image file (for vision models)
            image_data: Raw image bytes (alternative to image_path)
            max_tokens: Maximum response length
            temperature: Creativity (0.0 = deterministic, 1.0 = creative)

        Returns:
            AIResponse with content and metadata
        """
        # Load image if path provided
        if image_path and not image_data:
            image_path = Path(image_path)
            if image_path.exists():
                with open(image_path, 'rb') as f:
                    image_data = f.read()

        # Route to provider-specific implementation
        if self.provider == AIProvider.CLAUDE:
            return self._complete_claude(prompt, system_prompt, image_data, max_tokens, temperature)
        elif self.provider == AIProvider.OPENAI:
            return self._complete_openai(prompt, system_prompt, image_data, max_tokens, temperature)
        elif self.provider == AIProvider.GEMINI:
            return self._complete_gemini(prompt, system_prompt, image_data, max_tokens, temperature)
        elif self.provider == AIProvider.OLLAMA:
            return self._complete_ollama(prompt, system_prompt, image_data, max_tokens, temperature)

    def chat(self,
             messages: List[Dict],
             system_prompt: str = None,
             max_tokens: int = 4096,
             temperature: float = 0.7) -> AIResponse:
        """
        Send a chat request with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                      e.g., [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
            system_prompt: Optional system instructions
            max_tokens: Maximum response length
            temperature: Creativity (0.0 = deterministic, 1.0 = creative)

        Returns:
            AIResponse with content and metadata
        """
        if self.provider == AIProvider.CLAUDE:
            return self._chat_claude(messages, system_prompt, max_tokens, temperature)
        elif self.provider == AIProvider.OPENAI:
            return self._chat_openai(messages, system_prompt, max_tokens, temperature)
        elif self.provider == AIProvider.GEMINI:
            return self._chat_gemini(messages, system_prompt, max_tokens, temperature)
        elif self.provider == AIProvider.OLLAMA:
            return self._chat_ollama(messages, system_prompt, max_tokens, temperature)

    def chat_with_tools(self,
                        messages: List[Dict],
                        tools: List[Dict],
                        system_prompt: str = None,
                        max_tokens: int = 4096,
                        temperature: float = 0.7) -> 'ToolResponse':
        """
        Send a chat request with tool definitions (native tool use).

        Args:
            messages: Conversation history
            tools: List of tool definitions with name, description, input_schema
            system_prompt: Optional system instructions
            max_tokens: Maximum response length
            temperature: Creativity

        Returns:
            ToolResponse with content, tool_calls, and metadata
        """
        if self.provider == AIProvider.CLAUDE:
            return self._chat_with_tools_claude(messages, tools, system_prompt, max_tokens, temperature)
        elif self.provider == AIProvider.OPENAI:
            return self._chat_with_tools_openai(messages, tools, system_prompt, max_tokens, temperature)
        elif self.provider == AIProvider.OLLAMA:
            # Ollama: fall back to regular chat (tool support varies by model)
            return self._chat_with_tools_ollama(messages, tools, system_prompt, max_tokens, temperature)
        else:
            raise NotImplementedError(f"Tool use not implemented for {self.provider.value}")

    def _chat_with_tools_claude(self, messages, tools, system_prompt, max_tokens, temperature) -> 'ToolResponse':
        """Claude native tool use."""
        # Convert messages - handle both simple strings and complex content
        claude_messages = []
        for msg in messages:
            content = msg["content"]
            # If content is already a list (tool results), use as-is
            if isinstance(content, list):
                claude_messages.append({"role": msg["role"], "content": content})
            else:
                claude_messages.append({"role": msg["role"], "content": content})

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": claude_messages,
            "tools": tools,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self._client.messages.create(**kwargs)

        # Parse response content blocks
        text_content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

        return ToolResponse(
            content=text_content,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            provider="claude",
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
        )

    def _chat_with_tools_openai(self, messages, tools, system_prompt, max_tokens, temperature) -> 'ToolResponse':
        """OpenAI native tool use (function calling)."""
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            openai_messages.append({"role": msg["role"], "content": msg["content"]})

        # Convert tools to OpenAI format
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            })

        response = self._client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=openai_tools if openai_tools else None,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Parse response
        message = response.choices[0].message
        text_content = message.content or ""
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments) if tc.function.arguments else {}
                })

        return ToolResponse(
            content=text_content,
            tool_calls=tool_calls,
            stop_reason=response.choices[0].finish_reason,
            provider="openai",
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
        )

    def _chat_with_tools_ollama(self, messages, tools, system_prompt, max_tokens, temperature) -> 'ToolResponse':
        """
        Ollama chat (tools not natively supported in most models).
        Falls back to regular chat but returns ToolResponse format.
        """
        # Use regular Ollama chat (tool descriptions can be included in system prompt if needed)
        chat_response = self._chat_ollama(messages, system_prompt, max_tokens, temperature)

        # Convert AIResponse to ToolResponse format (no tool calls)
        return ToolResponse(
            content=chat_response.content,
            tool_calls=[],  # Ollama doesn't support tool calling in most models
            stop_reason="end_turn",
            provider="ollama",
            model=self.model,
            usage=chat_response.usage,
            raw_response=chat_response.raw_response
        )

    @staticmethod
    def build_tool_result_message(tool_call_id: str, result: Union[str, Dict], is_error: bool = False) -> Dict:
        """
        Build a tool result message to send back after executing a tool.

        Args:
            tool_call_id: The ID from the tool_call
            result: The result data (will be JSON serialized if dict)
            is_error: Whether this is an error result

        Returns:
            Message dict ready to append to conversation
        """
        if isinstance(result, dict):
            result_str = json.dumps(result, indent=2, default=str)
        else:
            result_str = str(result)

        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result_str,
                "is_error": is_error
            }]
        }

    def _chat_claude(self, messages, system_prompt, max_tokens, temperature) -> AIResponse:
        """Claude chat with conversation history."""
        # Convert messages to Claude format
        claude_messages = []
        for msg in messages:
            claude_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": claude_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self._client.messages.create(**kwargs)

        return AIResponse(
            content=response.content[0].text,
            provider="claude",
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
        )

    def _chat_openai(self, messages, system_prompt, max_tokens, temperature) -> AIResponse:
        """OpenAI chat with conversation history."""
        openai_messages = []

        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            openai_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        response = self._client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return AIResponse(
            content=response.choices[0].message.content,
            provider="openai",
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
        )

    def _chat_gemini(self, messages, system_prompt, max_tokens, temperature) -> AIResponse:
        """Google Gemini chat with conversation history."""
        import google.generativeai as genai

        # Build conversation history for Gemini
        history = []
        for msg in messages[:-1]:  # All but last message go to history
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        # Start chat with history
        chat = self._client.start_chat(history=history)

        # Get the last message to send
        last_message = messages[-1]["content"] if messages else ""
        if system_prompt and not history:
            last_message = f"{system_prompt}\n\n{last_message}"

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        response = chat.send_message(last_message, generation_config=generation_config)

        return AIResponse(
            content=response.text,
            provider="gemini",
            model=self.model,
            usage={
                "input_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "output_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
            },
            raw_response={}
        )

    def _complete_claude(self, prompt, system_prompt, image_data, max_tokens, temperature) -> AIResponse:
        """Claude/Anthropic completion."""
        messages = []

        # Build message content
        content = []

        # Add image if provided
        if image_data:
            # Detect image type from magic bytes
            media_type = "image/png"
            if image_data[:3] == b'\xff\xd8\xff':
                media_type = "image/jpeg"
            elif image_data[:4] == b'\x89PNG':
                media_type = "image/png"
            elif image_data[:4] == b'GIF8':
                media_type = "image/gif"
            elif image_data[:4] == b'RIFF':
                media_type = "image/webp"

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(image_data).decode('utf-8')
                }
            })

        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        # Make API call
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self._client.messages.create(**kwargs)

        return AIResponse(
            content=response.content[0].text,
            provider="claude",
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
        )

    def _complete_openai(self, prompt, system_prompt, image_data, max_tokens, temperature) -> AIResponse:
        """OpenAI/GPT completion."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message
        if image_data:
            # Encode image
            media_type = "image/png"
            if image_data[:3] == b'\xff\xd8\xff':
                media_type = "image/jpeg"

            b64_image = base64.standard_b64encode(image_data).decode('utf-8')

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_image}"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return AIResponse(
            content=response.choices[0].message.content,
            provider="openai",
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
        )

    def _complete_gemini(self, prompt, system_prompt, image_data, max_tokens, temperature) -> AIResponse:
        """Google Gemini completion."""
        import google.generativeai as genai

        # Build content parts
        parts = []

        if image_data:
            # Gemini expects PIL Image or raw bytes with mime type
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_data))
            parts.append(img)

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        parts.append(full_prompt)

        # Configure generation
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        response = self._client.generate_content(
            parts,
            generation_config=generation_config
        )

        return AIResponse(
            content=response.text,
            provider="gemini",
            model=self.model,
            usage={
                "input_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "output_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
            },
            raw_response={}
        )

    def _complete_ollama(self, prompt, system_prompt, image_data, max_tokens, temperature) -> AIResponse:
        """Ollama local completion (uses OpenAI-compatible API)."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message
        if image_data:
            # Ollama vision models support base64 images
            media_type = "image/png"
            if image_data[:3] == b'\xff\xd8\xff':
                media_type = "image/jpeg"

            b64_image = base64.standard_b64encode(image_data).decode('utf-8')

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_image}"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return AIResponse(
                content=response.choices[0].message.content,
                provider="ollama",
                model=self.model,
                usage={
                    "input_tokens": getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
                    "output_tokens": getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
                },
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
            )
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "connection" in error_msg.lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.ollama_base_url}.\n\n"
                    "Make sure Ollama is running:\n"
                    "1. Download Ollama (FREE): https://ollama.com/download\n"
                    "2. Install and run it\n"
                    "3. Pull a model: ollama pull llama3.2\n"
                    "4. Try again"
                )
            raise

    def _chat_ollama(self, messages, system_prompt, max_tokens, temperature) -> AIResponse:
        """Ollama chat with conversation history (uses OpenAI-compatible API)."""
        ollama_messages = []

        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=ollama_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return AIResponse(
                content=response.choices[0].message.content,
                provider="ollama",
                model=self.model,
                usage={
                    "input_tokens": getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
                    "output_tokens": getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
                },
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
            )
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "connection" in error_msg.lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.ollama_base_url}.\n\n"
                    "Make sure Ollama is running:\n"
                    "1. Download Ollama (FREE): https://ollama.com/download\n"
                    "2. Install and run it\n"
                    "3. Pull a model: ollama pull llama3.2\n"
                    "4. Try again"
                )
            raise

    def analyze_respiratory_trace(self,
                                   image_path: Union[str, Path],
                                   question: str = "Describe what you see in this respiratory trace.") -> AIResponse:
        """
        Analyze a respiratory trace image using vision capabilities.

        Args:
            image_path: Path to screenshot of trace
            question: Specific question about the trace

        Returns:
            AIResponse with analysis
        """
        system_prompt = """You are an expert in respiratory physiology analyzing plethysmography traces.
The trace shows airflow or pressure signals from breathing recordings.
- Positive deflections typically indicate inspiration (breathing in)
- Negative deflections indicate expiration (breathing out)
- Look for: breath rate variability, sighs (large breaths), apneas (pauses), sniffing (rapid shallow breaths)
- Note any artifacts or unusual patterns that might affect analysis."""

        return self.complete(
            prompt=question,
            system_prompt=system_prompt,
            image_path=image_path,
            temperature=0.3  # Lower temperature for more consistent analysis
        )

    def suggest_file_groupings(self, files_metadata: List[Dict]) -> AIResponse:
        """
        Ask AI to suggest how to group files into experiments.

        Args:
            files_metadata: List of dicts with file info (name, protocol, path, etc.)

        Returns:
            AIResponse with suggested groupings in JSON format
        """
        # Build context from file metadata
        files_summary = []
        for f in files_metadata:
            summary = {
                "file_name": f.get('file_name', ''),
                "protocol": f.get('protocol', ''),
                "folder": str(Path(f.get('file_path', '')).parent.name) if f.get('file_path') else '',
                "keywords": f.get('keywords_display', ''),
            }
            files_summary.append(summary)

        system_prompt = """You are helping organize electrophysiology data files.
Analyze the file names, protocols, and folder structure to suggest logical experiment groupings.
Return your suggestions as JSON in this format:
{
    "groups": [
        {
            "name": "Suggested experiment name",
            "files": ["file1.abf", "file2.abf"],
            "reasoning": "Why these files belong together"
        }
    ],
    "ungrouped": ["files that don't clearly belong to a group"]
}"""

        prompt = f"""Please analyze these ABF files and suggest how to group them into experiments:

{json.dumps(files_summary, indent=2)}

Consider:
- Protocol names often indicate experiment type (e.g., "30Hz_opto", "baseline")
- Folder names may indicate animal ID or date
- File naming patterns
- Keywords in file names

Return only the JSON grouping suggestion."""

        return self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2  # Low temperature for consistent groupings
        )

    def extract_metadata_from_notes(self, notes_content: str, file_names: List[str]) -> AIResponse:
        """
        Extract metadata from experiment notes and map to files.

        Args:
            notes_content: Text content from experiment notes file
            file_names: List of ABF file names to match

        Returns:
            AIResponse with extracted metadata in JSON format
        """
        system_prompt = """You are extracting experimental metadata from lab notes.
Look for information like:
- File names and their associated conditions
- Animal IDs (often numbers like "4523", "M1", etc.)
- Sex (M/F)
- Strain (e.g., "VgatCre", "C57BL/6")
- Laser/stim power (e.g., "10mW", "5mW")
- Experimental conditions (baseline, drug, etc.)

Return JSON mapping file names to their metadata:
{
    "files": {
        "filename.abf": {
            "animal_id": "extracted value or null",
            "sex": "M/F or null",
            "strain": "extracted value or null",
            "power": "extracted value or null",
            "condition": "extracted value or null"
        }
    },
    "notes": "Any relevant notes about extraction accuracy"
}"""

        prompt = f"""Extract metadata from these experiment notes and map to the listed files.

Files to match:
{json.dumps(file_names, indent=2)}

Experiment notes:
---
{notes_content}
---

Return only the JSON with extracted metadata."""

        return self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1  # Very low temperature for accurate extraction
        )


# Convenience functions
def get_available_providers() -> List[str]:
    """Get list of provider names that could be used."""
    return ["ollama", "claude", "openai", "gemini"]  # Ollama first since it's free


def get_provider_info() -> Dict[str, Dict]:
    """Get detailed info about each provider."""
    return {
        "ollama": {
            "name": "Ollama (Local)",
            "description": "FREE - Runs locally on your computer",
            "requires_api_key": False,
            "download_url": "https://ollama.com/download",
            "default_model": "llama3.2",
            "models": ["llama3.2", "llama3.2:1b", "llama3.1", "mistral", "codellama", "llava"],
        },
        "claude": {
            "name": "Claude (Anthropic)",
            "description": "Paid - Requires API key",
            "requires_api_key": True,
            "signup_url": "https://console.anthropic.com/",
            "default_model": "claude-sonnet-4-20250514",
            "models": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        },
        "openai": {
            "name": "OpenAI (GPT)",
            "description": "Paid - Requires API key",
            "requires_api_key": True,
            "signup_url": "https://platform.openai.com/api-keys",
            "default_model": "gpt-4o",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        },
        "gemini": {
            "name": "Gemini (Google)",
            "description": "Paid - Requires API key",
            "requires_api_key": True,
            "signup_url": "https://aistudio.google.com/apikey",
            "default_model": "gemini-1.5-pro",
            "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
        },
    }


def check_provider_available(provider: str) -> Dict[str, bool]:
    """
    Check if a provider's SDK is installed and API key/server is available.

    Returns:
        Dict with 'sdk_installed', 'api_key_set' (or 'server_running' for Ollama)
    """
    result = {"sdk_installed": False, "api_key_set": False, "server_running": False}

    provider = provider.lower()

    if provider == "claude":
        try:
            import anthropic
            result["sdk_installed"] = True
        except ImportError:
            pass
        result["api_key_set"] = bool(os.environ.get("ANTHROPIC_API_KEY"))

    elif provider == "openai":
        try:
            import openai
            result["sdk_installed"] = True
        except ImportError:
            pass
        result["api_key_set"] = bool(os.environ.get("OPENAI_API_KEY"))

    elif provider == "gemini":
        try:
            import google.generativeai
            result["sdk_installed"] = True
        except ImportError:
            pass
        result["api_key_set"] = bool(os.environ.get("GOOGLE_API_KEY"))

    elif provider == "ollama":
        # Ollama uses OpenAI SDK
        try:
            import openai
            result["sdk_installed"] = True
        except ImportError:
            pass
        # Check if Ollama server is running
        result["api_key_set"] = True  # Ollama doesn't need API key
        result["server_running"] = check_ollama_running()

    return result


def check_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running and accessible."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method='GET')
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        return False


def get_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Get list of models available in local Ollama installation."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method='GET')
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return [m['name'] for m in data.get('models', [])]
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, Exception):
        return []


# Test/demo code
if __name__ == "__main__":
    print("=" * 60)
    print("AI CLIENT TEST")
    print("=" * 60)

    # Check what's available
    print("\nChecking available providers...")
    provider_info = get_provider_info()
    for provider in get_available_providers():
        status = check_provider_available(provider)
        info = provider_info.get(provider, {})
        sdk = "✓" if status["sdk_installed"] else "✗"

        if provider == "ollama":
            server = "✓ running" if status["server_running"] else "✗ not running"
            print(f"  {provider}: SDK={sdk} Server={server} (FREE! {info.get('download_url', '')})")
            if status["server_running"]:
                models = get_ollama_models()
                if models:
                    print(f"    Available models: {', '.join(models[:5])}")
        else:
            key = "✓" if status["api_key_set"] else "✗"
            print(f"  {provider}: SDK={sdk} API_KEY={key}")

    # Try Ollama first (it's free!)
    ollama_status = check_provider_available("ollama")
    if ollama_status["sdk_installed"] and ollama_status["server_running"]:
        print("\n" + "-" * 60)
        print("Testing Ollama (FREE local LLM)...")
        print("-" * 60)

        try:
            models = get_ollama_models()
            model = models[0] if models else "llama3.2"
            client = AIClient(provider="ollama", model=model)
            response = client.complete(
                "What is 2 + 2? Answer in one word.",
                max_tokens=10
            )
            print(f"Model: {model}")
            print(f"Response: {response.content}")
            print(f"Tokens used: {response.usage}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("\n" + "-" * 60)
        print("Ollama not running.")
        print("To use FREE local AI:")
        print("  1. Download: https://ollama.com/download")
        print("  2. Install and run Ollama")
        print("  3. Run: ollama pull llama3.2")
        print("-" * 60)

    # Try Claude if available
    claude_status = check_provider_available("claude")
    if claude_status["sdk_installed"] and claude_status["api_key_set"]:
        print("\n" + "-" * 60)
        print("Testing Claude API...")
        print("-" * 60)

        try:
            client = AIClient(provider="claude")
            response = client.complete(
                "What is 2 + 2? Answer in one word.",
                max_tokens=10
            )
            print(f"Response: {response.content}")
            print(f"Tokens used: {response.usage}")
        except Exception as e:
            print(f"Error: {e}")
