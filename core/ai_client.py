"""
AI Client - Unified interface for LLM APIs (Claude, GPT, Gemini).

This module provides a simple interface for integrating AI capabilities
into PhysioMetrics. Users bring their own API keys.

Usage:
    from core.ai_client import AIClient

    client = AIClient(provider='claude', api_key='sk-ant-...')
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


@dataclass
class AIResponse:
    """Standardized response from AI models."""
    content: str
    provider: str
    model: str
    usage: Dict  # Token counts
    raw_response: Dict  # Full API response for debugging


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
    }

    def __init__(self, provider: str = "claude", api_key: str = None, model: str = None):
        """
        Initialize AI client.

        Args:
            provider: One of 'claude', 'openai', 'gemini'
            api_key: API key for the provider (or set via environment variable)
            model: Specific model to use (or uses default for provider)
        """
        # Normalize provider
        provider_lower = provider.lower()
        if provider_lower == "claude" or provider_lower == "anthropic":
            self.provider = AIProvider.CLAUDE
        elif provider_lower == "openai" or provider_lower == "gpt":
            self.provider = AIProvider.OPENAI
        elif provider_lower == "gemini" or provider_lower == "google":
            self.provider = AIProvider.GEMINI
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'claude', 'openai', or 'gemini'")

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
    return ["claude", "openai", "gemini"]


def check_provider_available(provider: str) -> Dict[str, bool]:
    """
    Check if a provider's SDK is installed and API key is available.

    Returns:
        Dict with 'sdk_installed' and 'api_key_set' booleans
    """
    result = {"sdk_installed": False, "api_key_set": False}

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

    return result


# Test/demo code
if __name__ == "__main__":
    print("=" * 60)
    print("AI CLIENT TEST")
    print("=" * 60)

    # Check what's available
    print("\nChecking available providers...")
    for provider in get_available_providers():
        status = check_provider_available(provider)
        sdk = "✓" if status["sdk_installed"] else "✗"
        key = "✓" if status["api_key_set"] else "✗"
        print(f"  {provider}: SDK={sdk} API_KEY={key}")

    # Try a simple test if Claude is available
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
    else:
        print("\nClaude not available for testing.")
        print("To test, install anthropic SDK and set ANTHROPIC_API_KEY environment variable.")
