"""
LLM provider abstraction layer supporting multiple providers.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    tokens_used: int = 0
    finish_reason: str = "stop"


class LLMProvider:
    """
    Unified interface for different LLM providers.
    
    Supports OpenAI, Anthropic, Azure OpenAI, and local models.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        provider: Optional[str] = None
    ):
        """
        Initialize LLM provider.
        
        Args:
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            provider: Force specific provider (openai, anthropic, azure)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Auto-detect provider if not specified
        if provider is None:
            provider = self._detect_provider(model)
        
        self.provider = provider
        self.client = self._initialize_client()
        
        logger.info(f"Initialized LLM provider: {provider} with model {model}")
    
    def _detect_provider(self, model: str) -> str:
        """Detect provider from model name."""
        if "gpt" in model.lower():
            return "openai"
        elif "claude" in model.lower():
            return "anthropic"
        else:
            return "openai"  # default
    
    def _initialize_client(self):
        """Initialize the appropriate API client."""
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """
        Generate a completion.
        
        Args:
            system_prompt: System instructions
            user_prompt: User message
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text
        """
        if self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt, **kwargs)
        elif self.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """Generate using OpenAI API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """Generate using Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        
        return response.content[0].text
    
    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ):
        """Generate with streaming response."""
        if self.provider == "openai":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
