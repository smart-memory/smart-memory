"""
Unified LLM Client

Consolidates all LLM client implementations across SmartMemory into a single,
maintainable client supporting multiple providers with consistent interfaces.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Type

from smartmemory.configuration import MemoryConfig
from .providers import OpenAIProvider, AnthropicProvider, AzureOpenAIProvider
from .response_parser import ResponseParser, StructuredResponse

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Unified response format from LLM providers."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class LLMClient:
    """
    Unified LLM client supporting multiple providers with consistent interfaces.
    
    Consolidates functionality from:
    - smartmemory.ontology.llm_manager.LLMOntologyManager._get_default_llm_client()
    - smartmemory.utils.llm.call_llm()
    - Various scattered OpenAI client initializations
    """

    def __init__(self,
                 provider: str = "openai",
                 config: Optional[MemoryConfig] = None,
                 api_key: Optional[str] = None,
                 **provider_kwargs):
        """
        Initialize unified LLM client.
        
        Args:
            provider: Provider name ("openai", "anthropic", "azure_openai")
            config: Configuration object (defaults to global config)
            api_key: Override API key
            **provider_kwargs: Provider-specific configuration
        """
        self.provider_name = provider
        self.config = config or MemoryConfig()
        self.response_parser = ResponseParser()

        # Initialize provider using unified config directly
        self.provider = self._create_provider(api_key, **provider_kwargs)

        logger.info(f"Initialized LLM client with provider: {provider}")

    def _create_provider(self, api_key: Optional[str], **kwargs):
        """Create provider instance based on configuration."""
        provider_classes = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "azure_openai": AzureOpenAIProvider
        }

        provider_class = provider_classes.get(self.provider_name)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {self.provider_name}")

        return provider_class(
            config=self.config,
            provider=self.provider_name,
            api_key=api_key,
            **kwargs
        )

    def chat_completion(self,
                        messages: List[Dict[str, str]],
                        model: Optional[str] = None,
                        temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None,
                        **kwargs) -> LLMResponse:
        """
        Generate chat completion using the configured provider.
        
        Consolidates functionality from:
        - LLMOntologyManager._call_llm()
        - utils.llm.call_llm() fallback path
        """
        try:
            response = self.provider.chat_completion(
                messages=messages,
                model=model or self._get_default_model(),
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            return LLMResponse(
                content=response["content"],
                model=response["models"],
                provider=self.provider_name,
                usage=response.get("usage"),
                metadata=response.get("metadata")
            )

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise

    def structured_completion(self,
                              messages: List[Dict[str, str]],
                              response_model: Type[Any],
                              model: Optional[str] = None,
                              **kwargs) -> StructuredResponse:
        """
        Generate structured completion with Pydantic models parsing.
        
        Consolidates functionality from:
        - utils.llm.call_llm() with response_model
        - utils.llm.run_ontology_llm()
        """
        try:
            # Try structured parsing first
            structured_response = self.provider.structured_completion(
                messages=messages,
                response_model=response_model,
                model=model or self._get_default_model(),
                **kwargs
            )

            if structured_response:
                return StructuredResponse(
                    parsed_data=structured_response["parsed_data"],
                    raw_content=structured_response.get("raw_content"),
                    model=structured_response["models"],
                    provider=self.provider_name,
                    success=True
                )

        except Exception as e:
            logger.warning(f"Structured completion failed, trying fallback: {e}")

        # Fallback to JSON parsing
        return self._structured_fallback(messages, response_model, model, **kwargs)

    def _structured_fallback(self,
                             messages: List[Dict[str, str]],
                             response_model: Type[Any],
                             model: Optional[str],
                             **kwargs) -> StructuredResponse:
        """Fallback to JSON parsing when structured completion fails."""
        # Add JSON instruction to messages
        json_instruction = (
            "Return ONLY a valid JSON object that matches the required schema. "
            "Do not include markdown fences or commentary."
        )

        fallback_messages = messages + [
            {"role": "system", "content": json_instruction}
        ]

        response = self.chat_completion(
            messages=fallback_messages,
            model=model,
            **kwargs
        )

        # Parse JSON response
        parsed_data = self.response_parser.parse_json_response(
            response.content, response_model
        )

        return StructuredResponse(
            parsed_data=parsed_data,
            raw_content=response.content,
            model=response.model,
            provider=self.provider_name,
            success=parsed_data is not None
        )

    def simple_completion(self,
                          prompt: str,
                          model: Optional[str] = None,
                          **kwargs) -> LLMResponse:
        """
        Simple completion for single prompt.
        
        Convenience method for basic LLM calls.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, model, **kwargs)

    def ontology_completion(self,
                            user_content: str,
                            response_model: Type[Any],
                            system_prompt: Optional[str] = None,
                            **kwargs) -> StructuredResponse:
        """
        Ontology-specific completion method.
        
        Consolidates functionality from:
        - utils.llm.run_ontology_llm()
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        return self.structured_completion(
            messages=messages,
            response_model=response_model,
            **kwargs
        )

    def _get_default_model(self) -> str:
        """Get default models for the provider from unified config."""
        # Try extractor.llm.models first (most common location)
        extractor_config = getattr(self.config, 'extractor', {})
        if isinstance(extractor_config, dict):
            llm_config = extractor_config.get('llm', {})
            if llm_config and llm_config.get('models'):
                return llm_config['models']

        # Try direct llm.models
        llm_config = getattr(self.config, 'llm', {})
        if llm_config and llm_config.get('models'):
            return llm_config['models']

        # Provider defaults
        defaults = {
            'openai': 'gpt-4',
            'anthropic': 'claude-3-sonnet-20240229',
            'azure_openai': 'gpt-4'
        }

        return defaults.get(self.provider_name, 'gpt-4')

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        return {
            "provider": self.provider_name,
            "default_model": self._get_default_model(),
            "supported_features": self.provider.get_supported_features(),
            "configuration": {
                "provider": self.provider_name,
                "default_model": self._get_default_model()
            }
        }

    def validate_connection(self) -> bool:
        """Validate connection to the LLM provider."""
        try:
            test_response = self.simple_completion(
                "Hello",
                max_tokens=10
            )
            return test_response.content is not None
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False


# Backward compatibility functions
def get_default_llm_client(provider: str = "openai") -> LLMClient:
    """
    Get default LLM client - backward compatibility for LLMOntologyManager.
    """
    return LLMClient(provider=provider)


def call_llm(*,
             model: str,
             messages: Optional[List[Dict[str, str]]] = None,
             system_prompt: Optional[str] = None,
             user_content: Optional[str] = None,
             response_model: Optional[Type[Any]] = None,
             **kwargs):
    """
    Backward compatibility function for utils.llm.call_llm().
    """
    client = LLMClient()

    # Build messages
    if not messages:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_content:
            messages.append({"role": "user", "content": user_content})

    if response_model:
        response = client.structured_completion(
            messages=messages,
            response_model=response_model,
            model=model,
            **kwargs
        )
        return response.parsed_data, response.raw_content
    else:
        response = client.chat_completion(
            messages=messages,
            model=model,
            **kwargs
        )
        return None, response.content


def run_ontology_llm(*,
                     model: str,
                     user_content: str,
                     response_model: Type[Any],
                     system_prompt: Optional[str] = None,
                     **kwargs):
    """
    Backward compatibility function for utils.llm.run_ontology_llm().
    """
    client = LLMClient()
    response = client.ontology_completion(
        user_content=user_content,
        response_model=response_model,
        system_prompt=system_prompt,
        model=model,
        **kwargs
    )
    return response.parsed_data, response.raw_content
