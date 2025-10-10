"""
LLM Provider Implementations

Unified provider interfaces for different LLM services.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, config, provider: str, api_key: Optional[str] = None, **kwargs):
        self.config = config
        self.provider = provider
        self.api_key = api_key or self._get_api_key()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._client = None
        self._initialize_client(**kwargs)

    def _get_api_key(self) -> Optional[str]:
        """Get API key from unified config."""
        import os

        # Try extractor.llm first (most common location)
        try:
            extractor_config = getattr(self.config, 'extractor', {})
            if isinstance(extractor_config, dict):
                llm_config = extractor_config.get('llm', {})
                if isinstance(llm_config, dict):
                    api_key = llm_config.get('openai_api_key') or llm_config.get('api_key')
                    if api_key:
                        return api_key
        except (AttributeError, TypeError):
            pass

        # Try direct llm section
        try:
            llm_config = getattr(self.config, 'llm', {})
            if isinstance(llm_config, dict):
                api_key = llm_config.get('api_key')
                if api_key:
                    return api_key
        except (AttributeError, TypeError):
            pass

        # Environment variable fallbacks
        env_keys = {
            'openai': ['OPENAI_API_KEY'],
            'anthropic': ['ANTHROPIC_API_KEY'],
            'azure_openai': ['AZURE_OPENAI_API_KEY', 'OPENAI_API_KEY']
        }

        for env_key in env_keys.get(self.provider, []):
            value = os.getenv(env_key)
            if value:
                return value

        return None

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value from unified config."""
        # Try extractor.llm first
        try:
            extractor_config = getattr(self.config, 'extractor', {})
            if isinstance(extractor_config, dict):
                llm_config = extractor_config.get('llm', {})
                if isinstance(llm_config, dict) and key in llm_config:
                    return llm_config[key]
        except (AttributeError, TypeError):
            pass

        # Try direct llm section
        try:
            llm_config = getattr(self.config, 'llm', {})
            if isinstance(llm_config, dict) and key in llm_config:
                return llm_config[key]
        except (AttributeError, TypeError):
            pass

        return default

    @abstractmethod
    def _initialize_client(self, **kwargs):
        """Initialize the provider client."""
        pass

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion."""
        pass

    @abstractmethod
    def structured_completion(self, messages: List[Dict[str, str]], response_model: Type[Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Generate structured completion with models parsing."""
        pass

    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """Get list of supported features."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""

    def _initialize_client(self, **kwargs):
        """Initialize OpenAI client."""
        try:
            import openai
            from openai import OpenAI

            if not self.api_key:
                raise ValueError("OpenAI API key is required")

            client_kwargs = {"api_key": self.api_key}
            api_base = self._get_config_value("api_base") or self._get_config_value("base_url")
            if api_base:
                client_kwargs["base_url"] = api_base

            self._client = OpenAI(**client_kwargs)
            self.logger.info("OpenAI client initialized successfully")

        except ImportError:
            self.logger.error("OpenAI package not available. Install with: pip install openai")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate OpenAI chat completion."""
        try:
            # Prepare parameters
            params = {
                "models": kwargs.get("models", self._get_config_value("models", "gpt-4")),
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self._get_config_value("max_tokens", 2000)),
                "temperature": kwargs.get("temperature", self._get_config_value("temperature", 0.3))
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            response = self._client.chat.completions.create(**params)

            return {
                "content": response.choices[0].message.content,
                "models": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                "metadata": {
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id
                }
            }

        except Exception as e:
            self.logger.error(f"OpenAI chat completion failed: {e}")
            raise

    def structured_completion(self, messages: List[Dict[str, str]], response_model: Type[Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Generate structured completion using OpenAI Responses API."""
        try:
            # Try using Responses API if available
            if hasattr(self._client, 'responses'):
                # Convert dataclass to Pydantic if needed
                if hasattr(response_model, 'to_pydantic'):
                    obj = response_model()
                    pydantic_model = obj.to_pydantic().__class__
                else:
                    pydantic_model = response_model

                # Join messages for Responses API
                joined_content = "\n\n".join(m.get("content", "") for m in messages)

                params = {
                    "models": kwargs.get("models", self.config.default_model),
                    "input": [{"role": "user", "content": joined_content}],
                    "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "text_format": pydantic_model,
                    "reasoning": {"effort": kwargs.get("reasoning_effort", "minimal")},
                }

                response = self._client.responses.parse(**params)
                output_parsed = getattr(response, "output_parsed", None)

                if output_parsed is not None:
                    data = output_parsed.to_dict()

                    if data:
                        return {
                            "parsed_data": data,
                            "raw_content": None,
                            "models": kwargs.get("models", self.config.default_model)
                        }

            return None  # Fallback to JSON parsing

        except Exception as e:
            self.logger.debug(f"OpenAI structured completion failed: {e}")
            return None

    def get_supported_features(self) -> List[str]:
        """Get OpenAI supported features."""
        return [
            "chat_completion",
            "structured_completion",
            "function_calling",
            "streaming",
            "embeddings"
        ]


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    def _initialize_client(self, **kwargs):
        """Initialize Anthropic client."""
        try:
            import anthropic

            if not self.api_key:
                raise ValueError("Anthropic API key is required")

            self._client = anthropic.Anthropic(api_key=self.api_key)
            self.logger.info("Anthropic client initialized successfully")

        except ImportError:
            self.logger.error("Anthropic package not available. Install with: pip install anthropic")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
            raise

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate Anthropic chat completion."""
        try:
            # Convert messages format for Anthropic
            anthropic_messages = []
            system_message = None

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            params = {
                "models": kwargs.get("models", self.config.default_model),
                "messages": anthropic_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature)
            }

            if system_message:
                params["system"] = system_message

            response = self._client.messages.create(**params)

            return {
                "content": response.content[0].text,
                "models": response.model,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                "metadata": {
                    "finish_reason": response.stop_reason,
                    "response_id": response.id
                }
            }

        except Exception as e:
            self.logger.error(f"Anthropic chat completion failed: {e}")
            raise

    def structured_completion(self, messages: List[Dict[str, str]], response_model: Type[Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Anthropic doesn't support structured completion natively."""
        return None  # Will fallback to JSON parsing

    def get_supported_features(self) -> List[str]:
        """Get Anthropic supported features."""
        return [
            "chat_completion",
            "streaming"
        ]


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider implementation."""

    def _initialize_client(self, **kwargs):
        """Initialize Azure OpenAI client."""
        try:
            import openai
            from openai import AzureOpenAI

            if not self.api_key:
                raise ValueError("Azure OpenAI API key is required")

            if not self.config.api_base:
                raise ValueError("Azure OpenAI endpoint is required")

            self._client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.config.api_base,
                api_version=self.config.api_version or "2024-02-01"
            )
            self.logger.info("Azure OpenAI client initialized successfully")

        except ImportError:
            self.logger.error("OpenAI package not available. Install with: pip install openai")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate Azure OpenAI chat completion."""
        try:
            params = {
                "models": kwargs.get("models", self.config.default_model),
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature)
            }

            response = self._client.chat.completions.create(**params)

            return {
                "content": response.choices[0].message.content,
                "models": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                "metadata": {
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id
                }
            }

        except Exception as e:
            self.logger.error(f"Azure OpenAI chat completion failed: {e}")
            raise

    def structured_completion(self, messages: List[Dict[str, str]], response_model: Type[Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Azure OpenAI structured completion (similar to OpenAI)."""
        # Use same logic as OpenAI provider
        return OpenAIProvider.structured_completion(self, messages, response_model, **kwargs)

    def get_supported_features(self) -> List[str]:
        """Get Azure OpenAI supported features."""
        return [
            "chat_completion",
            "structured_completion",
            "function_calling",
            "streaming",
            "embeddings"
        ]
