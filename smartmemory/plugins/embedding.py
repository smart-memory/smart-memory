import logging
import numpy as np

from smartmemory.configuration import MemoryConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Abstracts embedding computation for different providers (OpenAI, Ollama, etc).
    """

    def __init__(self, config=None):
        if config is None:
            config = MemoryConfig().vector["embedding"]
        self.provider = config.get('provider', 'openai')
        self.model = config.get('models', 'text-embedding-ada-002')
        self.api_key = config.get("openai_api_key")
        self.ollama_url = config.get('ollama_url', 'http://localhost:11434')
        self.hf_api_key = config.get('huggingface_api_key')
        self.hf_model = config.get('huggingface_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self._hf_tokenizer = None
        self._hf_model_instance = None

    def embed(self, text):
        # Try Redis cache first for significant performance improvement
        try:
            from smartmemory.utils.cache import get_cache
            cache = get_cache()

            # Check cache for existing embedding
            cached_embedding = cache.get_embedding(text)
            if cached_embedding is not None:
                logger.debug(f"Cache hit for embedding: {text[:50]}...")
                return np.array(cached_embedding)

            logger.debug(f"Cache miss for embedding: {text[:50]}...")
        except Exception as e:
            logger.warning(f"Redis cache unavailable for embeddings: {e}")
            cache = None

        # Generate embedding via API
        if self.provider == 'openai':
            if not self.api_key:
                # Generate mock embedding for testing when no API key is available
                logger.warning("No OpenAI API key provided, generating mock embedding for testing")
                # Generate deterministic mock embedding based on text hash
                import hashlib
                text_hash = hashlib.md5(text.encode()).hexdigest()
                # Convert hash to 1536-dimensional vector (OpenAI embedding size)
                mock_embedding = []
                for i in range(1536):
                    # Use hash characters cyclically to generate values between -1 and 1
                    char_val = ord(text_hash[i % len(text_hash)]) / 255.0 * 2 - 1
                    mock_embedding.append(char_val)
                embedding = np.array(mock_embedding)
            else:
                import openai
                openai.api_key = self.api_key
                resp = openai.embeddings.create(input=text, model=self.model)
                embedding = np.array(resp.data[0].embedding)
        elif self.provider == 'ollama':
            import requests
            url = f"{self.ollama_url}/api/embeddings"
            resp = requests.post(url, json={"models": self.model, "prompt": text})
            resp.raise_for_status()
            embedding = np.array(resp.json()['embedding'])
        elif self.provider == 'huggingface':
            embedding = self._embed_huggingface(text)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

        # Cache the result for future use
        if cache is not None:
            try:
                cache.set_embedding(text, embedding.tolist())
                logger.debug(f"Cached embedding for: {text[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")

        return embedding

    def _embed_huggingface(self, text):
        """
        Generate embeddings using HuggingFace models.
        Supports both API-based and local model inference.
        """
        # Try API-based approach first if API key is provided
        if self.hf_api_key:
            try:
                import requests
                api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.hf_model}"
                headers = {"Authorization": f"Bearer {self.hf_api_key}"}
                response = requests.post(api_url, headers=headers, json={"inputs": text})
                response.raise_for_status()
                embedding = np.array(response.json())
                # HuggingFace API returns shape (1, seq_len, hidden_dim), take mean over seq_len
                if len(embedding.shape) == 3:
                    embedding = embedding[0].mean(axis=0)
                elif len(embedding.shape) == 2:
                    embedding = embedding.mean(axis=0)
                return embedding
            except Exception as e:
                logger.warning(f"HuggingFace API failed, falling back to local model: {e}")
        
        # Fall back to local model inference
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Lazy load model and tokenizer (cache for reuse)
            if self._hf_tokenizer is None or self._hf_model_instance is None:
                logger.info(f"Loading HuggingFace model: {self.hf_model}")
                self._hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
                self._hf_model_instance = AutoModel.from_pretrained(self.hf_model)
                self._hf_model_instance.eval()  # Set to evaluation mode
            
            # Tokenize and generate embeddings
            inputs = self._hf_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self._hf_model_instance(**inputs)
                # Use mean pooling over token embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embedding = embeddings[0].cpu().numpy()
            
            return embedding
            
        except ImportError:
            raise ImportError(
                "HuggingFace embeddings require 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate HuggingFace embedding: {e}")


def create_embeddings(text):
    return EmbeddingService().embed(text)
