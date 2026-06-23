"""LLM client for code understanding tasks"""
import requests
from typing import Optional
from backend.cache import Cache


class LLMClient:
    """Client for Ollama LLM API (self-hosted, offline).
    
    Tradeoff: Ollama allows offline inference with quantized models but
    lower quality than GPT-4. We accept quality loss for privacy/cost/latency.
    Easy to swap for OpenAI API by changing endpoint.
    """

    def __init__(
        self,
        model: str = "codellama",
        api_endpoint: str = "http://localhost:11434/api/generate",
        temperature: float = 0.2
    ):
        """Initialize LLM client.
        
        Args:
            model: Model name (e.g., 'codellama', 'mistral')
            api_endpoint: Ollama API endpoint
            temperature: Sampling temperature (0=deterministic, 1=random)
        """
        self.model = model
        self.api_endpoint = api_endpoint
        self.temperature = temperature
        self.cache = Cache()

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum response length
            
        Returns:
            Generated text
        """
        # Check cache
        cache_key = f"{self.model}:{prompt[:100]}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "num_predict": max_tokens
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json().get("response", "")

            # Cache result
            self.cache.set(cache_key, result)
            return result

        except requests.exceptions.ConnectionError:
            return "[Error: Could not connect to Ollama. Ensure 'ollama serve' is running]"
        except Exception as e:
            return f"[Error: {str(e)}]"

    def is_available(self) -> bool:
        """Check if LLM service is available."""
        try:
            requests.get(
                self.api_endpoint.replace("/api/generate", "/api/tags"),
                timeout=2
            )
            return True
        except:
            return False
