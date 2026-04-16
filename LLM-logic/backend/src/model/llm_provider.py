import os
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Tuple[str, str, str]:
        """
        Generate a response from the LLM given a prompt
        
        Returns:
            Tuple of (response_text, actual_provider, actual_model)
        """
        pass
    

class OpenAIProvider(LLMProvider):
    def __init__(self, model="gpt-4o-mini"):
        import openai
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content, "openai", self.model


class ClaudeProvider(LLMProvider):
    def __init__(self, model="claude-3-opus-20240229"):
        import anthropic
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        # Claude requires system prompt as a top-level parameter, not in messages
        messages = [{"role": "user", "content": prompt}]
        
        request_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2000)
        }
        
        if system_prompt:
            request_params["system"] = system_prompt
        
        response = self.client.messages.create(**request_params)
        return response.content[0].text, "claude", self.model


class GeminiProvider(LLMProvider):
    def __init__(self, model="gemini-1.5-pro-latest"):
        import google.generativeai as genai
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        # Add "models/" prefix if it doesn't already have it
        if not model.startswith("models/"):
            self.model = f"models/{model}"
        else:
            self.model = model
        
        # Replace deprecated models
        if self.model == "models/gemini-pro":
            self.model = "models/gemini-1.5-pro-latest"
        
        genai.configure(api_key=self.api_key)
        self.genai = genai
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        try:
            model = self.genai.GenerativeModel(self.model)
            
            # Gemini doesn't support system prompts directly
            if system_prompt:
                prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            
            response = model.generate_content(prompt)
            return response.text, "gemini", self.model
        except Exception as e:
            error_str = str(e).lower()
            # Handle quota exceeded errors
            if "quota" in error_str or "429" in error_str:
                print(f"[WARNING] Gemini quota exceeded, falling back to OpenAI")
                
                # Create an OpenAI provider as fallback
                openai_provider = OpenAIProvider(model="gpt-3.5-turbo")
                response, _, _ = openai_provider.generate(prompt, system_prompt, **kwargs)
                
                # Return the response but indicate it came from a fallback
                return response, "openai (fallback from gemini)", "gpt-3.5-turbo"
            else:
                # Re-raise other errors
                raise


def get_llm_provider(provider_name: str, model: Optional[str] = None) -> LLMProvider:
    """Factory function to get the appropriate LLM provider"""
    if provider_name.lower() == "openai":
        return OpenAIProvider(model=model if model else "gpt-4o-mini")
    elif provider_name.lower() == "claude":
        return ClaudeProvider(model=model if model else "claude-3-opus-20240229")
    elif provider_name.lower() == "gemini":
        return GeminiProvider(model=model if model else "gemini-1.5-pro-latest")
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


# Default configuration for providers and models
DEFAULT_PROVIDERS = {
    "openai": {
        "default_model": "gpt-4o-mini",
        "available_models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    },
    "claude": {
        "default_model": "claude-3-opus-20240229",
        "available_models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    },
    "gemini": {
        "default_model": "gemini-1.5-pro-latest",
        "available_models": ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-1.5-flash-001"]
    }
}

def get_available_providers():
    """Return information about available providers and models."""
    return DEFAULT_PROVIDERS