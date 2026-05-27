import os
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class Config:
    # LLM Provider - easy switch
    llm_provider: Literal["anthropic", "llama"] = "anthropic"

    # Anthropic models
    deep_model: str = "claude-opus-4-7"                   # portfolio manager, researchers
    fast_model: str = "claude-haiku-4-5-20251001"         # analysts

    # Llama API (OpenAI-compatible endpoint)
    llama_model: str = "Llama-4-Maverick-17B-128E-Instruct-FP8"
    llama_base_url: str = "https://api.llama.com/compat/v1/"

    # Analysis depth
    debate_rounds: int = 2   # bull/bear go back and forth this many times
    output_language: str = "German"

    # API Keys (read from env)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    llama_api_key: str = field(default_factory=lambda: os.getenv("LLAMA_API_KEY", ""))
    finnhub_api_key: str = field(default_factory=lambda: os.getenv("FINNHUB_API_KEY", ""))
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))
