"""
Unified LLM client supporting Anthropic (Claude) and Llama API (OpenAI-compatible).
"""

import json
import re
from typing import Any

from portfolio_analyzer.config import Config


class LLMClient:
    """
    Unified LLM client that abstracts over Anthropic and Llama providers.

    Methods:
        complete(system, user, deep=False) -> str
            Regular text completion. deep=True selects the more capable model.

        structured(system, user, schema_dict) -> dict
            Returns structured JSON output matching the given JSON schema.
    """

    def __init__(self, config: Config):
        self.config = config
        self._anthropic_client = None
        self._openai_client = None

        if config.llm_provider == "anthropic":
            self._init_anthropic()
        elif config.llm_provider == "llama":
            self._init_llama()
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm_provider!r}. Use 'anthropic' or 'llama'.")

    # ------------------------------------------------------------------
    # Initializers
    # ------------------------------------------------------------------

    def _init_anthropic(self) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic package not installed. Run: pip install anthropic") from e

        if not self.config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Export it or add it to your .env file.")

        self._anthropic_client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)

    def _init_llama(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e

        if not self.config.llama_api_key:
            raise ValueError("LLAMA_API_KEY is not set. Export it or add it to your .env file.")

        from openai import OpenAI
        self._openai_client = OpenAI(
            api_key=self.config.llama_api_key,
            base_url=self.config.llama_base_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, system: str, user: str, deep: bool = False) -> str:
        """
        Run a text completion and return the assistant's response as a string.

        Args:
            system: System prompt text.
            user:   User message text.
            deep:   If True, use the more powerful model (e.g. claude-opus-4-7 / Llama Maverick).
                    If False, use the faster/cheaper model.

        Returns:
            The model's text response.
        """
        if self.config.llm_provider == "anthropic":
            return self._anthropic_complete(system, user, deep=deep)
        else:
            return self._llama_complete(system, user, deep=deep)

    def structured(self, system: str, user: str, schema_dict: dict) -> dict:
        """
        Run a completion and return a structured dict that matches schema_dict.

        For Anthropic: uses tool_use / tool_choice={"type":"tool","name":"output"}.
        For Llama:     uses response_format={"type":"json_object"} and appends
                       the schema to the system prompt.

        Args:
            system:     System prompt.
            user:       User message.
            schema_dict: JSON Schema dict describing the expected output.

        Returns:
            Parsed dict with keys defined by schema_dict.
        """
        if self.config.llm_provider == "anthropic":
            return self._anthropic_structured(system, user, schema_dict)
        else:
            return self._llama_structured(system, user, schema_dict)

    # ------------------------------------------------------------------
    # Anthropic implementations
    # ------------------------------------------------------------------

    def _pick_anthropic_model(self, deep: bool) -> str:
        return self.config.deep_model if deep else self.config.fast_model

    def _anthropic_complete(self, system: str, user: str, deep: bool = False) -> str:
        model = self._pick_anthropic_model(deep)
        response = self._anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _anthropic_structured(self, system: str, user: str, schema_dict: dict) -> dict:
        """
        Use Anthropic tool_use with forced tool_choice to get structured output.
        The model MUST call the "output" tool, so its input is always our schema.
        """
        tool_def = {
            "name": "output",
            "description": (
                "Return your structured analysis result. "
                "Always call this tool with the required fields."
            ),
            "input_schema": schema_dict,
        }

        response = self._anthropic_client.messages.create(
            model=self._pick_anthropic_model(deep=True),
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "output"},
        )

        # Find the tool_use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "output":
                return block.input  # already a dict

        # Fallback: try to parse text response as JSON
        for block in response.content:
            if hasattr(block, "text"):
                return self._parse_json_fallback(block.text)

        raise RuntimeError("Anthropic structured call returned no tool_use block and no text.")

    # ------------------------------------------------------------------
    # Llama / OpenAI-compatible implementations
    # ------------------------------------------------------------------

    def _pick_llama_model(self, deep: bool) -> str:
        # With Llama API we currently use one model for both tiers.
        return self.config.llama_model

    def _llama_complete(self, system: str, user: str, deep: bool = False) -> str:
        model = self._pick_llama_model(deep)
        response = self._openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content

    def _llama_structured(self, system: str, user: str, schema_dict: dict) -> dict:
        """
        Use OpenAI-compatible json_object mode for structured output.
        We append the schema to the system prompt so the model knows the shape.
        """
        schema_instructions = (
            "\n\nYou MUST respond with a valid JSON object that strictly conforms to "
            "the following JSON Schema. Output ONLY the JSON object, no markdown, "
            "no explanation:\n"
            + json.dumps(schema_dict, indent=2)
        )
        augmented_system = system + schema_instructions

        model = self._pick_llama_model(deep=True)
        response = self._openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user},
            ],
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return self._parse_json_fallback(raw)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_fallback(text: str) -> dict:
        """
        Attempt to extract a JSON object from raw text.
        Handles markdown code fences (```json ... ```) and bare JSON.
        """
        # Strip markdown fences
        stripped = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if fence_match:
            stripped = fence_match.group(1)

        # Try direct parse
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Try to find the first {...} block
        brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response:\n{text[:500]}")
