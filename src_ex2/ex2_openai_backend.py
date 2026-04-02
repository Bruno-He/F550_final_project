from __future__ import annotations

import os
from typing import Any, Dict, List

from openai import OpenAI


class OpenAIResponsesBackend:
    """
    Minimal OpenAI Responses API backend for Exercise 2.
    """

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.2,
        max_output_tokens: int = 400,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _render_messages(messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = (m.get("role") or "user").upper()
            content = m.get("content") or ""
            parts.append(f"{role}:\n{content}")
        return "\n\n".join(parts)

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        prompt = self._render_messages(messages)

        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        text = getattr(resp, "output_text", None)
        if text is None:
            text = str(resp)

        return {"content": text.strip()}