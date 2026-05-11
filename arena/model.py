"""Model adapter."""

from __future__ import annotations

import os

from openai import OpenAI


def call(model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
    if model == "local" or model.startswith("local:"):
        client = OpenAI(
            base_url=os.environ.get("ARENA_BASE_URL", "http://localhost:8000/v1"),
            api_key="local",
        )
        name = "chat" if model == "local" else model.split(":", 1)[1]
    elif model.startswith("openai:"):
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        name = model.split(":", 1)[1]
    elif model.startswith("anthropic:"):
        client = OpenAI(
            base_url="https://api.anthropic.com/v1",
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        name = model.split(":", 1)[1]
    else:
        raise ValueError(f"unknown model spec: {model}")

    resp = client.chat.completions.create(
        model=name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
