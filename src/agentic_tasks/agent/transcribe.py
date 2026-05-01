"""Transcribe a Telegram voice note via an OpenAI-compatible audio endpoint.

Uses the OpenAI SDK pointed at any OpenAI-compatible provider (Groq's
whisper-large-v3 by default — see ``config.llm_base_url``). Telegram voice
notes are OGG/Opus, which both providers accept natively, so we pass the raw
bytes through without re-encoding.
"""

from __future__ import annotations

import logging
from io import BytesIO
from time import perf_counter
from typing import Any

from openai import OpenAI

from agentic_tasks.config import get_settings

log = logging.getLogger(__name__)


def _audio_client() -> OpenAI:
    s = get_settings()
    kwargs: dict[str, Any] = {"api_key": s.llm_api_key}
    if s.llm_base_url:
        kwargs["base_url"] = s.llm_base_url
    return OpenAI(**kwargs)


def transcribe_audio(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    """Send audio bytes to the configured provider for transcription."""
    s = get_settings()
    client = _audio_client()

    audio = BytesIO(audio_bytes)
    audio.name = filename

    start = perf_counter()
    response: Any = client.audio.transcriptions.create(
        model=s.transcription_model,
        file=audio,
    )
    log.info(
        "transcription (%d bytes): %.0fms",
        len(audio_bytes),
        (perf_counter() - start) * 1000,
    )
    return response.text or ""
