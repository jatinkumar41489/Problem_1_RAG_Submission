"""Groq chat-completions client with bounded retries and categorised errors.

Retried (exponential backoff + jitter, at most ``max_retries``): rate limit (429), timeouts,
connection errors, 5xx. NOT retried: authentication, permission, model-not-found, bad request.
Token usage is read from the provider response (``usage``); it is never invented.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from app.config import Settings
from app.exceptions import GenerationError
from app.logging_config import get_logger, log_event

log = get_logger("groq")


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    attempts: int
    finish_reason: str | None = None


class GroqClient:
    def __init__(self, settings: Settings, client: Any | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.settings = settings
        self._client = client          # injectable for tests
        self._sleep = sleep

    def _get_client(self):
        if self._client is None:
            key = self.settings.groq_api_key
            if not key:
                raise GenerationError(
                    "GROQ_API_KEY is not set; generation is unavailable (retrieval still works).",
                    category="configuration")
            import groq
            # we do our own bounded retry loop, so disable the SDK's
            self._client = groq.Groq(api_key=key, timeout=self.settings.groq_timeout, max_retries=0)
        return self._client

    @staticmethod
    def _classify(exc: Exception) -> tuple[str, bool]:
        import groq
        if isinstance(exc, groq.AuthenticationError) or isinstance(exc, groq.PermissionDeniedError):
            return "authentication", False
        if isinstance(exc, groq.NotFoundError):
            return "model_not_found", False
        if isinstance(exc, (groq.BadRequestError, groq.UnprocessableEntityError)):
            return "bad_request", False
        if isinstance(exc, groq.RateLimitError):
            return "rate_limit", True
        if isinstance(exc, groq.APITimeoutError):
            return "timeout", True
        if isinstance(exc, groq.APIConnectionError):
            return "connection", True
        if isinstance(exc, groq.InternalServerError):
            return "server_error", True
        if isinstance(exc, groq.APIStatusError):
            return ("server_error", True) if getattr(exc, "status_code", 0) >= 500 else ("bad_request", False)
        return "unknown", False

    def complete(self, messages: list[dict], model: str | None = None, max_tokens: int | None = None,
                 temperature: float = 0.0) -> LLMResponse:
        client = self._get_client()
        model = model or self.settings.groq_model
        max_tokens = max_tokens or self.settings.max_output_tokens
        attempts = 0
        t0 = time.perf_counter()
        while True:
            attempts += 1
            try:
                resp = client.chat.completions.create(model=model, messages=messages,
                                                      max_tokens=max_tokens, temperature=temperature)
                break
            except GenerationError:
                raise
            except Exception as exc:  # categorise SDK errors
                category, retryable = self._classify(exc)
                if retryable and attempts <= self.settings.groq_max_retries:
                    delay = min(8.0, 0.5 * (2 ** (attempts - 1))) + random.uniform(0, 0.25)
                    log_event(log, "groq_retry", level=30, category=category, attempt=attempts, delay_s=round(delay, 2))
                    self._sleep(delay)
                    continue
                hint = {"model_not_found": f" Model {model!r} is not available to this account; set GROQ_MODEL "
                                           "to a model returned by scripts/verify_setup.py --list-models.",
                        "authentication": " Check GROQ_API_KEY."}.get(category, "")
                raise GenerationError(f"Groq request failed ({category}).{hint}", category=category,
                                      retryable=retryable, detail={"attempts": attempts}) from exc
        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            choice = resp.choices[0]
            text = choice.message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise GenerationError("Malformed Groq response (no choices/message).",
                                  category="malformed_response") from exc
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text, model=getattr(resp, "model", model) or model,
            prompt_tokens=getattr(usage, "prompt_tokens", None), completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None), latency_ms=latency_ms, attempts=attempts,
            finish_reason=getattr(choice, "finish_reason", None))
