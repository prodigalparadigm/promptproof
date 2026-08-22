"""Live provider backed by the Anthropic Messages API.

Imported lazily and installed via the optional ``[live]`` extra, so the package
- and its entire test suite - works with the SDK absent.

Notes on the request shape, which the SDK's own behaviour makes easy to get
wrong:

* ``temperature`` is deliberately never sent. It has been removed on the current
  model generation and passing it returns a 400. Reproducibility here comes from
  fixed cases and a fixed judge rubric, not from sampling parameters.
* ``thinking`` is deliberately never sent either. The right value differs by
  model family, and this harness is explicitly meant to be pointed at cheap
  models. Sending a frontier-shaped ``thinking`` block to an older model is a
  400; letting each model use its own default is correct and keeps the
  cross-model comparison honest.
* Because each model keeps its own default, models that think by default spend
  ``max_tokens`` on thinking before they emit any text. A ceiling that is
  comfortable for the reply alone can therefore return a truncated, textless
  response; that case is detected and reported as an actionable error rather
  than as an empty reply the judge would happily grade.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any, cast

from ..conversation import Message
from ..errors import ProviderError
from .base import Completion

DEFAULT_MAX_TOKENS = 4096

#: Set by Workload Identity Federation. The SDK resolves these itself at request
#: time and exposes nothing on the client, so the fail-fast credential check has
#: to look for them directly or it would reject a correctly configured machine.
_FEDERATION_ENV_VARS = (
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
)


class AnthropicProvider:
    """Calls the Anthropic Messages API.

    Args:
        api_key: Optional explicit key. When omitted the SDK resolves
            credentials itself (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``,
            or a stored CLI profile), so a machine already logged in needs no
            environment variable.
        timeout: Per-request timeout in seconds.
        max_retries: Passed to the SDK, which retries 408/409/429/5xx and
            connection errors with backoff.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        try:
            import anthropic  # noqa: PLC0415 - optional dependency
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise ProviderError(
                "the anthropic SDK is not installed; run "
                "pip install 'promptproof[live]' or use --provider stub",
                model="<none>",
            ) from exc

        self._anthropic = anthropic
        self.timeout = timeout
        self.max_retries = max_retries
        kwargs: dict[str, object] = {"timeout": timeout, "max_retries": max_retries}
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            kwargs["api_key"] = key
        self._client = anthropic.Anthropic(**kwargs)  # type: ignore[arg-type]
        self._assert_credentials()

    def _assert_credentials(self) -> None:
        """Fail fast, and legibly, when no credential can be resolved.

        The SDK defers this to request time and raises a bare ``TypeError`` from
        deep inside header construction. Surfacing it here means one clear line
        instead of every case in the matrix erroring identically.

        Deliberately biased toward letting the request through: an API key, an
        auth token, and a resolved profile all show up on the client, but
        federated credentials are resolved later and show up nowhere, so their
        environment variables are checked directly. A false "no credentials"
        that blocks a correctly configured machine is a worse failure than one
        clear 401 from the API.
        """
        client = self._client
        if any(getattr(client, attr, None) for attr in ("api_key", "auth_token", "credentials")):
            return
        if all(os.environ.get(name) for name in _FEDERATION_ENV_VARS):
            return
        raise ProviderError(
            "no Anthropic credentials found. Set ANTHROPIC_API_KEY, run `ant auth login`, "
            "or use --provider stub for a fully offline run",
            model="<none>",
        )

    @staticmethod
    def _reply_text(response: object, *, model: str, max_tokens: int) -> str:
        """Pull the assistant text out of one response, or say why there is none.

        Args:
            response: The SDK's message object.
            model: Model id, for error attribution.
            max_tokens: The ceiling that was requested, so a truncation error can
                name the number the caller has to raise.

        Raises:
            ProviderError: if the response carries no usable text.
        """
        stop_reason = getattr(response, "stop_reason", None)
        request_id = getattr(response, "_request_id", None)

        if stop_reason == "refusal":
            # A safety refusal is a real outcome, not an error. It is recorded as
            # the assistant's reply so the judge can grade it - a model that
            # refuses an adversarial probe has passed, not errored.
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return f"[model declined to respond; category={category}]"

        blocks = getattr(response, "content", None) or ()
        text = "".join(b.text for b in blocks if getattr(b, "type", None) == "text").strip()
        if text:
            return text

        if stop_reason == "max_tokens":
            raise ProviderError(
                f"{model} hit the {max_tokens}-token ceiling before emitting any text. "
                "Models that think by default spend this budget on thinking first - "
                "raise --max-tokens",
                model=model,
                request_id=request_id,
            )
        raise ProviderError(
            f"{model} returned no text content (stop_reason={stop_reason})",
            model=model,
            request_id=request_id,
        )

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Message],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        """Send one request and return the assistant text.

        Raises:
            ProviderError: for every SDK failure, tagged ``retryable`` where a
                later attempt could plausibly succeed. Errors are mapped from
                most specific to least so a 404 (bad model id) is never reported
                as a generic API error.
        """
        anthropic = self._anthropic
        started = time.monotonic()
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                # The SDK wants ``list[MessageParam]``, a TypedDict whose ``role``
                # is a Literal. ``as_dict`` returns the same shape but widens
                # ``role`` to ``str``; ``Message.role`` is already declared as that
                # Literal, so the cast restores information the dict lost rather
                # than asserting anything new.
                messages=cast("Any", [m.as_dict() for m in messages]),
            )
        except anthropic.NotFoundError as exc:
            raise ProviderError(
                f"model {model!r} not found or not available to this account", model=model
            ) from exc
        except anthropic.AuthenticationError as exc:
            raise ProviderError("authentication failed - check ANTHROPIC_API_KEY", model=model) from exc
        except anthropic.PermissionDeniedError as exc:
            raise ProviderError(f"this key may not use {model!r}", model=model) from exc
        except anthropic.BadRequestError as exc:
            raise ProviderError(f"request rejected: {exc}", model=model) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError("rate limited after SDK retries", model=model, retryable=True) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"API error {exc.status_code}: {exc}",
                model=model,
                retryable=exc.status_code >= 500,
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderError(
                f"request timed out after {self.timeout:g}s", model=model, retryable=True
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"connection failure: {exc}", model=model, retryable=True) from exc
        except TypeError as exc:  # the SDK's credential resolution failure path
            raise ProviderError(f"request could not be built: {exc}", model=model) from exc

        elapsed_ms = int((time.monotonic() - started) * 1000)
        text = self._reply_text(response, model=model, max_tokens=max_tokens)

        return Completion(
            text=text,
            model=response.model,
            stop_reason=response.stop_reason or "end_turn",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=elapsed_ms,
            request_id=getattr(response, "_request_id", None),
        )
