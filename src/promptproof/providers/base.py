"""Model provider interface.

The harness only ever needs one operation from a model: given a system prompt
and a conversation, produce the next assistant message. Keeping the interface
that narrow is what lets the offline stub exercise the *real* runner, judge,
and report code paths rather than a parallel test-only implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..conversation import Message


@dataclass(frozen=True, slots=True)
class Completion:
    """One assistant message plus whatever the provider could tell us about it."""

    text: str
    model: str
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    request_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ModelProvider(Protocol):
    """Anything that can turn (system prompt, conversation) into a reply."""

    name: str

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Message],
        max_tokens: int = 4096,
    ) -> Completion:
        """Produce the next assistant message.

        Raises:
            ProviderError: on any failure to obtain a completion.
        """
        ...
