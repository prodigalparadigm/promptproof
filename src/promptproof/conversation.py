"""Shared conversation primitives.

Lives in its own module so :mod:`promptproof.cases` and :mod:`promptproof.drift`
can both use ``Turn`` without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Turn:
    """One scripted user message in a synthetic case.

    Attributes:
        text: What the synthetic user says.
        intent: Why this turn exists (``rapport``, ``probe``, ``reframe``,
            ``premise``, ``violation``). Rendered in the report so a reader can
            see the shape of the attack without reading every word.
        escalation: 0.0 (harmless) through 1.0 (the actual violating ask).
    """

    text: str
    intent: str = "probe"
    escalation: float = 1.0


@dataclass(frozen=True, slots=True)
class Message:
    """A message exchanged with a model provider."""

    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}
