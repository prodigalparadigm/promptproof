"""Exception hierarchy for promptproof.

Every failure mode the harness can hit is one of these, so a caller can
distinguish "your config is wrong" from "the model provider fell over" from
"the judge said something we could not parse".
"""

from __future__ import annotations


class PromptProofError(Exception):
    """Base class for every error raised by this package."""


class SpecError(PromptProofError):
    """The behavior spec is missing, malformed, or internally inconsistent."""


class ProviderError(PromptProofError):
    """A model provider failed to produce a completion.

    Carries enough context to attribute the failure to a specific model and,
    where the provider supplied one, an upstream request id.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.retryable = retryable
        self.request_id = request_id


class JudgeError(PromptProofError):
    """The judge returned something that could not be interpreted as a verdict."""
