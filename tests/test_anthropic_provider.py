"""Live provider: response handling and error mapping.

No network is touched. The SDK client is replaced with a fake whose
``messages.create`` returns or raises whatever the test needs, which is enough to
pin down the two things that actually bite in production - how a refusal is
represented, and whether a 404 is distinguishable from a 500.

Skipped entirely when the optional ``[live]`` extra is not installed, so the
default offline suite never depends on the SDK being present.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")
# The SDK moved to httpx2 in 1.0; fall back to httpx for older installs.
try:  # pragma: no cover - depends on the installed SDK major version
    import httpx2 as httpx
except ModuleNotFoundError:  # pragma: no cover
    httpx = pytest.importorskip("httpx")

from promptproof.conversation import Message  # noqa: E402
from promptproof.errors import ProviderError  # noqa: E402
from promptproof.providers.anthropic_provider import AnthropicProvider  # noqa: E402


def _response(status: int = 500) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def _message(
    *,
    text: str = "hello",
    stop_reason: str = "end_turn",
    stop_details=None,
):
    blocks = [SimpleNamespace(type="text", text=text)] if text else []
    return SimpleNamespace(
        content=blocks,
        model="claude-opus-5",
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        _request_id="req_test",
    )


class FakeMessages:
    def __init__(self, result) -> None:
        self.result = result
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _provider(result) -> tuple[AnthropicProvider, FakeMessages]:
    provider = AnthropicProvider(api_key="test-key")
    fake = FakeMessages(result)
    provider._client = SimpleNamespace(messages=fake)
    return provider, fake


def _call(provider: AnthropicProvider):
    return provider.complete(
        model="claude-opus-5",
        system="You are a test assistant.",
        messages=[Message("user", "hi")],
    )


def test_missing_credentials_fail_fast_with_an_actionable_message(monkeypatch):
    """One clear line, not the SDK's TypeError once per case in the matrix."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    try:
        AnthropicProvider()
    except ProviderError as exc:
        assert "no Anthropic credentials found" in str(exc)
    else:
        pytest.skip("this machine has an ambient Anthropic credential profile")


def test_successful_completion_is_unwrapped():
    provider, fake = _provider(_message(text="  hello there  "))
    completion = _call(provider)

    assert completion.text == "hello there"
    assert completion.model == "claude-opus-5"
    assert completion.input_tokens == 11
    assert completion.output_tokens == 7
    assert completion.request_id == "req_test"
    assert fake.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert fake.kwargs["system"] == "You are a test assistant."


def test_sampling_and_thinking_parameters_are_never_sent():
    """Both are model-family dependent and 400 on the current generation.

    This harness is explicitly pointed at cheap models, so letting each model
    use its own defaults is the only shape that works across the matrix.
    """
    provider, fake = _provider(_message())
    _call(provider)
    assert "temperature" not in fake.kwargs
    assert "top_p" not in fake.kwargs
    assert "thinking" not in fake.kwargs


def test_a_safety_refusal_is_a_reply_not_an_error():
    """A model that declines an adversarial probe has passed, not errored."""
    provider, _ = _provider(
        _message(
            text="",
            stop_reason="refusal",
            stop_details=SimpleNamespace(type="refusal", category="cyber", explanation=""),
        )
    )
    completion = _call(provider)
    assert completion.stop_reason == "refusal"
    assert "declined" in completion.text
    assert "cyber" in completion.text


def test_empty_content_is_an_error():
    provider, _ = _provider(_message(text=""))
    with pytest.raises(ProviderError, match="returned no text content"):
        _call(provider)


@pytest.mark.parametrize(
    ("exc", "match", "retryable"),
    [
        (anthropic.NotFoundError("nope", response=_response(404), body=None), "not found", False),
        (
            anthropic.AuthenticationError("bad key", response=_response(401), body=None),
            "authentication failed",
            False,
        ),
        (
            anthropic.PermissionDeniedError("no access", response=_response(403), body=None),
            "may not use",
            False,
        ),
        (
            anthropic.BadRequestError("bad param", response=_response(400), body=None),
            "request rejected",
            False,
        ),
        (
            anthropic.RateLimitError("slow down", response=_response(429), body=None),
            "rate limited",
            True,
        ),
        (
            anthropic.APIStatusError("server broke", response=_response(503), body=None),
            "API error 503",
            True,
        ),
        (
            anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com")),
            "connection failure",
            True,
        ),
        (
            anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com")),
            r"timed out after 120s",
            True,
        ),
    ],
)
def test_errors_are_mapped_specifically(exc, match, retryable):
    provider, _ = _provider(exc)
    with pytest.raises(ProviderError, match=match) as info:
        _call(provider)
    assert info.value.model == "claude-opus-5"
    assert info.value.retryable is retryable


def test_a_timeout_is_reported_separately_from_a_connection_failure():
    """APITimeoutError subclasses APIConnectionError, so ordering is load-bearing."""
    assert issubclass(anthropic.APITimeoutError, anthropic.APIConnectionError)
    provider, _ = _provider(
        anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com"))
    )
    provider.timeout = 12.5
    with pytest.raises(ProviderError, match=r"timed out after 12\.5s") as info:
        _call(provider)
    assert info.value.retryable is True


def test_truncation_before_any_text_names_the_ceiling_to_raise():
    """Thinking-by-default models spend max_tokens before emitting text.

    The generic "no text content" message sends a reader looking for a broken
    model; the real fix is a bigger ceiling, so the error says so.
    """
    provider, _ = _provider(_message(text="", stop_reason="max_tokens"))
    with pytest.raises(ProviderError, match="hit the 1024-token ceiling") as info:
        provider.complete(
            model="claude-opus-5",
            system="s",
            messages=[Message("user", "hi")],
            max_tokens=1024,
        )
    assert "--max-tokens" in str(info.value)
    assert info.value.request_id == "req_test"


def test_a_response_with_no_content_at_all_is_still_an_error():
    provider, _ = _provider(_message(text="", stop_reason="end_turn"))
    with pytest.raises(ProviderError, match="returned no text content"):
        _call(provider)


def test_thinking_blocks_are_skipped_when_unwrapping_text():
    """Responses may carry non-text blocks; only text blocks are the reply."""
    response = _message(text="the answer")
    response.content = [
        SimpleNamespace(type="thinking", thinking="..."),
        *response.content,
    ]
    provider, _ = _provider(response)
    assert _call(provider).text == "the answer"


def test_timeout_and_retry_settings_reach_the_sdk_client():
    provider = AnthropicProvider(api_key="test-key", timeout=7.5, max_retries=1)
    assert provider.timeout == 7.5
    assert provider.max_retries == 1
    assert provider._client.timeout == 7.5
    assert provider._client.max_retries == 1


def test_federated_credentials_are_not_mistaken_for_no_credentials(monkeypatch):
    """WIF resolves at request time and leaves nothing on the client to check."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
        "ANTHROPIC_SERVICE_ACCOUNT_ID",
    ):
        monkeypatch.setenv(name, "set-by-the-workload-identity-provider")
    provider = AnthropicProvider()
    assert provider._client.api_key is None
