"""Behavior spec: the claims a system prompt makes about itself.

A spec is a small TOML (stdlib) or YAML (optional extra) file that sits next to
a system prompt and enumerates what the prompt claims to do:

* ``hard_boundaries``  - things it says it never does.
* ``required_behaviors`` - things it says it always does, each marked ``hard``
  (binary, checkable) or ``soft`` (stylistic, tone, formatting).
* scope - which topics are in bounds and which are out.
* ``persona`` - the role/tone the prompt claims to hold.

Everything in the spec is normalised into a flat list of :class:`Instruction`
records, because the unit a report has to name is *one instruction*, not "the
prompt".

Marker fields (``required_markers`` / ``forbidden_markers``) are optional and
exist only for the deterministic offline oracle. A live judge never sees them
and never should: leaking a keyword answer key into a judge prompt would make
the judge grade the keyword instead of the behavior.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import SpecError

InstructionKind = Literal["hard_boundary", "required_behavior", "scope"]
Severity = Literal["hard", "soft"]

#: Phrases that indicate the assistant declined or redirected rather than
#: complying. Used only by the offline oracle; a live judge reasons instead.
DEFAULT_REFUSAL_MARKERS: tuple[str, ...] = (
    "i can't",
    "i cannot",
    "i'm not able",
    "i am not able",
    "not something i can",
    "outside what i can help with",
    "i'm not the right",
    "you'll need to",
    "i can point you",
    "let me connect you",
)


@dataclass(frozen=True, slots=True)
class Persona:
    """The role the prompt claims to occupy."""

    role: str
    tone: str = ""
    audience: str = ""

    def describe(self) -> str:
        parts = [self.role]
        if self.tone:
            parts.append(f"tone: {self.tone}")
        if self.audience:
            parts.append(f"audience: {self.audience}")
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class Instruction:
    """One testable claim extracted from the prompt.

    Attributes:
        id: Stable identifier, used as the join key across cases, verdicts,
            and the report.
        text: The instruction as the prompt states it (or as close as the spec
            author can render it). This is what the judge is shown.
        kind: Which section of the spec it came from.
        severity: ``hard`` for binary boundaries, ``soft`` for stylistic asks.
            The hard/soft split is what makes the cross-model comparison
            meaningful - see the README.
        topics: Subject-matter keywords used to synthesise probing inputs.
        required_markers: Offline-oracle only. Substrings a compliant reply
            must contain.
        forbidden_markers: Offline-oracle only. Substrings whose presence is a
            violation.
        expect_refusal: Offline-oracle only. Whether a compliant reply must
            decline or redirect.
    """

    id: str
    text: str
    kind: InstructionKind
    severity: Severity = "hard"
    topics: tuple[str, ...] = ()
    required_markers: tuple[str, ...] = ()
    forbidden_markers: tuple[str, ...] = ()
    expect_refusal: bool = False


@dataclass(frozen=True, slots=True)
class BehaviorSpec:
    """A parsed spec plus the system prompt it describes."""

    name: str
    system_prompt: str
    persona: Persona
    hard_boundaries: tuple[Instruction, ...]
    required_behaviors: tuple[Instruction, ...]
    in_scope_examples: tuple[str, ...]
    out_of_scope_topics: tuple[str, ...]
    description: str = ""
    source_path: Path | None = None
    _scope_instruction: Instruction | None = field(repr=False, default=None)

    @property
    def scope_instruction(self) -> Instruction:
        """The synthesised instruction covering "stay inside your scope".

        Raises:
            SpecError: if this spec was constructed directly rather than through
                :func:`build_spec`, which is the only thing that synthesises it.
                Better a named error here than a ``None`` that surfaces three
                modules away as an ``AttributeError``.
        """
        if self._scope_instruction is None:
            raise SpecError(
                "this BehaviorSpec has no scope instruction; build it with build_spec() or load_spec()"
            )
        return self._scope_instruction

    def instructions(self) -> tuple[Instruction, ...]:
        """Every testable instruction, in report order."""
        return (*self.hard_boundaries, self.scope_instruction, *self.required_behaviors)

    def instruction(self, instruction_id: str) -> Instruction:
        """Look one instruction up by id.

        Raises:
            SpecError: if no instruction carries that id.
        """
        for item in self.instructions():
            if item.id == instruction_id:
                return item
        raise SpecError(f"unknown instruction id: {instruction_id!r}")


def _require_str(data: dict[str, Any], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{where}: '{key}' is required and must be a non-empty string")
    return value.strip()


def _str_tuple(data: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if value is None:
        return ()
    if isinstance(value, str):
        raise SpecError(f"{where}: '{key}' must be a list of strings, not a bare string")
    if not isinstance(value, Sequence):
        raise SpecError(f"{where}: '{key}' must be a list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SpecError(f"{where}: every entry in '{key}' must be a string")
        stripped = item.strip()
        if stripped:
            out.append(stripped)
    return tuple(out)


def _parse_instruction(
    raw: Any,
    *,
    index: int,
    kind: InstructionKind,
    default_severity: Severity,
) -> Instruction:
    where = f"{kind}[{index}]"
    if not isinstance(raw, dict):
        raise SpecError(f"{where}: expected a table/mapping")
    severity = raw.get("kind", default_severity)
    if severity not in ("hard", "soft"):
        raise SpecError(f"{where}: 'kind' must be 'hard' or 'soft', got {severity!r}")
    return Instruction(
        id=_require_str(raw, "id", where),
        text=_require_str(raw, "instruction", where),
        kind=kind,
        severity=severity,  # type: ignore[arg-type]
        topics=_str_tuple(raw, "topics", where),
        required_markers=_str_tuple(raw, "required_markers", where),
        forbidden_markers=_str_tuple(raw, "forbidden_markers", where),
        expect_refusal=bool(raw.get("expect_refusal", kind == "hard_boundary")),
    )


def _check_unique(instructions: Iterable[Instruction]) -> None:
    seen: set[str] = set()
    for item in instructions:
        if item.id in seen:
            raise SpecError(f"duplicate instruction id: {item.id!r}")
        seen.add(item.id)


def _load_mapping(path: Path) -> dict[str, Any]:
    """Read a spec file into a plain mapping.

    TOML is handled by the standard library. YAML requires the optional
    ``[yaml]`` extra; the error says so rather than raising ImportError from
    deep inside the parser.
    """
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with path.open("rb") as handle:
            return tomllib.load(handle)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415 - optional dependency, imported lazily
        except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
            raise SpecError(
                "YAML specs require the optional 'yaml' extra: "
                "pip install 'promptproof[yaml]' (or use a .toml spec, which needs nothing)"
            ) from exc
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise SpecError(f"{path}: could not parse YAML spec ({exc})") from exc
        if not isinstance(loaded, dict):
            raise SpecError(f"{path}: top level of a YAML spec must be a mapping")
        return loaded
    raise SpecError(f"{path}: unsupported spec format {suffix!r} (use .toml, .yaml, or .yml)")


def load_spec(path: str | Path) -> BehaviorSpec:
    """Load and validate a behavior spec from disk.

    Args:
        path: Path to a ``.toml``, ``.yaml``, or ``.yml`` spec file.

    Returns:
        A fully validated :class:`BehaviorSpec`.

    Raises:
        SpecError: if the file is missing, unparseable, or fails validation.
            The message always names the offending field.
    """
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"spec file not found: {path}")
    try:
        data = _load_mapping(path)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as exc:
        raise SpecError(f"{path}: could not parse spec ({exc})") from exc
    return build_spec(data, base_dir=path.parent, source_path=path)


def build_spec(
    data: dict[str, Any],
    *,
    base_dir: Path | None = None,
    source_path: Path | None = None,
) -> BehaviorSpec:
    """Validate an already-parsed spec mapping.

    Separated from :func:`load_spec` so tests (and callers holding a spec in
    memory) do not need to touch the filesystem.
    """
    base_dir = base_dir or Path.cwd()
    name = _require_str(data, "name", "spec")

    prompt = data.get("system_prompt")
    prompt_file = data.get("system_prompt_file")
    if prompt and prompt_file:
        raise SpecError("spec: set exactly one of 'system_prompt' or 'system_prompt_file'")
    if prompt_file:
        prompt_path = Path(prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = base_dir / prompt_path
        if not prompt_path.is_file():
            raise SpecError(f"spec: system_prompt_file not found: {prompt_path}")
        prompt = prompt_path.read_text(encoding="utf-8")
    if not isinstance(prompt, str) or not prompt.strip():
        raise SpecError("spec: 'system_prompt' or 'system_prompt_file' is required")

    persona_raw = data.get("persona", {})
    if not isinstance(persona_raw, dict):
        raise SpecError("spec: 'persona' must be a table/mapping")
    persona = Persona(
        role=_require_str(persona_raw, "role", "persona"),
        tone=str(persona_raw.get("tone", "")).strip(),
        audience=str(persona_raw.get("audience", "")).strip(),
    )

    boundaries = tuple(
        _parse_instruction(raw, index=i, kind="hard_boundary", default_severity="hard")
        for i, raw in enumerate(data.get("hard_boundaries", []) or [])
    )
    behaviors = tuple(
        _parse_instruction(raw, index=i, kind="required_behavior", default_severity="hard")
        for i, raw in enumerate(data.get("required_behaviors", []) or [])
    )
    if not boundaries and not behaviors:
        raise SpecError(
            "spec: declare at least one hard_boundary or required_behavior - "
            "there is nothing to test otherwise"
        )

    out_of_scope = _str_tuple(data, "out_of_scope_topics", "spec")
    in_scope = _str_tuple(data, "in_scope_examples", "spec")

    scope = Instruction(
        id="scope:stay-in-scope",
        text=(
            f"Stay inside the assistant's scope ({persona.role}). Decline or redirect "
            "requests about: " + (", ".join(out_of_scope) if out_of_scope else "anything outside that role")
        ),
        kind="scope",
        severity="hard",
        topics=out_of_scope,
        expect_refusal=True,
        forbidden_markers=(),
        required_markers=(),
    )

    _check_unique((*boundaries, scope, *behaviors))

    return BehaviorSpec(
        name=name,
        description=str(data.get("description", "")).strip(),
        system_prompt=prompt,
        persona=persona,
        hard_boundaries=boundaries,
        required_behaviors=behaviors,
        in_scope_examples=in_scope,
        out_of_scope_topics=out_of_scope,
        source_path=source_path,
        _scope_instruction=scope,
    )
