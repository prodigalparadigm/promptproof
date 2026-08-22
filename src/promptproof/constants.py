"""Constants shared across modules that must not drift apart.

The two sentinels are the contract between the prompts this package renders and
the offline stub provider that recognises them. They lived in two files once;
one edit to a rubric header would have silently routed every judge call to the
subject role instead, and the offline suite would have gone quietly green on
nonsense. One definition, imported everywhere.
"""

from __future__ import annotations

#: First line of ``rubrics/judge_system.md``. Also a useful grep target in
#: provider logs.
JUDGE_SENTINEL = "PROMPTPROOF-JUDGE-V1"

#: First line of ``rubrics/rewrite_system.md``.
REWRITE_SENTINEL = "PROMPTPROOF-REWRITE-V1"

#: Held constant across subject models on purpose: varying the judge and the
#: model under test at the same time makes the comparison uninterpretable.
DEFAULT_JUDGE_MODEL = "claude-opus-5"

#: Environment variable that overrides the judge model when ``--judge-model``
#: is not passed.
JUDGE_MODEL_ENV_VAR = "PROMPTPROOF_JUDGE_MODEL"
