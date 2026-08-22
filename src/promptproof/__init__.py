"""promptproof - synthetic evaluation harness for system prompts and agent skills."""

from .cases import AXES, TestCase, generate_cases
from .conversation import Message, Turn
from .drift import DRIFT_STRATEGIES, DriftStrategy, build_drift_ladder
from .errors import JudgeError, PromptProofError, ProviderError, SpecError
from .judge import Judge, Verdict
from .report import Report, build_report, render_json, render_markdown
from .rewrite import Rewrite, Rewriter
from .runner import CaseResult, Harness, RunResult
from .spec import BehaviorSpec, Instruction, Persona, build_spec, load_spec

__version__ = "0.1.0"

__all__ = [
    "AXES",
    "DRIFT_STRATEGIES",
    "BehaviorSpec",
    "CaseResult",
    "DriftStrategy",
    "Harness",
    "Instruction",
    "Judge",
    "JudgeError",
    "Message",
    "Persona",
    "PromptProofError",
    "ProviderError",
    "Report",
    "Rewrite",
    "Rewriter",
    "RunResult",
    "SpecError",
    "TestCase",
    "Turn",
    "Verdict",
    "__version__",
    "build_drift_ladder",
    "build_report",
    "build_spec",
    "generate_cases",
    "load_spec",
    "render_json",
    "render_markdown",
]
