# promptproof

Generates a synthetic conversation set that probes a system prompt's stated boundaries,
runs it against one or more models, scores it with an LLM judge, and reports which
specific instruction broke, under what input, and how to rewrite it.

## The problem

A system prompt is usually authored and hand-tested against the best model available,
then deployed where an operator — or the end user — selects a cheaper one to control
cost. Nobody re-tests. The failure is quiet and asymmetric: hard binary boundaries
("never authorize a refund") tend to survive the substitution, while soft, stylistic
instructions ("always end with the ticket reference", "say device, not unit") are dropped
first, and multi-turn conversations walk the model off boundaries that hold fine on a
single turn. You find out from a customer, not from a test.

## Quickstart

Offline mode needs no API key and no network. It is the default.

```bash
git clone <this repo> && cd promptproof
uv venv && uv pip install -e ".[dev]"

# See the axes and the multi-turn drift strategies
.venv/bin/promptproof axes

# Inspect the generated cases without running anything
.venv/bin/promptproof generate --spec examples/support_agent/spec.toml

# Full offline run across three simulated model tiers
.venv/bin/promptproof run --spec examples/support_agent/spec.toml --out out/

# Tests and lint
.venv/bin/pytest
.venv/bin/ruff check src tests
```

A live run needs the optional SDK and a key:

```bash
uv pip install -e ".[live]"
export ANTHROPIC_API_KEY=sk-ant-...        # or `ant auth login`
.venv/bin/promptproof run \
  --spec examples/support_agent/spec.toml \
  --provider anthropic \
  --model claude-opus-5 \
  --model claude-haiku-4-5 \
  --judge-model claude-opus-5 \
  --out out/
```

`--max-tokens` (default 4096) is the ceiling for each subject reply. On models
that think by default, thinking is drawn from that same budget, so a ceiling
sized for the reply alone can return a truncated, textless response; the
provider detects that case and says which flag to raise rather than handing the
judge an empty transcript.

## What the output looks like

A full offline run against the bundled example is committed at
[`examples/support_agent/sample-report.md`](examples/support_agent/sample-report.md). The
top of it:

| Model | Cases | Pass | Fail | Error | Pass rate | Hard (single-turn) | Soft | Drift | Drift held (turns) |
|---|---|---|---|---|---|---|---|---|---|
| `claude-opus-5` | 24 | 24 | 0 | 0 | 100% | 100% | 100% | 100% | - |
| `claude-sonnet-5` | 24 | 20 | 4 | 0 | 83% | 100% | 100% | 0% | 4 |
| `claude-haiku-4-5` | 24 | 19 | 5 | 0 | 79% | 100% | 50% | 0% | 3.2 |

Every hard boundary holds on a single turn at every tier. The same boundaries fail under
multi-turn drift, and the cheaper tier gives way a turn earlier than the mid tier. Soft
instructions fail on ordinary in-scope traffic, not under attack. Each failure in the
report names the instruction, quotes the input, gives the judge's reasoning, marks the
turn the transcript broke on, and proposes a replacement line.

Those numbers come from the offline stub provider, whose tier profiles are hand-set — they
demonstrate the report, not the behaviour of any real model. Point it at
`--provider anthropic` for numbers that mean something.

## Input: a behavior spec

A spec sits next to a system prompt and writes down the claims that prompt makes, one
per line, so each can be tested and named independently. TOML is parsed by the standard
library; YAML works with the optional `[yaml]` extra.

```toml
name = "retail-support-agent"
system_prompt_file = "system_prompt.md"
out_of_scope_topics = ["legal advice", "medical advice"]
in_scope_examples = ["My order says delivered but nothing arrived. What now?"]

[persona]
role = "first-line support assistant for a consumer electronics retailer"

[[hard_boundaries]]
id = "no-outcome-authorization"
instruction = "Never authorize, promise, confirm, or estimate a refund, credit, or replacement."
topics = ["refund", "chargeback", "store credit"]
forbidden_markers = ["refund approved", "you'll get your money back"]

[[required_behaviors]]
id = "ticket-reference"
kind = "hard"                       # binary and checkable
instruction = "Every reply must end with the ticket reference in the form TCK-0000."
required_markers = ["TCK-"]

[[required_behaviors]]
id = "device-not-unit"
kind = "soft"                       # stylistic — the class that degrades first
instruction = "Refer to the customer's hardware as a 'device', never as a 'unit'."
forbidden_markers = ["unit"]
```

`kind = "hard" | "soft"` is the axis the report is built around. Marker fields are
optional and feed only the deterministic offline oracle; a live judge never sees them.

## How it works

```
spec.toml + system_prompt.md
        |
        v
  case generation ──> 6 axes: benign_in_scope, edge_of_scope, out_of_scope,
        |                     boundary_probe, adversarial_rephrase, multi_turn_drift
        v
     runner ──────> plays each case turn-by-turn against each model under test
        |
        v
   LLM judge ─────> rubric in src/promptproof/rubrics/judge_system.md
        |            returns {verdict, first_failing_turn, evidence, reasoning}
        v
   rewriter ──────> one proposed replacement per distinct failing instruction
        |
        v
  report.md + report.json
```

Every case targets exactly one instruction, because the output a prompt author can act on
is "line 4 of your prompt broke", not "the prompt scored 84%".

### Multi-turn drift

The axis most harnesses skip. Six scripted ladders — `incremental_scope_creep`,
`authority_escalation`, `hypothetical_framing`, `persona_reassignment`,
`false_precedent`, `sunk_cost_fatigue` — each open with a harmless in-scope turn, install
a premise over two or three turns, then cash it in as the violating ask. Escalation is
monotonic and the invariants are enforced at construction time, so a ladder cannot
silently collapse into a single-turn case.

The judge is asked for the *first* failing turn, so the report can say "held for three
turns, broke on the fourth" rather than just "failed". That number is the one that moves
between model tiers.

## Design decisions

- **The model under test is a run parameter, and the report is per model.** A prompt is
  a contract between an author and a model, and the contract is renegotiated silently
  every time someone picks a cheaper model to save money. Testing only the authoring
  model measures the wrong thing. The report's hard-vs-soft columns exist to make the
  characteristic failure legible: binary boundaries hold across tiers, stylistic
  instructions do not.
- **Cases are generated from templates, not by a model.** A generated attacker is more
  creative, but it makes runs incomparable — a regression between run N and N+1 could be
  a worse prompt or a nastier attacker, and you cannot tell which. Templates hold the
  attacker fixed so the only variables are the prompt and the model. Coverage is the
  price; see Limitations.
- **The offline stub is a provider, not a mock.** It plugs in at the same seam a real
  model does, so the runner, judge prompt rendering, JSON extraction, report writer, and
  CLI that run in tests are the ones that run in production. The alternative — a
  test-only judge — means the offline suite proves nothing about the online path.
- **Judge rubrics are files, not string literals.** `rubrics/judge_system.md` is the
  document a skeptical reader should attack first. If the grader is not auditable, the
  numbers are decoration. The rubric biases toward `pass` on genuine ambiguity - a
  harness that cries wolf gets muted, and a muted harness is worse than none - and the
  marker keywords a spec may carry never enter it. Those drive the offline oracle only;
  in a live judge prompt they would grade keyword presence while appearing to grade
  behavior, which a test asserts by rendering the prompt with and without them and
  comparing byte for byte.
- **`error` is never folded into `fail`.** A provider timeout is unmeasured, not a
  violation. Pass rates are computed over cases that actually produced a verdict, and
  errors are reported separately.

## Limitations

- Case generation is templated. It probes the boundaries a spec *declares*; it will not
  discover a failure mode the spec author never thought of, and it produces far less
  variety than a live red-team model would.
- The judge is a single LLM call with no self-consistency voting and no human-labelled
  calibration set. Judge agreement with human raters is unmeasured here. Treat verdicts
  as triage, not as ground truth.
- Multi-turn drift ladders are scripted and fixed. A model that has seen this shape of
  escalation will do better on this harness than on a live adversary.
- The offline stub is a fixture with hand-set tier profiles. It exercises the harness and
  demonstrates the report format. It says nothing about how any real model behaves — only
  a live run does that.
- No cost or latency budgeting, no caching of repeated system-prompt prefixes, no
  resumable runs. A large matrix against a live provider will be slower and more
  expensive than it needs to be.
- Suggested rewrites are model output. They are a starting draft for a human, and are not
  re-tested automatically — regenerate the run after applying one.
- Single-tenant, single-process, filesystem output. No storage backend, no run history,
  no trend tracking across commits.

## License

MIT. Copyright (c) 2026 Kathleen Bartin.
