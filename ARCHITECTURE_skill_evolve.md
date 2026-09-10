# Skill Evolver — Architecture

Component: session-to-skill distillation (`/skill-mine`, `/direct-skill-generation`, `/skill-review`)
Status: working draft; reflects branch `extract-session-history-generate-skill-md-and-other` as of
2026-09-08 — config `schema_version` 2, prompt contract 2, `FEATURES_VERSION` 4, `PROVENANCE_VERSION` 4,
decision report version 1.

## 1. Purpose

Skill Evolver turns a completed interactive msAgent session into a draft of a reusable
skill (`SKILL.md`). It is invoked on demand by the user, analyzes the recorded trajectory of
the session — user messages, tool calls, tool results and errors, approvals — and writes the
distilled knowledge into the standard skill library layout as an **inactive proposal**,
**without mutating the analyzed session** in any way.

What counts as knowledge worth keeping is a *selection policy* (`strict_knowledge` or
`reusable_workflow`, section 18); a *demo overlay* (`demo_mode`) additionally admits a
confirmed but trivial procedure as a teaching skill named `demo-<task-class>`. The overlay
lowers the novelty bar, never the evidence bar: every proposal, demo or not, passes the same
evidence checks, the same validator, the same semantic review and the same secrets scan.

The component follows the msAgent philosophy that domain expertise lives in prompts and
skills, not in code: the classification, rendering and review methodology is a set of
user-editable prompts plus one policy block per stage, the executable part is a thin, generic
pipeline shared by both generating commands.

## 2. User-facing behavior

| Invocation | Effect |
|---|---|
| `/direct-skill-generation [last\|<thread-id>] [--dry-run] [--policy strict_knowledge\|reusable_workflow] [--demo\|--no-demo]` | Analyze the **current** thread (no argument; deprecated, see section 17), the most recent **previous** thread (`last`) or an explicit thread; `--dry-run` stops after the code-only stage |
| `/skill-mine [--threads N] [--since 7d] [--dry-run] [--thread <id>] [--policy strict_knowledge\|reusable_workflow] [--demo\|--no-demo]` | Mine several recorded threads, up to `max_plans_per_thread` proposals per thread (section 17) |
| `/trajectories [list \| show <thread-id>]` | Browse the recorded trajectories (section 17) |
| `/skill-review [list \| accept <name> \| reject <name>]` | Review, promote or delete a proposal; the table has a `DEMO` column, and accepting a demo proposal asks one extra explicit confirmation (section 17.4) |

`--policy`, `--demo`, `--no-demo` and `--dry-run` are the same four flags on both generating
commands (`cli_args.take_shared_flag`); they override `config.skill.evolver.yml` for **this run
only** — the file is never changed. `--demo` together with `--no-demo`, or any flag given twice,
is an error that ends the command, not a precedence rule.

Output: `<root>/.proposals/<thread-id>/<skill-name>/SKILL.md` plus `provenance.json`, where
`<root>` is `<working_dir>/skills` or `generation.output_dir` from the component config (`~`
expanded, a relative path resolved against the working directory). The skill name is the
validated frontmatter `name` — for a demo proposal it starts with `demo-`; collisions inside
the thread folder get `-2`, `-3`, … suffixes. **Nothing is written into a library category.**
A proposal becomes active only when a human reviews it and moves the folder to
`<root>/<category>/<skill-name>/` (a `create` proposal) or replaces the existing `SKILL.md`
with it (an `update` proposal). Until then it is invisible to `/skills`, to the `get_skill`
tool and to the agent's system prompt (section 9).

Every non-dry run also writes a **decision report** into the project's private state
(`<state>/skill-evolver/decisions/`, section 20) — also when nothing was saved — so that a
thread without a recorded trajectory, a session below the evidence threshold, a `nothing`
verdict, a `SKILL.md` that fails validation twice, a failed quality review, an exhausted LLM
budget or a package with a secret-looking value all end **without writing a proposal** but
with a record of why. A dry run creates no LLM, writes no proposal and no report.

## 3. Component inventory

Component files:

| File | Role |
|---|---|
| `src/msagent/skill_evolver/pipeline.py` | The **single implementation** of the per-thread stage sequence, `run_thread(RunContext, ThreadInput) -> ThreadResult` (section 7): both gates, bundle, context fit, classify v2, the `expand_context_once` round, `render_plan` (render → validate → semantic review → provenance v4 → proposal), the LLM call budget, the decision report; plus the run helpers both commands share (`print_policy_block`, `report_config_error`, `bundle_preview`, `gate_lines`, `load_prompts`, `LazyLlm`, `record_gate_refusal`). Never imports `mining` or `direct_skill_generation` |
| `src/msagent/skill_evolver/direct_skill_generation.py` | `DirectSkillGenerationHandler`: argument parsing (`parse_direct_args`, `DirectOptions`), config/prompt loading, `_gather_evidence`, dry run, one `run_thread` call; re-exports `STAGES`, `DirectSkillGenerationConfig`, `PlanContext`, `PlanOutcome`, `PlanTally` for the tests; the legacy session-replay helpers are kept but unused (section 8) |
| `src/msagent/skill_evolver/mining.py` | `SkillMiningHandler`: the multi-thread `/skill-mine` command, `parse_mine_args`, thread selection, the dry-run tables and the per-thread loop over `run_thread` (section 17) |
| `src/msagent/skill_evolver/config.py` | `config.skill.evolver.yml` schema_version 2: pydantic model `SkillEvolverConfig` (strict, `extra="forbid"`), the flat `DirectSkillGenerationConfig` view, `load_skill_evolver_config`, in-memory v1 migration, `SkillEvolverConfigError` with dotted paths, `apply_overrides`, `effective_rules` / `EffectiveRules`, `resolve_output_dir` / `output_root` (section 5). Imports only stdlib, `yaml`, `pydantic` and `msagent.core` path constants |
| `src/msagent/skill_evolver/prompts.py` | Stage prompt resolution under prompt contract 2: `resolve_stage_prompt`, `PromptText`, `StagePrompts`, `prompt_sha256`, `declared_contract`, `REQUIRED_PLACEHOLDERS`, `PromptContractError` (section 6) |
| `src/msagent/skill_evolver/policy.py` | The policy blocks inserted into the three prompts: `selection_policy_block(selection)` (classify), `render_policy_block(demo)`, `review_policy_block(demo)`, `DEMO_NAME_PREFIX = "demo-"` (section 18) |
| `src/msagent/skill_evolver/budget.py` | `CountingLlm` / `LlmBudgetExhausted` (the hard stop on `ainvoke` calls), `llm_call_bound`, `thread_call_ceiling`, `ContextBudget` (section 19) |
| `src/msagent/skill_evolver/cli_args.py` | `take_shared_flag`, `SharedFlags`, `CliArgsError`, `POLICY_USAGE` — the four flags shared by both generating commands |
| `src/msagent/skill_evolver/report.py` | The decision report: `decisions_dir`, `write_report` (atomic, private), `is_synthetic`, `REPORT_VERSION = 1` (section 20) |
| `src/msagent/skill_evolver/features.py` | Code-only candidate extraction over recorded trajectories: `Episode`, seven detectors (six scoring kinds plus the demo-only `observed_procedure`), `extract_episodes`, `mine_cross_session`, `classify_approval`, `group_incidents`, `evidence_score`, `gate_decision` (demo-aware), `expand_episode_context`, `REQUIRED_ROLES`, `has_required_evidence`, `DetectorNote`, `FEATURES_VERSION` (section 14) |
| `src/msagent/skill_evolver/retrieval.py` | Stdlib BM25 over skill descriptions (`SkillDoc`, `BM25Index`) used by the `skill_gap` detector; Han runs are indexed as CJK bigrams |
| `src/msagent/skill_evolver/bundle.py` | Evidence bundle with `E<n>` episode ids and `[evN]` fragment ids, demo ranking, `EXCLUDED_CODE` (section 15) |
| `src/msagent/skill_evolver/classify.py` | Classify contract v2: `Decision` / `ReasonCode`, `ClassifyResult`, `Classification`, `_contract_violations`, one corrective retry, `ClassifyParseError` / `ClassifyContractError` (section 15) |
| `src/msagent/skill_evolver/render.py` | Render stage: `plan_render`, budgeted `select_plan_evidence` / `EvidenceSelection`, `render_skill_md` with one corrective call, `revise_skill_md` (the single correction after a failed review), `resolve_library_skill` (section 16) |
| `src/msagent/skill_evolver/review.py` | Semantic review: `review_skill_md`, `ReviewIssue` / `ReviewReply` / `ReviewResult`, and `render_and_review` — the per-plan stage both commands call, at most 6 LLM calls per plan (section 16) |
| `src/msagent/skill_evolver/validator.py` | Code validation of a rendered `SKILL.md`: `validate_skill_md(..., required_prefix=)`, `ValidationResult`, `skill_name`, the secret patterns (`scan_secrets`, `redact_secrets`) (section 16) |
| `src/msagent/skill_evolver/writer.py` | Proposal writer: `build_provenance` (v4), `scan_package`, `SecretsDetected`, `write_proposal` into `.proposals/` (section 16) |
| `src/msagent/skill_evolver/exgraph_context.py` | `attach_stored_graph`: the stored experience-graph appendix of the classify prompt (context, never citable) |
| `src/msagent/cli/handlers/session_history.py` | Shared **read-only** access to persisted thread history: `load_history` (thread resolution, incl. `last`), `latest_other_thread`, `trim_history` |
| `src/msagent/cli/handlers/trajectories.py` | `TrajectoriesHandler`: `/trajectories list` and `show` over `trajectory_recorder.export` (section 17) |
| `src/msagent/cli/handlers/skill_review.py` | `SkillReviewHandler`: `/skill-review list` (with the `DEMO` column), `accept` (demo confirmation, `demo-` prefix re-checked) and `reject` over `.proposals/` (section 17.4) |
| `resources/configs/default/config.skill.evolver.yml` | Packaged default component config, the commented schema_version 2 document (section 5) |
| `resources/configs/default/skill-evolver/prompts/<stage>/prompt_v2.md` | Packaged stage prompts of contract 2 for `classify/`, `render/` and `review/`; `classify/prompt_v1.md` and `render/prompt_v1.md` stay in the package for the legacy prompt rule (section 6); `default/prompt_v1.md` is the legacy replay prompt |
| `tests/fixtures/trajectories/skill_evolver_signals.jsonl` | Hand-written trajectory exercising every per-trajectory scoring detector |
| `tests/fixtures/trajectories/skill_evolver_demo_success.jsonl` | Synthetic thread `thread-demo-synthetic` (agent `SyntheticDemo`, working dir `/synthetic/demo`, header `"synthetic": true`): one `dispatch` turn, `read_file` → `bash` (sum) → `bash` (check → `TOTAL_OK`), no scoring episode at all, exactly one `observed_procedure` under demo (section 14) |
| `tests/ut/skill_evolver/test_features.py`, `test_retrieval.py`, `test_bundle.py`, `test_classify.py`, `test_render.py`, `test_validator.py`, `test_writer.py`, `test_direct_skill_generation.py`, `test_exgraph_context.py` | Detector, retrieval, bundle, classify v2, render, validator, writer v4 and handler tests on scripted fake LLMs; no-langchain/no-network probes |
| `tests/ut/skill_evolver/test_evolver_config.py`, `test_prompts.py`, `test_policy.py`, `test_budget.py`, `test_cli_args.py`, `test_review.py`, `test_pipeline.py`, `test_report.py` | Tests of the new modules (the config file is named `test_evolver_config.py`: a `test_config.py` basename collides with `tests/ut/exgraph/test_config.py` when both directories run in one pytest invocation); `test_pipeline.py` drives `run_thread` with scripted replies (budget stops, context fit, expand round, `nothing` causes, demo overlay on both gates, review correction, report in `finally`) |
| `tests/ut/skill_evolver/test_demo_pipeline.py`, `test_demo_fixture_executable.py` | The demo fixture end to end on a scripted LLM (both commands, on/off, `on_nothing: stop`, duplicate of an active skill, secrets, exgraph on/off), and the executability of the fixture commands and of the rendered example in a whitelisted `python3 -c` sandbox (section 21) |
| `tests/ut/cli/handlers/test_skill_mining.py`, `test_trajectories_handler.py` | Command tests: parsers (shared flags included), tables, the no-LLM dry run with overrides, the per-thread loop on a scripted LLM, `/skill-review` incl. the `DEMO` marker and the accept confirmation |

Modified files (integration points):

| File | Change |
|---|---|
| `src/msagent/cli/dispatchers/commands.py` | Handler instantiation, `"/direct-skill-generation"` entry in `_register_commands()`, `cmd_direct_skill_generation` delegate; the three commands of section 17 and the `[deprecated]` marker on the generator's help text |
| `src/msagent/cli/handlers/__init__.py` | Export of `DirectSkillGenerationHandler`, `SkillMiningHandler`, `TrajectoriesHandler`, `SkillReviewHandler` |
| `src/msagent/core/constants.py` | `CONFIG_SKILL_EVOLVER_FILE_NAME`, `SKILL_EVOLVER_CONFIG_FOLDER_NAME` |
| `src/msagent/core/storage_layout.py` | `_seed_skill_evolver_defaults()` called from `validate_and_initialize_storage_layout()`; `"skill-evolver"` added to `_MANAGED_DIRECTORIES` |
| `src/msagent/skills/factory.py` | `SkillFactory.load_skills` skips dot-directories below a scanned root (`_in_hidden_dir`), so `.proposals/` never enters the catalogue |
| `src/msagent/exgraph/enrich.py` | Episode nodes are keyed by `Episode.primary_seq` instead of `evidence_seq[0]` (section 14: the `task` context item would otherwise merge two `error_recovery` nodes of one turn) |
| `src/msagent/exgraph/skills.py` | `_evolver_output_dir` reads `generation.output_dir` through `config.load_skill_evolver_config` + `resolve_output_dir` (lazy import, fail-open) instead of its own top-level-key parser |

## 4. Runtime layout and startup seeding

```text
~/.msagent/                              (MSAGENT_HOME)
├── config/
│   └── config.skill.evolver.yml         # component configuration, schema_version 2 (section 5)
└── skill-evolver/                       # isolated component folder
    └── prompts/
        ├── classify/
        │   ├── prompt_v1.md            # contract 1 (kept for the legacy prompt rule, section 6)
        │   └── prompt_v2.md            # evidence → JSON decisions + candidates (section 15)
        ├── render/
        │   ├── prompt_v1.md            # contract 1 (kept for the legacy prompt rule)
        │   └── prompt_v2.md            # candidates → SKILL.md (section 16)
        ├── review/
        │   └── prompt_v2.md            # SKILL.md → semantic review JSON (section 16)
        └── default/
            └── prompt_v1.md            # legacy replay prompt (unused by handle())
```

Seeding is performed once per process start by `_seed_skill_evolver_defaults()`
(`core/storage_layout.py`), invoked from `validate_and_initialize_storage_layout()`
right after the managed directories are ensured and **before** the early-return
branches, so upgrades of an existing home also receive new files. Mapping:

- `resources/configs/default/config.skill.evolver.yml` → `~/.msagent/config/config.skill.evolver.yml`
- `resources/configs/default/skill-evolver/**` → `~/.msagent/skill-evolver/**` (whole component tree)

Semantics: **copy-if-missing** — user edits are never overwritten; new files added in
later builds do arrive, changed defaults for already-materialized files do not. Concretely,
a home seeded by an earlier build receives `prompts/{classify,render}/prompt_v2.md` and the
new `prompts/review/` folder automatically, while its `config.skill.evolver.yml` (flat v1) and
its `prompt_v1.md` copies stay as they are. Both cases are handled at run time, not at
startup: a v1 config is migrated in memory (section 5), an untouched `prompt_v1.md` copy is
bypassed in favour of the packaged `prompt_v2.md`, and a **customised** v1 prompt stops the
run with a migration instruction before any LLM exists (section 6) — a stale methodology
never runs silently. Failures of the seeding itself are logged as warnings and never abort
startup. The `skill-evolver` home folder participates in the standard layout validation
(regular directory, no symlink) via `_MANAGED_DIRECTORIES`.

Design note: `~/.msagent/skill-evolver/` is deliberately **outside** every stock
loading mechanism (`ConfigRegistry` does not scan it). It is owned exclusively by this
component, which keeps it isolated from the agent/LLM/MCP configuration lifecycle.

## 5. Configuration (schema_version 2)

`config.skill.evolver.yml` is read once per command run by `config.load_skill_evolver_config`
(resolution chain: user file in `~/.msagent/config/` → packaged default in the wheel →
dataclass defaults; the file is never written). The packaged document:

| Key | Default | Validation (`SkillEvolverConfig`, pydantic strict, `extra="forbid"`) |
|---|---|---|
| `schema_version` | `2` (required) | whole number equal to 2; else `unsupported schema_version N; this build supports 2` / `must be a whole number, got '2'` |
| `demo_mode` | `false` | boolean (`"false"` is not: `must be a boolean (true/false), got 'false'`) — the demo overlay, section 18 |
| `classification.policy` | `strict_knowledge` | `must be one of strict_knowledge, reusable_workflow, got '…'` |
| `gate.min_evidence_score` | `1.0` | `must be a finite non-negative number, got …` — the YAML integer `1` is accepted, a boolean is not; threshold of both gates (section 14) |
| `evidence.excerpt_max_chars` | `1000` | `must be a positive whole number, got 0` (`must be a whole number, got True`) — longest cut of one event in the bundle |
| `evidence.bundle_max_chars` | `30000` | positive whole number — the bundle budget; the model context is checked separately at run time (section 19) |
| `evidence.surrounding_events` | `2` | `must be a whole number >= 0, got -1` — neighbours per side added by `expand_context_once` |
| `evidence.cross_session_limit` | `20` | positive whole number — newest trajectories of the agent mined for `repeated_procedure` |
| `on_nothing.action` | `expand_context_once` | `must be one of stop, expand_context_once, got '…'` |
| `generation.max_plans_per_thread` | `3` | positive whole number — extra plans are `deferred` |
| `generation.max_llm_calls_per_thread` | `16` | positive whole number — the hard stop of `CountingLlm` (section 19) |
| `generation.category` | `default` | text, `must be one directory name without separators, got 'a/b'` — recorded in provenance, never written to automatically |
| `generation.output_dir` | `null` | text or null, `must be a non-empty path, got ''`; stored as written, `~` expanded and a relative path resolved against `working_dir` by `resolve_output_dir` / `output_root` (default `<working_dir>/skills`) |
| `prompts.contract_version` | `2` | whole number, `must equal 2 (prompt contract of this build)` |
| `prompts.classify`, `prompts.render`, `prompts.review` | `prompt_v2.md` | text, `must be a file name without path separators or '..', got '/etc/passwd'` — file inside `skill-evolver/prompts/<stage>/` (section 6) |
| `diagnostics.save_decision_report` | `true` | boolean — the decision report (section 20) |
| `diagnostics.save_evidence_text` | `false` | boolean — additionally store the bundle text next to the report |

Any other key is `unknown key`; a section given as a scalar is `must be a mapping`; a non-string
name is `must be text, got 5`. Nothing falls back
silently: **every** problem of the file is collected and raised at once as
`SkillEvolverConfigError`, whose `lines()` have the exact shape

```text
<file>: <dotted.path>: <message>
```

e.g. `config.skill.evolver.yml: gate.min_evidence_score: must be a finite non-negative number, got 'abc'`
(the file name is the user or packaged file; a YAML syntax error or a non-mapping document is
reported under the path `<file>`). The two generating commands catch it (`mining.handle`,
`dsg.handle`) and print the lines through `pipeline.report_config_error`, followed by the muted
hint `Fix ~/.msagent/config/config.skill.evolver.yml (schema_version: 2) or delete it to use the
packaged defaults; nothing was run.`; `skill_review.handle` prints the same lines (it loads no
prompts, so it never meets a `PromptContractError`). The session continues, no LLM is created.

**v1 files (in-memory migration).** A file *without* `schema_version` is the flat v1 document
of earlier builds. It is normalised in memory — nothing is written back — and the result
carries `legacy_format=True`:

| v1 key | Becomes | Notes |
|---|---|---|
| `min_evidence_score` | `gate.min_evidence_score` | validated by the v2 model; an error is reported under the **v1** key (`min_evidence_score: must be a finite non-negative number, got 'abc'`) |
| `max_plans` | `generation.max_plans_per_thread` | idem (`max_plans: must be a positive whole number, got 0`) |
| `category` | `generation.category` | idem |
| `output_dir` | `generation.output_dir` | idem |
| `active` | kept as `DirectSkillGenerationConfig.active` (legacy replay variant folder) | validated as one directory name; note `active: legacy replay field, ignored` |
| `prompt_file` | kept as `DirectSkillGenerationConfig.prompt_file` | validated as a file name; drives the legacy prompt rule of section 6; note `prompt_file: legacy setting; prompt files are resolved by contract_version (see prompts)` |
| anything else | error `<key>: unknown key` | |

Every other v2 key takes its default; an empty file is a v1 file with defaults. The notes are
printed as `Config note: …` by `print_policy_block` and recorded in `config.requested.notes`.
The reverse is an error: a v1 key inside a `schema_version: 2` file is reported as
`legacy v1 key in a schema_version 2 file; use gate.min_evidence_score` (resp.
`… use generation.max_plans_per_thread` etc.) or, for `active` / `prompt_file`,
`… remove it (prompts are configured under prompts:)`, before the model validation runs.

**Precedence: defaults → file → CLI → effective rules.** `load_skill_evolver_config` yields the
flat, frozen `DirectSkillGenerationConfig` (the *requested* configuration: `source` is
`"defaults"`, `"packaged"` or the user file path; `notes`; `overrides`). `apply_overrides(cfg,
policy=, demo=)` applies `--policy` / `--demo` / `--no-demo` for this run and appends
`"--policy <p>"`, `"--demo"` or `"--no-demo"` to `overrides`. `effective_rules(cfg,
working_dir=)` produces the frozen `EffectiveRules` every stage reads: `policy`, `demo`,
`selection` (`demo_workflow` when demo, else the policy), the copied limits and budgets,
`category`, `output_root`, the two diagnostics flags. **Demo changes the selection only** —
`test_demo_keeps_limits_unchanged` pins that `max_plans`, `max_llm_calls`, `bundle_max_chars`,
`excerpt_max_chars`, `surrounding_events` and `cross_session_limit` are identical with and
without the overlay. Both records are kept apart everywhere downstream: `config.requested`
(`DirectSkillGenerationConfig.as_record()`, the v2 nested shape plus `source`, `legacy_format`,
`overrides`, `notes`) and `config.effective` (`EffectiveRules.as_record()`, flat) appear in
`provenance.json` and in the decision report, and the console says where each value came
from (`Requested policy: reusable_workflow (--policy)`, `Demo mode: true (--demo)` versus
`(config.skill.evolver.yml)`).

The config is intentionally **not** part of the `VersionedConfig`/`ConfigRegistry` framework:
it is component-local, has no cross-references to resolve, and `schema_version` plus the
in-memory v1 migration carry its whole migration story. `exgraph/skills.py` reads
`generation.output_dir` through the same loader (lazily imported, fail-open with a debug log)
so the two consumers can never disagree on where `.proposals/` lives.

## 6. Prompt resolution

The evidence pipeline has three stages (`config.STAGES = ("classify", "render", "review")`) and
one prompt file per stage, named explicitly by `prompts.<stage>` (default `prompt_v2.md`).
`prompts.resolve_stage_prompt(user_root, cfg, stage, *, notes)` looks for
`~/.msagent/skill-evolver/prompts/<stage>/<name>` first and the packaged
`resources/configs/default/skill-evolver/prompts/<stage>/<name>` second — the user copy wins.
There is **no glob fallback any more**: a missing file is
`ValueError("prompt file not found: <user path> (nor packaged <packaged path>); check
prompts.<stage> in config.skill.evolver.yml")`, never a concatenation of whatever `*.md` lies in
the folder (a typo must not silently switch the methodology). There is **no prompt text
embedded in Python** — the packaged templates are the single source of truth.

**Prompt contract.** Every contract-2 prompt file starts with two header lines:

```text
## description: <one sentence>
## contract_version: 2
```

`declared_contract(text)` searches `CONTRACT_HEADER_RE` (`^## contract_version:\s*(\d+)\s*$`) in
the first `HEADER_LINES` (5) lines; a file without the header is contract 1
(`LEGACY_CONTRACT`). The declared contract must equal `prompts.contract_version`, and every
placeholder of `REQUIRED_PLACEHOLDERS[stage]` must be present — `{skill_library}`,
`{evidence_bundle}`, `{selection_policy}` for classify; `{candidates}`, `{existing_skill}`,
`{render_policy}` for render; `{skill_md}`, `{candidates}`, `{evidence}`, `{existing_skill}`,
`{review_policy}` for review. A violation is a `PromptContractError` that stops the command
before any LLM exists (printed by `report_config_error`, section 5):
`<path> declares contract_version 1 but prompts.contract_version is 2; Migrate: …` or
`<path> lacks the {render_policy} placeholder required by contract 2`.

**Legacy `prompt_file` rule.** Only a v1 config (`legacy_format=True`) with
`prompt_file: prompt_v1.md` triggers it: when the user's `prompts/<stage>/prompt_v1.md` is
absent or **byte-identical** to the packaged `prompt_v1.md`, the packaged
`prompts/<stage>/prompt_v2.md` is used and a warning is printed — `prompt_file 'prompt_v1.md'
is a legacy setting; using packaged prompts/<stage>/prompt_v2.md (contract 2)`; when the copy was
**edited**, the run stops with `PromptContractError`:

```text
<user path> is a customised contract-1 prompt; this build needs contract 2. Migrate: copy
<packaged root>/<stage>/prompt_v2.md to <user root>/<stage>/ and port your edits, then set
prompts.<stage>: prompt_v2.md in config.skill.evolver.yml (schema_version 2). The file was not modified.
```

The same `Migrate: …` sentence follows a declared-contract mismatch of a v2 config that points
at a contract-1 file. Files are never modified by the pipeline. (The review stage has no
packaged `prompt_v1.md`; a user `review/prompt_v1.md` would count as customised — seeding never
creates one, so the case is theoretical.)

**Hashes.** `prompt_sha256(text)` is `"sha256:" + sha256(text)` of the **template as loaded,
before any substitution**; `PromptText(stage, text, source, sha256, contract_version)` carries
it and `StagePrompts.hashes()` / `variants()` put `prompt_hashes` and `prompt_variants` (the
resolved source paths) of all three stages into `provenance.json` and the decision report, so a
proposal can be traced to the exact prompt bytes that produced it. The handler's
`_load_stage_prompt(root, cfg, stage) -> (text, source)` keeps its historical signature (tests
patch it); `pipeline.load_prompts(loader, root, cfg)` hashes whatever text the loader returned
and records `cfg.contract_version` — a patched fake gets an honest hash and the contract check
lives only in `resolve_stage_prompt`, i.e. on real files. The legacy replay
`_load_prompt_template(root, cfg)` resolves the explicit file `<root>/<active>/<prompt_file or
prompt_v1.md>` (user, then packaged) with neither glob nor contract check.

**Substitution.** Placeholders are substituted by code, never with `str.format`, so braces
inside the data are inert. The classify template receives `{skill_library}` (programmatic
inventory of the loaded skills) and exactly one `{selection_policy}` block via `str.replace` in
the pipeline — `run_thread` raises `ValueError("classify template has no {selection_policy}
placeholder")` when the placeholder is absent, and `classify()` refuses a template that still
contains either placeholder — and `{evidence_bundle}` inside `classify()`. The render template
gets `{render_policy}`, `{candidates}` and `{existing_skill}`, the review template
`{review_policy}`, `{skill_md}`, `{candidates}`, `{evidence}` and `{existing_skill}`, each in one
regex pass inside `render_skill_md` / `review_skill_md`. The legacy replay prompt keeps
`{agent}`, `{thread_id}`, `{working_dir}` and `{history}`.

## 7. Execution pipeline

Both commands run the same code: they parse their flags, load config and prompts, gather the
evidence of a thread by code, and hand a `ThreadInput` to `pipeline.run_thread` under one
`RunContext` per run (`/skill-mine` reuses it for every thread of the run: the passing ones go
through `run_thread`, the gate-refused ones through `record_gate_refusal`, section 20).

```text
/direct-skill-generation [last|<id>] [--dry-run] [--policy P] [--demo|--no-demo]
/skill-mine [...] [--dry-run] [--policy P] [--demo|--no-demo]
        │
        ▼
 _load_config() → apply_overrides(policy, demo) → effective_rules(working_dir)
        │                       SkillEvolverConfigError / PromptContractError →
        │                       report_config_error (error lines + hint), stop; no LLM
        ▼
 load_prompts(_load_stage_prompt, root, cfg)   classify / render / review, contract 2,
        │                       sha256 of each template (§6)
        ▼
 load_history() / select_trajectories()        read-only thread resolution; the JSONL via
        │                       find_trajectory_file; no file → print_error, no LLM;
        │                       TrajectoryReadError → "Trajectory unreadable (…); the LLM
        │                       was not called" (no report either: no thread was built)
        ▼
 _gather_evidence()             extract_episodes(current, skill_index, demo=rules.demo,
        │                       notes) + mine_cross_session over the agent's newest
        │                       cross_session_limit trajectories (§14); under demo the
        │                       observed_procedure detector runs and its drop reasons
        │                       land in ThreadInput.notes
        ▼
 print_policy_block()           muted: Requested policy (source), Demo mode (source),
        │                       Effective selection, the demo overlay line, Config note(s)
        ▼
 --dry-run?  ── yes ──▶ gate_lines() + bundle_preview(); "Dry run: nothing was written,
        │               no LLM was created, no decision report saved." — stop
        ▼ no
 run_thread(run, thread)        ┌─ report written in `finally` (§20) ─────────────────┐
        │                       │                                                     │
        ▼                       │                                                     │
 gate 1  gate_decision(episodes, min_score, demo=rules.demo)                         │
        │                       muted "Evidence score: S (min_evidence_score M; N     │
        │                       episodes, K incidents)" and "Gate: pass|skip          │
        │                       (describe())"; demo passes via demo_override when a   │
        │                       complete episode exists (§14); refused → print_info   │
        │                       "Nothing to save: …" (+ "; <detail>" under demo), stop │
        ▼                                                                             │
 build_evidence_bundle(max_chars=bundle_max_chars, excerpt_chars, demo)  (§15)       │
        │                       excluded episodes → warning + insufficient_context_   │
        │                       budget code rejection; second gate over bundle.kept   │
        │                       (demo-aware, only when something was excluded);       │
        │                       failing → stop. attach_stored_graph appendix.          │
        ▼                                                                             │
 LazyLlm.get() → CountingLlm(limit=max_llm_calls_per_thread), ContextBudget.for_window │
        │                       the first and only LLM construction site (§19)        │
        ▼                                                                             │
 classify template ← {skill_library}, one {selection_policy} block (str.replace)      │
        │                       context fit, once: bundle cut to the room the prompt │
        │                       leaves ("Bundle cut to R chars …", re-gated) or stop │
        │                       "Nothing to save: the classify prompt does not fit   │
        │                       the model context (…)" when the room < 500 chars     │
        ▼                                                                             │
 classify(text, bundle.shown, llm, template, episode_ids=bundle.episode_ids, demo)    │
        │                       contract v2 (§15): one call + at most one corrective │
        │                       retry; muted "Decision E1,E2: accept|reject (code)"  │
        │                       and "Candidate '<title>': accepted[ (trivial          │
        │                       procedure allowed)]"; rejected candidates → warning  │
        │                       + invalid_evidence; parse/contract error → recorded  │
        │                       in the report, thread failed, re-raised             │
        ▼                                                                             │
 expand_context_once            only when verdict == nothing ∧ insufficient_evidence  │
        │                       ∧ on_nothing == expand_context_once: expand_episode_ │
        │                       context(only=flagged episodes, surrounding_events);  │
        │                       retried only if the new bundle shows a strict        │
        │                       superset of fragments ("Insufficient evidence: one   │
        │                       more classification round with N added events");    │
        │                       else "expand_context_once: no new context could be   │
        │                       shown; not retried"; second nothing is final          │
        ▼                                                                             │
 nothing by cause               "Nothing to save: insufficient evidence in thread …  │
        │                       (classifier)" | "… no candidate with valid evidence  │
        │                       refs … (invalid_evidence)" | "… no durable learning   │
        │                       found in thread …" — stop                             │
        ▼                                                                             │
 plan_render(candidates, skills, max_plans)   (§16)                                   │
        │                       reference → read_existing_skill; coverage text_read| │
        │                       unverified; note "reference <skill>: <title>          │
        │                       (classifier's claim; skill text read|unreadable,      │
        │                       coverage not verified)"; rejected/deferred reported   │
        ▼                                                                             │
 for each plan: render_plan()   llm.exhausted → "Deferred plan: … — llm call budget   │
        │                       exhausted (U/L)"; render_and_review ≤ 6 calls (§16):  │
        │                       render (≤2) → validate → review (≤2) → one revise →   │
        │                       review; provenance v4 (§16) → write_proposal (secrets │
        │                       scan of the whole package). Success: "Skill proposal  │
        │                       saved to …", activation hint, muted "Quality review:  │
        │                       passed[ after one correction]", "Verification:        │
        │                       evidence_supported; not executed by generator",       │
        │                       "Proposal: saved, inactive[, DEMO]". Refusals:        │
        │                       print_error "plan <label>: …" by code, nothing written │
        └──────────────────────────────────────────────────────────────────────────────┘
```

The `llm` is `LLMFactory.create(load_llm_config(ctx.model, ctx.working_dir))`, one client for
every stage of the run, wrapped per thread in a `CountingLlm`; every stage runs outside the
graph (no tools, no checkpointer, no middleware). All pre-gate and status lines go through
`console.print` as muted text, never `print_info` — the tests that count `info` entries pin
this. `run_thread` records classify parse/contract errors as code rejections and re-raises
them, so `/skill-mine` counts the thread as failed (`thread <id>: classify: reply violates the
contract after one retry: …`) instead of implying "nothing to save".

## 8. Session replay design (legacy, unused by `handle()`)

`_generate_skill_md` and its helpers remain in the module but are no longer called
(user decision, 2026-09-04: keep until the evidence pipeline has proven itself, then
delete). The design is recorded here for that decision.

Instead of serializing history into prompt text, the component **replays the genuine
message sequence** to the analysis model, preceded by a guard `SystemMessage`
(`_REPLAY_SYSTEM_PROMPT`) stating that the session is completed, must not be continued,
and only the final instruction message is to be followed.

Normalization (`_prepare_replay_messages` and helpers):

- **Reasoning folding** — providers reject a `reasoning_content` field on inbound
  messages, so past private reasoning is moved into the visible text of each
  `AIMessage` as a `<past_reasoning>…</past_reasoning>` block; the raw field is
  stripped.
- **Orphan tool calls** — a trailing `AIMessage` with `tool_calls` that never received
  its `ToolMessage` (interrupted session) is dropped to satisfy the strict
  assistant/tool pairing rules of OpenAI-compatible APIs.
- **Budget trimming** — `trim_history()` (`cli/handlers/session_history.py`) trims the
  replay to `_HISTORY_BUDGET_RATIO` (0.6) of the model's `context_window` (token
  counting via `utils.compression.calculate_message_tokens`). The default `head_tail`
  strategy keeps the first 6 messages (the task statement — the most valuable part for
  distillation) and cuts from the **middle**, keeping the newest tail; the tail is
  re-aligned to a `HumanMessage` boundary, and the head is shortened to end on a
  complete tool exchange so no `tool_call` is left without its `ToolMessage`. Only if
  the head alone exceeds the budget does the function fall back to the `tail` strategy
  on the full history (oldest dropped, newest kept; `tail` is also selectable
  explicitly). The omission count is reported to the model inside the instruction
  (`[Note: N messages from the middle of the session were omitted due to context
  limits.]`; in the degenerate fallback case the omitted messages are in fact the
  oldest ones).

What the analyst model sees natively: user turns, assistant turns with structured
`tool_calls` (name + arguments), tool results including error payloads, folded past
reasoning. What it cannot see: the session's system prompt (not stored in `messages`
— it is injected per-call by middleware in normal operation) and reasoning the
provider never returned. Multimodal blocks (images) pass through unchanged and require
a vision-capable analysis model.

## 9. Non-mutation guarantees

The analyzed thread is untouched by construction, on these pillars:

1. history is obtained only through read-only APIs (`aget_state` / `aget_tuple`), and the
   trajectory JSONL only through the recorder's reader;
2. the LLM is called directly through `LLMFactory`, bypassing the graph, the
   checkpointer and the middleware stack — nothing is appended to any thread;
3. files are written by the handler with plain file I/O — no agent tools run, so no
   HITL interrupts can fire; **no command of the trajectory is ever executed** by the
   generator (the provenance says so: `verification.level` is always `evidence_supported`);
4. the output is a **proposal** outside every skill scanner: `SkillFactory.load_skills`
   skips dot-directories below a scanned root, and
   `<root>/.proposals/<thread>/<name>/SKILL.md` lies one level below the depth at which
   `AgentFactory._resolve_existing_paths` and the deepagents `SkillsMiddleware` look
   for skills (`<child>/<sub>/SKILL.md`). The flat layout `.proposals/<name>/` would
   not do: `_resolve_existing_paths` would list `.proposals` as a deepagents source,
   and an `update` proposal (same name as a library skill, so it passes the name
   allow-list) would win the last-source-wins merge whenever the real skill is flat
   under `<working_dir>/skills/<name>/` or `output_dir` points at a later root such as
   `~/.msagent/skills`. Guarded by `test_writer.py::test_proposals_not_scanned_by_skill_factory`,
   `test_proposals_invisible_to_agent_factory_sources` and
   `test_direct_skill_generation.py::test_proposal_invisible_to_skill_scanners`;
5. the decision report lives in the project's **private state** (`ProjectPaths.root`, 0700
   directories, 0600 files), never under the skills root, and `/skill-review accept` moves the
   proposal folder only — it never reads, copies or moves anything from the state directory.

The in-repo precedent for this pattern is `CompressionHandler`
(`aget_state` + direct LLM call).

## 10. Relationship to the main codebase

Reused core services:

| Service | Usage |
|---|---|
| `initializer` (bootstrap singleton) | `app_paths` (home/config layout: the component config dir and prompt root), `get_project_paths(working_dir).root` (the private state that holds `trajectories/` and `skill-evolver/decisions/`), `load_llm_config`, `llm_factory`, `get_checkpointer`, cached skill catalog |
| `core.paths.AppPaths` | `AppPaths.resolve().config_dir` as the default config dir of `load_skill_evolver_config` (the loader must not import `initializer`, so `exgraph` can import it lazily) |
| `ConfigRegistry` (indirect) | Model alias resolution for the session's LLM |
| `LLMFactory` | Analysis model = the session's configured model; its `LLMConfig.context_window` (default `None`) feeds `ContextBudget` |
| `SkillFactory.parse_frontmatter` | Frontmatter parsing inside the validator, so "valid" means "the skill loader reads the same data" |
| `trajectory_recorder.export` / `reader` | `resolve_trajectories_dir`, `find_trajectory_file`, `load_trajectory`, `load_trajectories` — read-only access to the recorded JSONL |
| `exgraph` | `exgraph_context.attach_stored_graph` appends the stored experience graph to the classify prompt (context, never citable); `exgraph/enrich.py` consumes `features.extract_episodes(traj)` without the demo flag and keys Episode nodes by `primary_seq` |
| `utils.compression.calculate_message_tokens` | Token budgeting for the (legacy) replay |
| `/threads` thread-listing SQL | Reused verbatim in `latest_other_thread` |

Deliberately untouched subsystems: graph assembly (`AgentFactory`), middleware stack,
approval/HITL, MCP, checkpointer write paths. The command is a pure CLI-layer feature;
removing its registration and the seeding call detaches it completely.

Feedback loop: proposals do **not** re-enter the platform on their own. A human promotes
a proposal by moving it into `<root>/<category>/` (or by replacing the library file it
revises); from then on the standard discovery path applies (`SkillFactory` scan →
`_FilteredSkillsMiddleware` per-turn refresh → `/skills` and slash shortcuts).

## 11. Key design decisions

- **One pipeline, two thin commands.** `pipeline.run_thread` is the only implementation of the
  stage sequence; `/skill-mine` and `/direct-skill-generation` parse flags, load config and
  prompts, gather evidence and print summaries. A rule that exists once cannot drift between
  the commands.
- **Direct LLM calls, no tools, a hard budget.** Classify (1 + 1 corrective), the optional
  expand round (1 + 1), and per plan render (≤ 2) + review (≤ 2) + revise (1) + review (1):
  at most `2 + 2 + 6 × max_plans` calls per thread, capped by `max_llm_calls_per_thread`
  through `CountingLlm` (section 19). Deterministic cost and latency, no headless-interrupt
  handling; the library inventory is injected as a programmatic `{skill_library}` snapshot
  instead of a tool call.
- **Evidence, not transcript.** The model sees code-extracted episodes (section 15), never the
  session; every candidate cites fragment ids the bundle contained, every decision cites
  episode ids the bundle showed, and AI narration is never evidence.
- **Policy as one inserted block, never in the fixed text.** The fixed prompt texts carry no
  selection criteria; the effective selection chooses exactly one `{selection_policy}` block,
  the effective demo flag the `{render_policy}` / `{review_policy}` texts (section 18). A
  prompt can therefore never demand non-obviousness and allow triviality at the same time
  (`test_policy.py::test_demo_assembly_has_no_novelty_requirement`).
- **Demo lowers novelty, never evidence.** The overlay bypasses the score gate
  (`demo_override`), extracts `observed_procedure` chains and allows a duplicate of a library
  skill as a separate `demo-<task-class>` proposal; everything about evidence, structure,
  safety, secrets and budgets stays exactly as in an ordinary run, and the proposal is inactive
  like any other (section 18).
- **Code decides what is a skill.** `validate_skill_md` rejects what the render prompt
  forbids; the semantic reviewer compares the file with the candidates and evidence; the model
  gets exactly one corrective turn per failure kind, and a second failure ends the plan with
  the error list instead of a repaired file.
- **Proposals, not library writes; a report, always.** The user promotes a proposal by hand;
  `provenance.json` next to it (policy, config, prompt hashes, review, verification level) and
  the decision report in the private state make a bad draft — or an empty run — debuggable.
- **Prompt-as-data with a contract.** The methodology lives in versioned, user-overridable
  Markdown that declares its contract and is hashed into every proposal; Python holds only the
  pipeline and refuses a stale contract loudly.
- **Fail loud on configuration.** Every config problem is reported at once with its dotted
  path; nothing degrades to a default silently. A v1 file is migrated in memory, not rewritten.
- **Component isolation.** Own home folder, own config file, own seeding, own private state
  folder — zero coupling to the config-migration framework at this stage.
- **Copy-if-missing seeding.** First launch of a new build materializes everything;
  user customizations survive upgrades, and the run-time contract check catches the ones that
  need migrating.

## 12. Limitations and future work

- **Updates are applied by hand.** An `update` proposal is the full revised text of the
  existing skill (its `name` is enforced by the validator, `target.base_sha256` records the
  file it was written against); the user replaces the library file after review. The model
  cannot inspect or modify the library: no skill tools are exposed, the programmatic
  `{skill_library}` snapshot is its only view. The candidate mechanism for an agentic evolver
  (a **thread fork** via `graph.aupdate_state` plus filesystem tools) is unchanged and still not
  implemented.
- **What the reviewer does not prove.** A `pass` from the semantic review means "faithful to
  the candidates and the evidence shown", not that the procedure was executed or works in
  another environment; `verification.level` is always `evidence_supported` with the note `not
  executed by generator`. The second level, `execution_verified`, is defined
  (`VERIFICATION_LEVELS`) but never written by the generator: it would require a real run on a
  prepared fixture in an isolated environment, which only the test suite does (section 21).
  A `reference` candidate is likewise the classifier's claim: the pipeline reads the referenced
  skill's text and records `coverage: text_read | unverified`, it never confirms coverage.
- **Unreadable trajectory, no report.** An empty or corrupt trajectory file raises
  `TrajectoryReadError`; both commands print `Trajectory unreadable (…); the LLM was not called`
  and create no LLM, but no decision report is written because no thread could be built. (A
  file whose damaged lines still parse to zero turns does reach the gate and gets a report.)
- **Context budget only when the window is known.** `ContextBudget` enforces the prompt size
  only when `LLMConfig.context_window` is configured; the default `None` enforces nothing. The
  constants (4 chars per token, 60 % usable, 2048 reply tokens) are starting values, not
  measurements (section 19).
- **Legacy replay path.** `_generate_skill_md` and its helpers, the `Nothing to save.`
  sentinel, `prompts/default/prompt_v1.md` and the `active` field are kept in the module but
  `handle()` no longer calls them (user decision, 2026-09-04: remove once the evidence
  pipeline has proven itself).
- **One render per plan.** Every accepted candidate of a plan goes into one `SKILL.md`; the
  existing skill text is passed only for the `update` plan of one skill. Candidates past
  `max_plans_per_thread` are deferred, never merged.
- **Validator gaps (deliberate, spec-literal):** deepagents' extra name rules (no trailing
  `-`, no `--`), non-empty `## Inputs` / `## Outputs`, a mandatory H1 title and the phrase
  "doesn't work" are not checked. The `^\d+$` task-id rule is unreachable behind the
  `^[a-z]…` name pattern and is kept as documentation. Whether a prohibition such as "Never
  use `--force`" is evidence-backed is the reviewer's call (`unsafe_claim`), not the
  validator's.
- **Secret patterns are patterns.** The `credential assignment` rule (`password|passwd|
  secret|token|api[_-]?key [:=] <value ≥ 8 chars>`; the key may be prefixed or suffixed —
  `GITHUB_TOKEN=`, `DB_PASSWORD=`, `access_token=`, `AWS_SECRET_ACCESS_KEY=` — and may be
  JSON-quoted, `"password": "…"`, the shape in which bundle excerpts render tool arguments) can
  still flag a lowercase word that ends the line, such as `secret: configuration`, in a
  hand-written skill; because `/skill-review accept` re-validates, such a file is refused until
  reworded. Placeholders (`<…>`, `$…`, `{…}`, words like `your` / `example` / `placeholder` /
  `redacted` / `xxx`, bare `ENV_NAMES`, each also when wrapped in markdown backticks) pass, and
  so does Inputs prose: after `<key>:` (never after `=`), a lowercase word that ends a clause or
  is followed by more words (`- password: required, prompted interactively`, `- token: obtained
  from the CI settings page`) is not a value.
- **Classify coverage is advisory.** The prompt says every shown `E<n>` should appear in a
  decision; code rejects only *unknown* ids, so the report may show episodes no decision
  mentions. A quoted `contract_version: "2"` is a schema violation (one corrective retry).
- `mine_cross_session` loads the newest `cross_session_limit` (20) trajectories of the
  agent on every run; a corrupt neighbouring file (`TrajectoryReadError`) aborts the
  command loudly instead of being skipped.
- `.proposals/` under the repository's own `skills/` (when the CLI runs with the repo
  as working dir) is not git-ignored; adding it to `.gitignore` is the user's call.
- The component config is outside `VersionedConfig`; a future schema_version 3 will need the
  same in-memory migration treatment v1 received.
- **One fragment per event.** When two episodes cite the same event, the bundle shows it
  once, with the cut chosen by the episode rendered first; the other episode's block repeats
  that line. Its own structured facts are still in its block, so nothing is lost, but a second
  cut of the same event is not shown.
- **exgraph node ids changed** with `primary_seq` (e.g. `user_correction:2` → `:17` on the
  signals fixture); shards are rebuilt per `remember_thread`, so there is no migration, but
  ids saved in older artifacts differ.
- **Real-LLM smoke test.** Section 21 records whether the synthetic demo thread was run
  against a real model; until then the scripted-LLM tests are the only end-to-end evidence.

## 13. Operational notes

- Development runs use the editable install (`uv pip install -e .`); wheel rebuilds
  require `pip install --force-reinstall --no-deps dist/<wheel>` because the version
  number does not change between local builds.
- After hand-merging changes, run `ruff check --select F821,F401 src/` (or
  `pre-commit run`) — undefined-name regressions in this component have historically
  been the dominant failure mode.
- Seeding can be exercised against a clean home with
  `MSAGENT_HOME=$(mktemp -d) msagent config --show`.
- The full component check is `pytest tests/ut/skill_evolver tests/ut/trajectory_recorder
  tests/ut/cli tests/ut/exgraph tests/it/exgraph -q`; tests import
  `msagent.cli.handlers` before the evolver modules (the pre-existing import cycle noted in
  `test_direct_skill_generation.py`).
- A stale user prompt or a broken config shows up as the `report_config_error` block at the
  start of a run; the fix is always in `~/.msagent/config/config.skill.evolver.yml` or
  `~/.msagent/skill-evolver/prompts/`, never in the package.

## 14. Deterministic feature extraction (no LLM)

Candidate discovery is code, not prompt. `src/msagent/skill_evolver/features.py` turns one
`Trajectory` (the typed reader model of `msagent.trajectory_recorder`) into `Episode` records,
`group_incidents()` joins the episodes that describe the same events, `evidence_score()` adds
the incidents up and `gate_decision()` says whether — and why — a thread reaches the LLM stage.
The module is stdlib only — importing it must not load langchain, which
`tests/ut/skill_evolver/test_features.py` enforces in a subprocess — so it runs in tests and CI
without an LLM, and every detection is reproducible. The rules below are `FEATURES_VERSION` 4.

```python
@dataclass(frozen=True, slots=True)
class EvidenceItem:
    ref: EvidenceRef          # file name + physical line (+ seq for display), see the recorder doc
    role: str                 # what the event is to the episode: error, fixed_call, task, step, result, ...
    required: bool            # part of the minimum without which the episode cannot be shown
    snippet: str | None = None   # content-bearing cut chosen by the detector ("…"-marked);
                              # None = the bundle shows the head of the recorded event

@dataclass(frozen=True, slots=True)
class Episode:
    kind: Literal["error_recovery", "user_correction", "retry_loop", "approval_denied",
                  "repeated_procedure", "skill_gap", "observed_procedure"]
    thread_id: str
    source: str               # file name of the episode's own thread
    evidence: list[EvidenceItem]   # never empty, refs unique, >= 1 required, >= 1 of `source`
    tool_sequence: list[str]
    facts: dict[str, Any]     # JSON-safe, kind-specific details; text cut to 200 chars, cuts marked
    weight: float             # 0.0..1.0 (0.0 only for observed_procedure)
    anchors: list[str] = field(default_factory=list)   # "<run_id>#<seq>" of the events the
                              # episode is about, never of context events; [] for skill_gap
    evidence_seq -> list[int] # property: sorted seqs of the own-source items (tables, display)
    primary_seq -> int        # property: smallest own-source seq among the *required* items;
                              # the identity exgraph keys Episode nodes on (stable when
                              # context items are added); falls back to evidence_seq[0]

@dataclass(frozen=True, slots=True)
class DetectorNote:           # why a detector left a candidate out (dry run, decision report)
    detector: str
    reason: str               # a DROP_* constant
    detail: str
    seqs: list[int] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class GateDecision:
    incidents: list[list[Episode]]
    score: float
    passes: bool
    reason: str               # GATE_NO_EPISODES | GATE_SCORE_REACHED | GATE_STRONG_CORRECTION |
                              # GATE_SCORE_BELOW | GATE_DEMO_OVERRIDE ("demo_override")
    detail: str = ""          # demo only: which kinds have required evidence, or
                              # GATE_DEMO_NOT_APPLICABLE
    describe() -> str         # "reason" or "reason; detail" — what every table and line prints

FEATURES_VERSION = 4              # recorded in provenance.json; bumped with any rule or weight
DEFAULT_MIN_EVIDENCE_SCORE = 1.0  # gate threshold when the config sets none
REQUIRED_ROLES = {"error_recovery": {"error", "fixed_call", "result"}, "user_correction": {"correction"},
                  "retry_loop": {"attempt"}, "approval_denied": {"approval"},
                  "repeated_procedure": {"step"}, "skill_gap": {"first_call", "user_message"},
                  "observed_procedure": {"task", "step", "result"}}
GATE_DEMO_OVERRIDE = "demo_override"
GATE_DEMO_NOT_APPLICABLE = "no episode has complete required evidence"
OBSERVED_MAX_CHAINS = 3; OBSERVED_MAX_CALLS = 5; OUTCOME_HEAD = 60; OUTCOME_TAIL = 139

extract_episodes(traj, *, skill_index=None, demo=False, notes=None) -> list[Episode]
mine_cross_session(trajs, *, min_support=2) -> list[Episode]   # repeated_procedure only
classify_approval(request, decision) -> ApprovalVerdict        # one approval, read by structure
has_required_evidence(episode) -> bool                         # required items cover REQUIRED_ROLES
group_incidents(episodes) -> list[list[Episode]]               # connected components over anchors
evidence_score(episodes) -> float                              # sum over incidents of max(weight)
gate_decision(episodes, *, min_score, demo=False) -> GateDecision
expand_episode_context(episodes, traj, *, surrounding_events, only=None) -> list[Episode]
```

| Kind | Weight | Rule (v4, one private `_detect_<kind>` each) | Facts | Anchors · Evidence (**required** / context) |
|---|---|---|---|---|
| `error_recovery` | 0.6 | inside one stream (turn group × subagent, see *Context streams*): a `status == "error"` call followed within `RECOVERY_WINDOW` (5) calls of the same stream by an `ok` call of the same tool; the first such `ok` call decides — a non-empty raw argument diff is the recovery, identical arguments (transient failure, or calls recorded without `tool.start`) are nothing; same-tool `error` / `orphan` calls in between are skipped. A following `dispatch` turn, another subagent and an orphan never recover; two failures sharing one recovery give two episodes (two diffs) | `tool`, `error_type`, `error` (`error` or, for a `tool.result` with `status=error`, its `output_text`; head 100 + tail 180, the cut marked), `args_diff` (`added` / `removed` clipped; `changed{old,new}` windowed on the change: common prefix and suffix dropped, 60 chars of context each side), `calls_between`, `subagent` | anchors: the failed call and the recovery · **`error`** (end of the failed call, snippet = the error text), **`fixed_call`** (start of the ok call), **`result`** (its end) / `failed_call` (start of the failed call), `task` (the user message that opened the turn group; v4 context) |
| `user_correction` | 0.9 strong / 0.5 weak | adjacent turn groups: the head of the later group has a `user_message` containing a `STRONG_CORRECTION_MARKERS` or `WEAK_CORRECTION_MARKERS` phrase (en + zh, case-insensitive, at a word start unless the marker opens with a Han character; the negations `不，` / `no,` only when they open the message) **and** the group's actions differ from the previous group's — a tool added or removed (name sets over all calls of each group, any subagent) or a tool used in both groups whose last call before and first call after differ in normalized arguments. A marker without an observed change is nothing; a head without a user message (`resume`) cannot correct, and the prelude turn is never the corrected group. `strength` is `strong` when a strong marker is present, else `weak` with `WEAK_CORRECTION_WEIGHT` | `correction_text` (the window of 120 chars each side of the first marker in the whitespace-collapsed message, cuts marked — a long message keeps the phrase, not its head), `strength`, `markers`, `tools_before`, `tools_after`, `changes{tools_added, tools_removed, args_changed{tool: diff}}` (diffs windowed on the change), `run_id_before`, `run_id_after` | anchor: the correcting turn (`turn.start` of the head) · **`correction`** (that turn, snippet = the marker window) / `corrected_turn`, and per tool in `args_changed` a `before_call` and an `after_call` |
| `retry_loop` | 0.7 | inside one stream: calls chained by `(tool name, work object)` — a call recorded without arguments has no object and never chains; ≥ `RETRY_MIN_ATTEMPTS` (3) attempts (orphans count) with ≥ 2 distinct normalized argument sets **and** either a failed attempt (`status == "error"`) or variants differing in a `SEARCH_KEYS` key (`pattern` / `query` / `regex`). Reading three files is a fan-out and a parameter sweep that never failed is not a loop | `tool_name`, `work_object`, `attempts`, `reason` (`failed attempt` / `search key varies`), `args_variants` (normalized), `statuses`, `run_id` (turn of the first attempt), `outcome` (v4: the clipped result or error of the last attempt) | anchors: every attempt · per attempt **`attempt`** (start) and its **`result`** / **`error`** (end, error snippet = clipped error text) — required for the first and the last attempt, context in between; `task` (context, v4) |
| `approval_denied` | 1.0 | every `Approval` of a turn is read by `classify_approval(request, decision)`: `denied` is an episode naming the rejected actions only, `approved` is skipped, `unknown` is logged at debug level with its reason and skipped. Context: the calls of the approval's turn with `seq_start > approval.seq` plus the calls of the following turns of the **same group** (the `resume` continuation of an interrupted turn, never the next `dispatch` turn), capped at `DENIAL_CONTEXT_CALLS` (3), any subagent — `Approval` has no subagent field | `interrupt_id`, `run_id`, `tools` (names of the rejected actions), `denied_actions` (`verdict.denied`), `request`, `decision`, `next_tools` | anchor: the approval · **`approval`** / `next_call` ×≤3 |
| `skill_gap` | 0.4 | unchanged since v2: domain tools (anything but `get_skill` / `fetch_skills` / `get_tool` / `fetch_tools` / `run_tool`) were used, `skills_consulted` is empty, and the BM25 top hit of *user messages + tool names* against the library scores ≥ `SKILL_GAP_MIN_SCORE` (1.0). A description-fix candidate, not a new skill. Needs a `BM25Index` from `retrieval.py` (stdlib BM25 over `name + description`; the caller builds `SkillDoc(skill.display_name, skill.description)` — `features.py` never imports `msagent.skills`) | `candidate_skill`, `score`, `matched_terms`, `domain_tools` | no anchors — a trajectory-level observation · **`first_call`**, **`user_message`** (the first) / `user_message` (the rest) |
| `repeated_procedure` | 1.0 | `mine_cross_session` only: steps are *segments* of one stream — catalog calls are dropped and a call with `status != "ok"` closes the segment, so a repeated failure is never a procedure; tool-name n-grams (n = 2..5) inside segments present in ≥ `min_support` **distinct** `thread_id`s (`min_support < 2` raises). Only closed patterns are reported — a sub-n-gram with the same support as a longer one is dropped, so a shared five-step procedure is one episode, not ten. The episode belongs to the first supporting trajectory in input order and cites the steps of that trajectory **and** of the lexicographically first other supporting thread — proof from two sessions — while `support` counts every supporting thread. It says that several sessions issued these calls in this order and each returned `ok`, never that the task succeeded | `ngram`, `support`, `thread_ids` | anchors: the own-thread steps · **`step`** of both threads (the bundle indexes the second trajectory too) / per own-thread step its `result` (v4; collapses into the step when the call was recorded without `tool.start`), the owner group's `task` (v4) |
| `observed_procedure` | **0.00** | **demo only** (`extract_episodes(..., demo=True)`), always extracted, not only when nothing else fired: the segments of `_procedure_segments(min_len=1)` — consecutive `ok` domain calls of one stream — are cut into disjoint windows of at most `OBSERVED_MAX_CALLS` (5) calls **from the start** of the segment (the head of a procedure carries its inputs); at most `OBSERVED_MAX_CHAINS` (3) windows per trajectory become episodes, in model order. A window is dropped for the first problem found, in this order: the group head has no user message (`no_task_context`; a previous group's message is never borrowed), a call was recorded without `tool.start` (`missing_tool_start`; its arguments are unknown), any step's output matches an `ERROR_MARKERS` regex (`error_in_output`; the ten patterns, each compiled on its own so an inline `(?i)` is per pattern: `Traceback \(most recent call last\)`, `\b[A-Z][A-Za-z0-9_]*(?:Error\|Exception)\b`, `(?i)\berror:`, `(?i)\bfatal:`, `\bFAILED\b`, `No such file or directory`, `command not found`, `Permission denied`, `(?i)\b(?:exit code\|exit status)\s+[1-9]\d*\b`, `(?i)\bnon-zero exit\b` — no bare `error`, so `0 errors` is not a marker), the last output is empty (`empty_result`); plus `chain_limit`, `duplicate_chain` (a defensive invariant — windows are disjoint) and `no_completed_calls` (no segment at all: an assistant saying "done" without a completed call yields nothing; **AI messages are never read**). Every drop is a `DetectorNote` for the dry run and the report. `status: ok` alone is never proof — the observed output is | `task` (the user message), `run_id`, `subagent`, `chain_index`, `calls`, `steps[{tool, work_object, args, result}]`, `outcome` (last output, head 60 + tail 139), `verification` (`observed output of <tool>: <outcome>`) | anchors: `run_id#seq_start` of every step (they may share an incident with a `retry_loop` / `error_recovery` of the same calls — the score does not change, the max weight wins) · **`task`** (the group's `turn.start`), per call **`step`** (start) and **`result`** (end, snippet = clipped output) — all required, the episode is a complete package or nothing |

Design points:

- **Evidence is real and addressable.** Every `EvidenceItem.ref` is the file name and physical
  line of one source event (`EvidenceRef`, built by `Trajectory.event_ref`; the recorder doc,
  section 9) and every anchor a `<run_id>#<seq>` of one; a property test over all fixtures
  re-reads each cited line and checks it holds that very `seq`. `seq` is display only: it
  restarts under a new `rec` after a process restart, which is why refs carry the line, anchors
  the `run_id`, and ordering is model order (turns in file order, spans in order), never
  trajectory-wide `seq` sorting. `Turn.approvals` are typed `Approval` records that keep their
  `seq` and `line` for this reason.
- **Required minimum and content-bearing cuts.** Each item has a role and a `required` flag: the
  required items are the minimum without which the episode is not worth showing (the table
  above); `REQUIRED_ROLES` names the roles each kind must carry as required items and
  `has_required_evidence()` checks them — the demo gate relies on that check. The evidence
  bundle trims context items under budget and excludes an episode whose required items do not
  fit (section 15). Text copied into facts or snippets is cut around what matters and every cut
  is marked with `…`: `_window` keeps `MARKER_CONTEXT` (120) chars around the correction marker,
  `_diff_window` drops the common prefix and suffix of a changed argument and keeps
  `DIFF_CONTEXT` (60) chars around the change, `_clip_edges` keeps the head (100) and tail (180)
  of an error, `_clip_outcome` the head (60) and tail (139) of an observed output, `_clip` cuts
  fact values at `VALUE_LIMIT` (200). Nothing absent is reconstructed: an empty result stays
  `(no output)` in the bundle, an orphan stays `orphan`.
- **Context items (v4 enrichment).** `error_recovery`, `retry_loop` and `repeated_procedure`
  cite the user message that opened their turn group as a non-required `task` item, and
  `retry_loop` / `repeated_procedure` cite the results (or errors) of their calls, so the
  classifier sees what was asked and what came back, not only the calls. Context never changes
  weights, anchors or scores. Because a `task` item at `seq 2` would give two `error_recovery`
  episodes of one turn the same `evidence_seq[0]`, consumers that need an identity — the
  exgraph Episode nodes in `exgraph/enrich.py` — key on `Episode.primary_seq` (the smallest
  *required* own seq), which enrichment cannot move (`test_p1_enrich.py::
  test_two_recoveries_in_one_turn_are_two_episode_nodes`).
- **Context streams.** Calls are compared inside one execution context: a *turn group* — a turn
  plus the adjacent turns with `source == "resume"` that continue it (the `/threads` path: no
  user message, a fresh `run_id`, no link field; adjacency plus `source` is the only continuation
  the recorder expresses) — split by `call.subagent` (`None` = root). `error_recovery`,
  `retry_loop`, the procedure segments and the observed chains never cross a following
  `dispatch` turn or a subagent boundary; `user_correction` compares whole groups (any
  subagent); the denial context runs into the `resume` turn of the same group only.
- **Normalized arguments and the work object.** `_normalize_args` drops `VOLATILE_KEYS`
  (`offset`, `limit`, `timeout`, `timeout_ms`), collapses whitespace and runs `PATH_KEYS` values
  through `posixpath.normpath` (`./cfg/dev.yml` == `cfg/dev.yml`); non-string values pass through.
  The *work object* of a call is the program plus the first non-option token of the first
  `COMMAND_KEYS` string (`msprof --collect train.py` → `msprof train.py`; `python a.py` ≠
  `python b.py`; a command wins over the `cwd` it runs in, which is context, not the object),
  else the first `PATH_KEYS` string, else `""` — the tool itself; a call recorded without
  arguments has none. `user_correction` and `retry_loop` compare normalized forms;
  `error_recovery` keeps the raw diff, because the diff is the knowledge.
- **Structured approval verdicts; unknown is unknown.** `classify_approval` matches
  `{"decisions": [...]}` by index against `request["action_requests"]` (a length mismatch is
  `unknown`; one decision without such a list applies to the flat request), a flat `{"action" |
  "type": ...}` dict against the request, and a bare legacy string only when it equals one of
  `LEGACY_DENIAL_ANSWERS` / `LEGACY_APPROVAL_ANSWERS`. `reject` denies; `approve` / `edit` /
  `respond` do not; any other type, free text (`No, don't`, `Cancel`), `None`, numbers and lists
  are `unknown` with a non-empty `reason` — never an assumed denial, so `approve` with the
  comment `no issues` is not a denial and a `no` in free text counts for nothing.
- **Incidents.** `anchors` are the keys of the events an episode is *about* (its calls, its
  correcting turn, its approval — never context events). `group_incidents` joins episodes that
  share an anchor (union-find over the list; an anchorless `skill_gap` is keyed on
  `kind@thread:evidence_seq`, so only an exact duplicate joins it), ordered by first episode.
  `evidence_score` is the sum over incidents of the heaviest episode in each: the two
  `error_recovery` and the `retry_loop` of one msprof chain count 0.7 once, independent incidents
  add up, and a duplicate episode changes neither the score nor the gate. An
  `observed_procedure` that shares its calls with a retry or recovery joins their incident at
  weight 0.0 and changes nothing. The signals fixture yields 6 episodes, 4 incidents and a
  score of 3.0 (v1 summed the weights); the demo fixture yields no scoring episode at all and
  exactly one `observed_procedure` (`test_features.py::EXPECTED_KINDS`, `EXPECTED_DEMO_CHAINS`).
- **Gate.** `gate_decision(episodes, *, min_score, demo=False)` checks, in this order:
  `no episodes` (first, so an empty thread never reaches `build_evidence_bundle`, even with
  `min_evidence_score: 0`); `score >= min_evidence_score`; `strong user correction` — a
  `user_correction` with `strength == "strong"` passes while `min_score <=
  DEFAULT_MIN_EVIDENCE_SCORE` (1.0), because one explicit correction with an observed change of
  action (weight 0.9) is worth the analysis at the default settings and a stricter threshold opts
  out of the rule; then, **only with `demo=True`**, the override: when at least one episode
  `has_required_evidence()` the thread passes with `reason = "demo_override"` and
  `detail = "<kinds sorted, comma-joined> has|have required evidence"` (e.g. `demo_override;
  observed_procedure has required evidence`), otherwise it is refused with the ordinary
  `score < min_evidence_score` reason and `detail = GATE_DEMO_NOT_APPLICABLE`; finally
  `score < min_evidence_score`. Score, weights and incidents are never altered by demo, and
  with `demo=False` the output is byte-identical to earlier versions. A weak correction (0.5)
  never admits a thread on its own. Passing the gate admits the thread to the LLM stage;
  whether a rule is worth keeping is decided there. The same call, with the same `demo`, runs
  at the second gate over `bundle.kept`, after a context cut and after the expand round.
- **Expand round primitive.** `expand_episode_context(episodes, traj, *, surrounding_events,
  only=None)` returns the episodes of `traj` with, for every cited call, the
  `surrounding_events` calls before and after it inside the same stream as non-required
  `context_call` / `context_result` / `context_error` items, the call's own missing result, and
  the group's user message as a `task` item; `surrounding_events == 0` adds only the missing
  results and the task, a negative value raises. `only` restricts the expansion to the listed
  indices (the pipeline passes the episodes named in the `insufficient_evidence` decisions).
  An episode that gains nothing is returned as the same object; the second thread of a
  `repeated_procedure` is never touched; anchors, facts and weights are unchanged — nothing is
  invented.
- **Known imprecisions**, documented rather than fixed:
  - markers are literal: `不，谢谢，我自己来` opens with a negation and, when the agent
    then does nothing, is a strong correction (the previous tools were "removed").
  - `error_recovery` links by tool name and context, not by work object: `bash pytest` error →
    `bash ls` ok is a recovery.
  - the work object of a command is the program plus its first positional token: `pip install X`
    and `pip install Y` are one object, `msprof --output ./prof train.py` → `msprof ./prof`.
  - `grep` without a `path` has the object `""`: three unrelated searches in one group are a
    `retry_loop`.
  - the denial context is not filtered by subagent (`Approval` has no such field).
  - the marker window is cut around the *first* marker in table order (strong first), not the
    earliest in the message; a message with several corrections shows one of them in full.
  - `skill_gap` fires on a single distinctive shared term in libraries of four or more skills
    (unchanged from v1).
  - the error-marker check of `observed_procedure` is textual: an output that quotes an error
    message while succeeding drops the chain (`error_in_output`), a failure printed without any
    marker passes it — the semantic review is the second line of defence.

Additions beyond the detector rule list, each one line to remove: a marker opening with a Han
character is matched without the word-start guard (`(?<!\w)` blocks `不对` inside `这不对`, as
Chinese is not space-delimited), and `retrieval.tokenize` indexes a Han run as overlapping
bigrams (the Lucene CJK scheme, no segmentation dictionary) with stopwords dropped from the
bigrams instead of cutting the run, so `性能` survives inside `文件的性能`.

Wiring (`_gather_evidence`, section 7): the thread's JSONL is located via
`export.resolve_trajectories_dir(state_dir=initializer.get_project_paths(ctx.working_dir).root)`
and `export.find_trajectory_file(dir, thread_id)`; **no file → `print_error` and return without
calling the LLM** (a thread without a recorded trajectory is refused, not passed through).
`pipeline.collect_episodes(current, others, *, skill_index, demo, notes)` runs
`extract_episodes` on the current trajectory and `mine_cross_session([current, *others])` over
the agent's newest `cross_session_limit` trajectories; with the current trajectory first in
that list, every shared pattern it supports belongs to the current thread, so the kept episodes
are exactly those with `thread_id == current.thread_id`. Each of them also cites one supporting
session's steps; `pipeline.supporting(others, episodes)` picks the trajectories those refs point
at and `_gather_evidence` returns `(current, supporting, episodes, notes)`, so the bundle indexes
`[current, *supporting]` (refs are unique per file and line, so several trajectories in one
bundle are unambiguous). `gate_decision(episodes, min_score=rules.min_evidence_score,
demo=rules.demo)` decides: a decision that does not pass ends the thread with an info message
naming the reason — `Nothing to save: no episodes detected in thread <id>`, or `Nothing to save:
evidence score X < min_evidence_score Y (N episodes, K incidents)` followed by `; no episode has
complete required evidence` under demo — and no LLM call; otherwise the episodes go to the
bundle → classify stage (section 15) and the candidates to the render stage (section 16).
`exgraph/enrich.py` keeps calling `extract_episodes(trajectory)` without the flag and never
sees `observed_procedure`.

Known gaps: files recorded before the `ignore_agent` fix have no `tool.*` events, so the
detectors see empty `tool_calls` there (no fallback to `AiMessage.tool_call_names` by design);
`record_approval` has no call site yet, so `approval_denied` fires only on fixtures until it is
wired.

Verification: `pytest tests/ut/skill_evolver -q` — detector positives and the mandatory negatives
("谢谢" is not a correction and a weak marker never makes a strong one; a fan-out over
different files, a parameter sweep that never failed and paging with a changing `offset` are not
retry loops; an `ok` in another subagent or in a following `dispatch` turn is not a recovery;
`approve` with the comment `no issues` is not a denial and an unreadable decision yields no
episode; catalog calls and failed calls are not procedure steps; an n-gram inside one trajectory
is not a procedure; an orphan-only chain, an AI claim of success and an `ok` output carrying a
traceback yield no `observed_procedure`; subagent streams and a recorder restart keep chains
apart), `classify_approval` over every recorded shape, the incidents-and-gate section (one
chain seen by several detectors counts once, a shared turn does not merge incidents, a
duplicate episode does not change the gate, the reasons, the strong correction rule, the demo
override only when the ordinary rules fail), the `skill_evolver_signals.jsonl` and
`skill_evolver_demo_success.jsonl` fixtures end to end, per-fixture kind and demo-chain counts,
the evidence property test (every ref re-reads to a line holding its `seq`, every anchor
resolves to a source event), the roles and required items of every detector, the windowed
cuts, `expand_episode_context` (neighbours, missing results and the task only), the
no-langchain/no-network subprocess probe.

## 15. Evidence bundle and JSON classification (LLM stage, library only)

The replay of the whole session (section 8) has been replaced by two stages that give the model
only evidence and get a structured answer back; both are wired into `run_thread` (section 7).

**`bundle.py`** — `build_evidence_bundle(episodes, trajectories, *, max_chars=30000,
excerpt_chars=EXCERPT_LIMIT, demo=False) -> EvidenceBundle(text, shown, episodes, episode_ids)`.
One markdown block per episode, heaviest first (stable for equal weights); under `demo` the
`observed_procedure` blocks come **first** and the rest keeps the weight order — they carry the
admission and are short, so a heavier episode's optional excerpts can never starve their
required ones. The pipeline passes `max_chars=bundle_max_chars`, `excerpt_chars=
excerpt_max_chars` and `demo` from the effective rules.

```
### Episode E1 — observed_procedure (weight 0.00, thread thread-d)
Observed procedure: 3 call(s) in one execution context (selected by code); ok status is not proof of task success
Tools: read_file, bash, bash
Facts:
- task: "Compute the sum of the amount column in input/sales.csv and check that the total is 60"
- run_id: "run-d1"
- subagent: null
- chain_index: 1
- calls: 3
- steps: [{"args": {"path": "input/sales.csv"}, "result": "id,amount\n1,10\n2,20\n3,30\n", "tool": "read_file", "work_object": "input/sales.csv"}, {"args": {"cmd": "python3 -c \"import csv,sys; print(sum(float(r[sys.argv[2]]) …
- outcome: "TOTAL_OK"
- verification: "observed output of bash: TOTAL_OK"
Excerpts:
- [ev1] user: "Compute the sum of the amount column in input/sales.csv and check that the total is 60"
- [ev2] tool.start read_file: {"path": "input/sales.csv"}
- [ev3] tool.result read_file (ok): id,amount 1,10 2,20 3,30
- [ev4] tool.start bash: {"cmd": "python3 -c \"import csv,sys; print(sum(float(r[sys.argv[2]]) for r in csv.DictReader(open(sys.argv[1]))))\" input/sales.csv amount"}
- [ev5] tool.result bash (ok): 60.0
- [ev6] tool.start bash: {"cmd": "python3 -c \"import csv,sys; total=sum(float(r[sys.argv[2]]) for r in csv.DictReader(open(sys.argv[1]))); print('TOTAL_OK' if total == float(sys.argv[3]) else f'MISMATCH {total}')\" input/sales.csv amount 60"}
- [ev7] tool.result bash (ok): TOTAL_OK

### Episode E2 — error_recovery (weight 0.60, thread thread-s)
Tools: bash, bash, bash
Facts:
- error: "msprof: unknown option --collect"
- args_diff: {"changed": {"cmd": {"new": "msprof --application train.py --output ./prof", "old": "msprof --collect train.py"}}}
Excerpts:
- [ev10] tool.error bash (error): msprof: unknown option --collect
- [ev11] tool.result bash (ok): profiling done
- [ev6] tool.start bash: {"cmd": "msprof --application train.py --output ./prof"}
```

No transcript and no chronology: facts come from the episode, excerpts are whitespace-collapsed
cuts (`excerpt_chars`, default 300) of the cited events — the detector's snippet when it chose
one (the marker window of a correction, the head and tail of an error or of an observed
output), else the head of the recorded event — resolved through an index keyed by
`EvidenceRef` built from the typed model of every trajectory passed in (turn starts, tool
starts and results, AI messages, approvals). A `tool.result` with `status=error` renders its
`output_text` as the error. Facts are re-clipped to 800 chars because
`approval_denied.decision` and list-valued facts are unbounded. A `repeated_procedure` block
adds `Support: N threads (counted by code); excerpts from 2 of them`, an `observed_procedure`
block the `OBSERVED_LINE` shown above, and excerpts of another trajectory carry `(thread <id>)`.

Every block carries an episode id `E<rank>` (`EvidenceBundle.episode_ids` maps it back to the
`Episode`; the classify decisions cite these) and every excerpt line a bundle-local fragment id
`[evN]`, assigned in order of first appearance when its block is committed; one event has one
id and one text across blocks (a later block citing an event already shown repeats the line).
`shown` is the registry of exactly those fragments — id, `EvidenceRef`, role, `required`, text
— and `episodes` the outcome of every input episode (`BundleEpisode(episode, status,
episode_id)`):

- `shown` — the whole block fitted;
- `trimmed` — only the required excerpts fitted; the context ones are replaced by
  `- … N more events not shown`, which is not citable;
- `excluded` — even the required excerpts did not fit (`EXCLUDED_CODE =
  "insufficient_context_budget"`): the block is absent and has no `E` id, the episode's weight
  must not carry the thread to the LLM, and scanning continues with lighter episodes.
  `EvidenceBundle.kept` lists the shown and trimmed episodes; `pipeline.report_bundle` prints
  `Excluded <kind> (weight W): insufficient context for the bundle budget` per exclusion and
  runs `gate_decision(bundle.kept, demo=rules.demo)` again when anything was excluded, stopping
  with `Nothing to save: N episodes excluded from the bundle; remaining evidence score S <
  min_evidence_score M` (plus `; <detail>` under demo) before creating an LLM if it fails.
  Required items are never trimmed, so an unexcluded bundle is not re-gated. A bare event
  number is never citable, and nothing raises for size — an empty bundle is a legitimate
  outcome.

Loud failures: an episode whose source is not among the trajectories or whose ref does not
resolve (episodes and trajectories must be the same data), a non-positive budget, an
`excerpt_chars` that leaves no room beside the ellipsis. Data conditions are tolerated and
documented: a source seen twice keeps its first copy (as `mine_cross_session` does), the
reader's synthetic `unknown` and prelude turns never shadow the real record at the same line,
and a recorder restart that repeats a `seq` yields two refs and two fragments. The stored
experience graph appendix (`exgraph_context.attach_stored_graph`) is appended to `bundle.text`
after budgeting and carries no ids: context, never citable evidence, and not recorded in
provenance (it does count towards the context fit of section 19).

**`classify.py` — contract v2.** `classify(bundle_text, valid_refs, llm, template, *,
episode_ids, demo=False) -> Classification`, with the pydantic reply models

```python
class Decision(BaseModel):
    episode_ids: list[StrictStr]              # >= 1, the E<n> ids of the bundle
    decision: Literal["accept", "reject"]
    reason_code: Literal["accepted", "routine_activity", "insufficient_evidence", "transient_observation",
                         "already_covered", "no_transferable_rule", "unsafe_procedure"]
    explanation: str                          # <= 500 chars, a diagnostic, not a reasoning trace
    evidence_refs: list[StrictStr] = []       # accept ⇔ reason_code == "accepted" (validator)

class Candidate(BaseModel):
    title, rule, evidence_refs: list[StrictStr], future_applicability: high | medium | low,
    target: {action: create | update | reference, existing_skill}, applies_when, constraints,
    expected_outcome, covered_by: str | None = None, candidate_id (assigned by code)

class ClassifyResult(BaseModel):
    contract_version: Literal[2]
    verdict: Literal["save", "nothing"]
    candidates: list[Candidate]
    decisions: list[Decision]                 # >= 1

@dataclass(frozen=True, slots=True)
class Classification:
    verdict; candidates; rejected: list[tuple[Candidate, str]]; decisions; calls: int (1 or 2);
    model_verdict                             # what the model said before the code check
    insufficient_evidence -> bool             # model_verdict == nothing and a decision says so
```

The prompt is `skill-evolver/prompts/classify/prompt_v2.md` (123 lines; header of section 6):
how to read the bundle (`E<n>` are cited in `decisions`, `[ev<k>]` lines are the only citable
evidence; `Support:` / `Observed procedure:` lines, fact values and the graph appendix are
context), the seven kinds, `{skill_library}`, the **requirements under every policy** (every
claim grounded in cited `ev` ids, conditions never invented, one-time values parameterised,
`expected_outcome` only from observed results — `status ok` is not success —, unsafe
procedures rejected, no added commands/flags/versions/guarantees, AI narration is not evidence,
an empty library does not lower the bar), the `# Selection policy` section holding exactly one
`{selection_policy}` block (section 18 — the words novelty / non-obviousness / trivial /
routine appear only inside the blocks, never in the fixed text), the candidate fields
(`covered_by` included), the decision fields with the reason codes, `{evidence_bundle}` and the
strict JSON output contract. The reason codes are worded so that the fixed text never
contradicts a policy block: `already_covered` is a reject code — the library already states
the rule and the policy offers no `reference` or `update` candidate worth emitting — while a
`reference` / `update` candidate a policy asks for is an `accept` with `accepted` (the
`decision` line says "produced a candidate (including a `reference` or `update` candidate)",
and the validator forces `accept ⇔ accepted`). The module is stdlib + pydantic: the LLM is duck-typed (`ainvoke`
over `(role, text)` pairs, the reply read through `.text` / `.content`), so the
import-isolation probe covers it. Post-processing is code, not prompt:

- the reply is stripped of `<think>` blocks and of a whole-reply fence, parsed with `json.loads`
  and validated against `ClassifyResult` (a v1 body, a missing or quoted `contract_version`, an
  empty `decisions` list, an `accept` with a non-`accepted` code are schema violations);
- the **consistency check** `_contract_violations` then enforces the contract on the parsed
  reply: `verdict 'save' requires at least one candidate`; `verdict 'nothing' requires an empty
  candidates list`; `decisions cite unknown episode ids [...]; the bundle shows [E1, E2, ...]`;
  `decisions cite evidence ids not shown in the bundle: [...]`; and under demo `candidate '…':
  target.action 'update' is not allowed in demo mode; use 'create' and name the covering skill in
  covered_by`;
- **one corrective retry per round**, for either failure kind: the bad reply is replayed as an
  assistant turn followed by `_CORRECTION` (the parse error) or `_CONTRACT_CORRECTION` (the
  violations as bullets, plus the demo hint `and use target.action "create" for every
  candidate`); a second failure raises `ClassifyParseError("classify: reply is not valid JSON
  after one retry: …")` or `ClassifyContractError("classify: reply violates the contract after
  one retry: …")` — the reply is never repaired, the thread is failed (section 7);
- `StrictStr` refs, so a seq number, a boolean or a float never passes as a citation;
- **code-side rejection of candidates**: one with empty `evidence_refs` is rejected with `empty
  evidence_refs`; one citing any id outside `valid_refs` (an excluded event, a seq, an
  invention) with `evidence not shown in the bundle: [...]` — rejected candidates are returned
  in `Classification.rejected`, printed as `Rejected '<title>': <reason>`, recorded in
  provenance (`candidates_rejected`) and in the report as code rejections with the code
  `invalid_evidence`; never repaired. A rejected candidate is *not* a contract violation, so
  the other candidates survive;
- kept candidates get `candidate_id` `c1`, `c2`, … in reply order (the join key of render,
  review and provenance; titles can collide), their refs deduplicated in the model's order;
- no candidates left → verdict `nothing` while `model_verdict` keeps what the model said, so
  the pipeline can tell `(invalid_evidence)` from the classifier's own `nothing`.

The three rejection codes are `CODE_PARSE_ERROR = "parse_error"`, `CODE_CONTRACT_ERROR =
"contract_error"` (thread-level, raised) and `CODE_INVALID_EVIDENCE = "invalid_evidence"`
(per candidate). Guards raise before any LLM call: blank bundle, empty `valid_refs`, empty
`episode_ids`, template without `{evidence_bundle}`, template still holding `{skill_library}`
or `{selection_policy}`.

**The expand round.** `Classification.insufficient_evidence` is the signal for
`on_nothing.action: expand_context_once`: when the verdict is `nothing`, the model itself said
`nothing`, and at least one decision carries `insufficient_evidence`, the pipeline expands the
episodes those decisions name (`expand_episode_context(only=…, surrounding_events)`), rebuilds
the bundle with the same budgets and retries **exactly once** — and only when the new bundle's
shown refs are a **strict superset** of the previous round's (`Insufficient evidence: one more
classification round with N added events (K per side)`); otherwise it prints
`expand_context_once: no new context could be shown; not retried`. The second bundle is
re-gated and context-fitted like the first; a second `nothing` is final. `covered_by` is a
plain library name the render stage resolves (`resolve_library_skill`) into the provenance
`covering_skill` record; when it names no skill the name is recorded with null path and hash
and a warning is printed (`covered_by '<name>' names no library skill; recorded as the
classifier's claim`).

Verification: `tests/ut/skill_evolver/test_bundle.py` (ordering, demo ranking, trim-then-exclude
budget, scanning past an excluded episode, one id per event, `E` ids, `excerpt_chars`, excerpt
resolution and clipping, the correction window and the argument diff reaching the text, an
error in `tool.result.output_text`, distinct refs after a recorder restart, every shown fragment
re-read from its physical line across all fixtures including `malformed_lines.jsonl`,
cross-session blocks) and `test_classify.py` (scripted fake LLM: fences, retry, 24 schema
violations, every contract violation with its one retry, rejection reasons, candidate ids,
`covered_by`, optional conditions, guards, the v2 prompt contract, isolation).

## 16. Render, validation, review and proposals

The classify verdict is turned into a file by four small modules, all stdlib + pydantic (the
import-isolation probe in `test_validator.py` covers them, `review` and `policy` included):

**`render.py`** — `plan_render(candidates, skills, *, max_plans) -> RenderPlans(plans, deferred,
references, rejected)` splits a thread's kept candidates into render plans and never changes a
candidate's target: every `create` is its own `RenderPlan(candidates, existing=None)` (no semantic
merging in this version), every `update` of one library skill shares a
`RenderPlan(candidates, existing=<Skill>)`, updates of different skills are separate plans. `update`
and `reference` targets go through one resolver (display name, or a bare name unique across
categories): a valid `reference` lands in `references` (the pipeline reads the skill text and
reports `reference <skill>: <title> (classifier's claim; skill text read|unreadable, coverage not
verified)`, never "already covered"), an unknown or ambiguous target in `rejected` as
`PlanRejection(candidate, code, detail)` with `invalid_target` / `ambiguous_target` — never
silently turned into `create`. Plans keep the order of their first candidate; those past
`max_plans_per_thread` (default 3) are `deferred` with a reason, targets untouched, so every
candidate is in exactly one of the four lists.

Evidence selection has **no fixed cap any more**. `select_plan_evidence(candidates, evidence, *,
budget_chars=None) -> list[EvidenceSelection(fragments, omitted, missing_required)]` quotes every
cited fragment when no budget is given; under `budget_chars` pass 1 places the **required**
fragments of all candidates (citation order) and pass 2 the optional ones, each costing
`len(text) + EVIDENCE_LINE_OVERHEAD` (6), so a shortage drops context before proof. An optional
fragment that does not fit is `omitted`; a required one is `missing_required` and makes the
plan unusable — `render_skill_md` raises `InsufficientContextBudget` and `render_and_review`
returns the code `insufficient_context_budget` **before any LLM call** (also when the selection
is empty). The selection is recorded in provenance as `render_evidence` /
`render_evidence_omitted`, so an incomplete set is never marked complete.

`render_skill_md(candidates, *, llm, template, policy_text, existing_skill=None,
expected_name=None, taken_names=(), evidence=None, required_prefix=None,
evidence_budget_chars=None) -> RenderResult(content, validation, calls, transcript,
render_evidence, render_evidence_omitted)` renders **one plan**: it fills `{render_policy}`
(mandatory; `policy.render_policy_block(demo)`), `{candidates}` and `{existing_skill}` (formatted
text or `None. Create a new skill.`) in one regex pass, calls the duck-typed LLM, strips
`<think>` blocks and a whole-reply fence, normalises line endings and validates. On errors the
model gets exactly one corrective turn (`("ai", bad reply)` + the error list); the result of the
second attempt is returned as is. `revise_skill_md(previous, issues, *, llm, ...)` is the single
corrective turn after a failed semantic review: it appends `_REVIEW_CORRECTION` with the
reviewer's issues to `previous.transcript`, makes exactly one call and validates once.

`format_candidates(candidates, evidence, *, selections)` renders one numbered block per
candidate: the title and applicability, `Rule`, then — only when the classifier filled them —
`When` (`applies_when`), `Constraints` (bullets) and `Expected outcome`, the `Target`, the line
`Covered by library skill: <name> (write a separate teaching skill; do not copy the library
text)` when `covered_by` is set, and `Evidence:` bullets with the **text** of every selected
fragment plus `(N context excerpts omitted for the prompt budget)` when some were. Fragment
ids, seqs and thread ids never enter the payload — they belong in provenance, and the reply is
the user-facing SKILL.md. A candidate without conditions is rendered without them: nothing is
invented on the way to the file.

**Prompt** `prompts/render/prompt_v2.md` (341 lines): the role plus how to read `Evidence` lines
(`user:` the task, `tool.start <tool>: {…}` an action with its real arguments, `tool.result
<tool> (ok): …` the observed result, `tool.error …` a failure, `ai: …` the agent's statement —
not proof), the `# Selection policy` section holding `{render_policy}`, the task rules (revise
the existing skill and keep its name, or create a durable kebab-case name — the policy may
prescribe a prefix; state `## Inputs` with their format and the prerequisites; concrete actions
with the exact command shape, the expected result and how to verify it; **never claim that a
test or verification ran unless the evidence shows it ran**; replace one-time values with
`<parameter>` placeholders but keep durable file names, columns and conventions; never add a
command, flag, API, version, unit, dependency or guarantee of success the evidence does not
show; a one-operation procedure gets exactly one Workflow step; a prohibition must come from
the evidence), `{candidates}`, `{existing_skill}`, the `# REQUIRED SKILL.md STRUCTURE` section
carried over from the v1 prompt with three edits (one step allowed when the procedure is one
operation; `Prohibitions are allowed when the evidence shows the failure they prevent`; three
sentences reworded so the fixed text contains no novelty requirement), and the output contract
(the bare `SKILL.md`, no fences, nothing around it).

**`validator.py`** — `validate_skill_md(content, *, expected_name=None, taken_names=(),
required_prefix=None) -> ValidationResult(ok, errors)` collects every violation (the corrective
call needs the whole list) and repairs nothing:

| Rule | Check |
|---|---|
| frontmatter | parsed with `SkillFactory.parse_frontmatter` (so "valid" means "the loader reads the same data"); the text must start with `---`, the delimiters must be alone on their lines, the body must start on a new line; a reply still wrapped in a code fence is an error |
| `name` | a non-empty string (YAML ints/bools are rejected, never coerced) matching `^[a-z][a-z0-9-]{2,48}$`; not a task identifier (`^\d+$`, `^(pr\|issue\|bug\|ticket)-\d+`, `^(fix\|debug\|audit)-.*$`); not in `taken_names` (a `create` must not shadow a library skill). With `required_prefix` (`demo-` for a demo proposal, new skills only): `name: '<n>' must start with 'demo-' (demo proposal)` and `name: 'demo-' has no task class after 'demo-'`. With `expected_name` (an update) the name must equal it and the pattern, prefix and taken-name rules are skipped — the model did not choose that name |
| `description` | a non-empty string starting with `Use when ` (so `Instructions for debugging` is rejected) |
| sections | `## Inputs`, `## Workflow`, `## Outputs` present exactly once (H1/H2 delimit sections, H3+ stays inside, headings and steps inside code fences are ignored); `## Constraints` / `## Examples` non-empty when present |
| workflow | at least **one** top-level numbered item (`1.` or `1)`) outside fences (`MIN_WORKFLOW_STEPS = 1`; message `Workflow: needs at least 1 numbered step, found 0`) — a one-operation procedure is a legitimate skill |
| folklore | `is broken`, `does not work`, `不工作`, `坏了` anywhere in the text (frontmatter included), case-insensitive, reported with the line number. `never use` is **not** folklore: whether a prohibition is evidence-backed is the reviewer's check (`unsafe_claim`) |
| secrets | every match of `SECRET_PATTERNS` — `aws access key` (`AKIA…`), `private key block` (`-----BEGIN … PRIVATE KEY-----`), `github token` (`ghp_/gho_/ghu_/ghs_/ghr_…`, `github_pat_…`), `slack token` (`xox[abpr]-…`), `openai-style key` (`sk-…` ≥ 20), `jwt`, `bearer token`, `credential assignment` (`password\|passwd\|secret\|token\|api[_-]?key [:=] <value ≥ 8 chars>`, prefixed / suffixed / JSON-quoted keys included, skipped for placeholders and clause prose after `:`; section 12) — reported as `line N: possible secret (<label>); remove it or replace it with a parameter`; the value itself is never echoed |

Additions beyond the task's rule list, each one line to remove: `expected_name`, `taken_names`,
the duplicate-heading error, the delimiter-line strictness and the fence-wrapped-reply error.
Deliberately not added: see section 12. `scan_secrets(text) -> list[SecretHit(line, label, start,
end)]` and `redact_secrets(text)` (span → `[REDACTED:<label>]`) are shared with the writer.

**`review.py`** — the semantic review stage, prompt `prompts/review/prompt_v2.md` (67 lines):

| Element | Behaviour |
|---|---|
| Inputs | `{review_policy}` (`policy.review_policy_block(demo)`), `{candidates}` (`format_review_candidates`: `[c1] <title> — target …`, Rule / When / Constraints / Expected outcome, `Cites: ev1, ev2`), `{evidence}` (`format_review_evidence`: `- [ev3] (required) …` / `- [ev4] (context) …` — exactly the fragments the renderer was quoted, ids shown, so the reviewer can cite them), `{existing_skill}` (the update's current text, or `None. Create a new skill.`), `{skill_md}` |
| Reply | `ReviewReply(verdict: pass \| fail, issues: [ReviewIssue(code, detail, evidence_refs)])` — `pass` requires `issues: []`, `fail` at least one issue; strict JSON, one parse retry (`_CORRECTION`), then `ReviewContractError` |
| Issue codes | `lost_condition` (a When/constraint/prerequisite/completion criterion missing or weakened), `order_changed` (step order differs where the evidence shows order matters), `command_mismatch` (command, argument, flag, path shape or result differs from the evidence), `unsupported_addition` (a command, flag, API, version, unit, dependency, step or guarantee not in candidates/evidence — **including a claimed test or verification not present in the evidence**), `missing_rule` (an accepted candidate not represented), `unsafe_claim` (a prohibition or safety statement without evidence), `one_time_value` (a one-time path/id/host/timestamp left in; durable conventions are fine), `name_policy` (the name violates the review policy: no `demo-` prefix, or not a new skill, under demo), `other` |
| What a pass means | "faithful to the candidates and evidence" — not "executed", not "works everywhere"; the reviewer compares and never rewrites; triviality, style, length and library overlap are not issues (`REVIEW_POLICY_NORMAL`: *Judge fidelity, not value*) |
| `review_skill_md(skill_md, candidates, fragments, existing_skill_text, *, llm, template, policy_text, corrective_retry=True) -> ReviewResult(verdict, issues, unknown_refs, calls)` | guards before any call (blank SKILL.md, no candidates, no fragments, missing placeholder); an issue citing an id the renderer never saw is kept and listed in `unknown_refs` with a warning, never dropped; `record()` is the JSON stored in provenance |
| `render_and_review(candidates, *, llm, render_template, review_template, render_policy, review_policy, evidence, existing_skill=None, existing_skill_text=None, expected_name=None, taken_names=(), required_prefix=None, evidence_budget_chars=None) -> RenderedSkill` | the per-plan stage both commands call: selection (empty or missing required → `insufficient_context_budget`, 0 calls) → `render_skill_md` (≤ 2 calls; invalid twice → `render_invalid`) → `review_skill_md` (≤ 2 calls; unusable reply → `quality_review_failed` with `reviewer reply invalid: …`) → on `fail` **one** `revise_skill_md` (1 call; invalid → `quality_review_failed` with `corrective SKILL.md invalid: …`) → `review_skill_md(corrective_retry=False)` (1 call; `fail` or invalid → `quality_review_failed`). **At most 6 `ainvoke` per plan.** `RenderedSkill(content, validation, review, render_evidence, render_evidence_omitted, calls, code, errors, corrected, initial_issues)`; `review_record()` = the final review plus `corrected` and `initial_issues` → provenance `quality_review`. `LlmBudgetExhausted` is not caught here |
| `VERIFICATION_EVIDENCE_SUPPORTED` | `{"level": "evidence_supported", "note": "not executed by generator"}` — the `verification` record of every proposal this stage passes |

The pipeline prints the outcome per plan: `Quality review: passed` or `Quality review: passed
after one correction`, `Verification: evidence_supported; not executed by generator`, and for
refusals `plan <label>: quality review failed after one correction; nothing written:` followed
by `  - <code>: <detail>` lines, `plan <label>: quality reviewer reply invalid; nothing written:
…`, or `plan <label>: required evidence does not fit the render budget
(insufficient_context_budget); nothing written:`. A SKILL.md the validator refused twice keeps
the historical `plan <label>: SKILL.md rejected after one correction; nothing written:` plus the
error list.

**`writer.py`** — `write_proposal(content, *, root, name, provenance, thread_id) -> Path` writes
`<root>/.proposals/<thread>/<name>/provenance.json` first and `SKILL.md` second (a skill never
exists without its provenance), refuses unsafe names and thread ids (it never writes outside
`.proposals/`), decides collisions by directory existence with an atomic `mkdir()` (`-2`, `-3`, …;
a half-written folder from a crash is skipped, not overwritten), and requires every key of
`REQUIRED_PROVENANCE_KEYS` with non-empty `thread_ids` and `candidates`. Before any directory is
created it also enforces the demo guard (`provenance["demo"] is True` with a target other than
`create` → `ValueError("demo proposal must target a new skill (create)")`) and the **secrets
guard**: `scan_package({"SKILL.md": text, "provenance.json": payload})` scans the **whole
package**, and a hit raises `SecretsDetected("proposal package contains possible secrets;
nothing written: SKILL.md:12: possible secret (github token); …")` — the pipeline records it as
the code rejection `secrets_detected` and prints `plan <label>: proposal package contains
possible secrets; nothing written: …`. Evidence text inside `provenance.json` is stored through
`redact_secrets`, so a token that appeared in a tool output the model was shown never reaches
the disk in clear (`[REDACTED:<label>]`).

`build_provenance(*, thread_ids, bundle, candidates, rejected, sources, model, prompt_variants,
category, target, policy, demo, config, versions, prompt_hashes, quality_review, verification,
covering_skill=None, render_evidence=None, render_evidence_omitted=None, generated_at=None)`
produces the **provenance v4** contract (`PROVENANCE_VERSION = 4`), scoped to the render plan
the proposal came from:

```json
{"provenance_version": 4,
 "thread_ids": ["<analysed thread>", "<threads a shared procedure relies on>"],
 "sources": {"<file name>": "<path of the trajectory>"},
 "episodes": [{"kind": "observed_procedure", "weight": 0.0, "thread_id": "...", "source": "<file name>",
               "bundle_status": "shown | trimmed | excluded",
               "evidence": [{"id": "ev1" | null, "source": "<file name>", "line": 2, "seq": 2,
                             "role": "task", "required": true}]}],
 "evidence_shown": {"ev1": {"source": "<file name>", "line": 2, "seq": 2, "role": "task",
                            "required": true, "text": "<the excerpt line the model saw, secrets redacted>"}},
 "candidates": [{"candidate_id": "c1", "title": "...", "rule": "...", "evidence_refs": ["ev1", "ev3"],
                 "applies_when": "..." | null, "constraints": [], "expected_outcome": "..." | null,
                 "future_applicability": "high", "covered_by": "<library skill>" | null,
                 "target": {"action": "create", "existing_skill": null}}],
 "candidates_rejected": [{"title": "...", "reason": "evidence not shown in the bundle: ['ev9']",
                          "evidence_refs": ["ev9"]}],
 "render_evidence": {"c1": ["ev1", "ev3"]},
 "render_evidence_omitted": {"c1": []},
 "observed_procedure_source": [{"source": "<file name>", "line": 4, "seq": 4, "role": "step"}],
 "model": "<llm_config.model>",
 "prompt_variants": {"classify": "<resolved path>", "render": "<resolved path>", "review": "<resolved path>"},
 "prompt_hashes": {"classify": "sha256:<hex>", "render": "sha256:<hex>", "review": "sha256:<hex>"},
 "features_version": 4,
 "versions": {"features": 4, "provenance": 4, "config_schema": 2, "prompts_contract": 2},
 "generated_at": "<ISO 8601, UTC>",
 "category": "<generation.category>",
 "target": {"action": "create | update", "existing_skill": "...", "existing_path": "...",
            "base_sha256": "<sha256 of the updated file; updates only>"},
 "policy": {"requested": "reusable_workflow", "selection": "demo_workflow", "source": "config | --policy"},
 "demo": true,
 "config": {"requested": {"schema_version": 2, "demo_mode": true, "classification": {...}, "...": "..."},
            "effective": {"policy": "...", "demo": true, "selection": "demo_workflow", "max_plans": 3, "...": "..."}},
 "quality_review": {"verdict": "pass", "issues": [], "unknown_refs": [], "calls": 1,
                    "corrected": false, "initial_issues": []},
 "verification": {"level": "evidence_supported", "note": "not executed by generator"},
 "covering_skill": {"display_name": "<library skill>", "path": "<its SKILL.md>", "sha256": "<hex>"} | null}
```

Three things are told apart: what the detectors **extracted** (`episodes`, every input episode
of the thread with its bundle outcome and every cited event — `id: null` marks an event this
proposal's candidates do not cite: never shown, or cited only by another plan of the thread),
what this proposal **rests on** (`evidence_shown`: the fragments its candidates cite, a subset of
what the classify model saw, id → file, physical line, seq, role and the redacted text;
`observed_procedure_source`: the physical lines of every `observed_procedure` episode the plan
cites, so a demo skill points back at the whole recorded chain), and what reached its
**render** call (`candidates` are exactly the rendered plan, `render_evidence` /
`render_evidence_omitted` the renderer's actual selection). The v4 keys make the *mode* of the
run reproducible: `policy` (the requested policy, the effective selection and whether the CLI
set it), `demo`, `config.requested` versus `config.effective` (section 5), `versions`,
`prompt_hashes` (section 6), `quality_review` and `verification` (above), `covering_skill` (the
library skill a demo proposal knowingly duplicates), and `target.base_sha256` (the exact file an
update was written against). Guards: a demo proposal that is not a `create`, an update without
`base_sha256`, an unknown `verification.level`, an incomplete `policy` or `prompt_hashes` record
raise `ValueError` before anything is written.

Compatibility: **v3** proposals are scoped to the plan but lack the v4 keys; **v2** proposals
hold every kept candidate of the thread and the whole registry; proposals written before that
carry no `provenance_version` (**v1**): their `candidates[].evidence_refs` are seq numbers, their
`episodes[]` rows have `evidence_seq` and there is no registry. `/skill-review` reads `category`,
`thread_ids`, `generated_at` and `target`, which every version shares, plus `demo` and
`policy.selection` when present — an older proposal without a `demo` key is **not** a demo. So
older proposals still list, accept and reject. `features_version` 4 marks the enriched
evidence-item contract of section 14 (3: evidence items; 2: incidents gate; 1: summed
weights). Property tests: every `evidence_shown` entry of a written `provenance.json` re-reads
to the physical line holding that `seq`, including across corrupted lines, and every
candidate's refs are a subset of the registry
(`test_writer.py::test_provenance_shown_fragments_resolve_to_source_lines`,
`test_direct_skill_generation.py::test_handle_writes_proposal_not_library`); the correcting
phrase of a long user message is in the registry text and in the LLM payloads
(`test_handle_long_correction_phrase_is_in_provenance`); a budget too small for any episode's
required evidence creates no LLM (`test_handle_bundle_exclusion_creates_no_llm`,
`test_skill_mining.py::test_real_run_bundle_exclusion_creates_no_llm`); a secret in a
trajectory never reaches the package (`test_writer.py::test_write_proposal_refuses_package_with_secret`,
`test_build_provenance_redacts_secrets_in_evidence_text`,
`test_demo_pipeline.py::test_secret_in_trajectory_never_reaches_the_package`).

Verification: `pytest tests/ut/skill_evolver -q` — validator positives and one negative per rule
(incl. `description: Instructions for debugging`, the one-step workflow passing, an
evidence-backed `Never use --force` passing, every secret label, placeholder credentials
passing), all-errors collection, writer layout/collisions/rejections/guards, the scanner
guarantees, render happy path / one correction / double failure / update name enforcement /
all required fragments beyond three / every missing required reported / refusal before the
LLM under a small budget / guards, the review stage (unknown flag blocked then corrected, a
lost constraint failing after one correction, the six-call ceiling, unknown refs kept), and
the handler end to end on the `skill_evolver_signals.jsonl` fixture with a scripted LLM
(proposal written, library untouched, refusals without an LLM call, threshold, `nothing`
verdict, fabricated refs `(invalid_evidence)`, double validation failure, reference-only,
unknown update target, update with existing text and `base_sha256`).

## 17. CLI surface: `/trajectories`, `/skill-mine`, `/skill-review`

The pipeline of sections 14-16 was reachable only through `/direct-skill-generation`, which
analyses one thread. Three commands open it up; the generator stays registered and working,
marked `[deprecated]` in `/help` and printing `Use /skill-mine for trajectory-based generation.`
at the start of every run — and it now shares the flags and the dry run of `/skill-mine`.

| Invocation | Effect |
|---|---|
| `/trajectories [list]` | Table of the project's recorded threads: thread, agent, turns, events, size, mtime, first user message |
| `/trajectories show <thread-id>` | One thread as markdown (`export.render_markdown`); the id may be a unique prefix |
| `/skill-mine [--threads N] [--since 7d] [--dry-run] [--thread <id>] [--policy strict_knowledge\|reusable_workflow] [--demo\|--no-demo]` | Mine several threads, up to `max_plans_per_thread` proposals per thread |
| `/direct-skill-generation [last\|<thread-id>] [--dry-run] [--policy strict_knowledge\|reusable_workflow] [--demo\|--no-demo]` | One thread through the same pipeline; the first token without `--` is the target, a second one is `unexpected argument '<tok>'`, an unknown flag is `unknown argument '--x'` |
| `/skill-review [list]` | Table of the proposals on disk: name, category, action, **demo** (`DEMO` or empty; `?` on an unreadable row), threads, age, description |
| `/skill-review accept <name>` | Re-validate (with the `demo-` prefix rule for a demo proposal) and move the folder into `<root>/<category>/<name>/`; a demo proposal asks one extra confirmation |
| `/skill-review reject <name>` | Delete the folder after an explicit confirmation |

Shared flags (`cli_args.take_shared_flag`): `--policy <name>` (`--policy requires a value`;
`--policy: expected one of strict_knowledge, reusable_workflow, got 'x'`; `--policy given
twice`), `--demo` / `--no-demo` (`--demo given twice`; `--demo and --no-demo cannot be
combined`), `--dry-run` (`--dry-run given twice`). They are applied by `apply_overrides` for the
current run only (section 5). New files: `src/msagent/cli/handlers/trajectories.py`
(`TrajectoriesHandler`), `src/msagent/skill_evolver/mining.py` (`SkillMiningHandler`),
`src/msagent/cli/handlers/skill_review.py` (`SkillReviewHandler`),
`tests/ut/cli/handlers/test_skill_mining.py` (`/skill-mine` and `/skill-review`) and
`tests/ut/cli/handlers/test_trajectories_handler.py`. Registration follows the existing pattern
exactly: export from `cli/handlers/__init__.py`, instantiate in `CommandDispatcher.__init__`, one
dict entry in `_register_commands()` and one `cmd_*` delegate whose **docstring is the help text**.
Slash completion needs no change — `Session` derives it from `dispatcher.commands.keys()`, and
`completers/reference.py` is the `@`-file-path completer, not a command table. There is no
subcommand or flag completion for any command in this CLI.

### 17.1 `/skill-mine`: up to `max_plans_per_thread` proposals per thread

`handle()` parses, then `_run()` loads the config (`_load_config` → `apply_overrides` →
`effective_rules`), resolves the trajectories directory, extracts evidence for every selected
thread in one `asyncio.to_thread` call (`select_trajectories(..., cross_session_limit=rules.
cross_session_limit)` + `mine_stats(targets, pool, skills, demo=rules.demo)`), prints the
selection line, the policy block (section 18) and the **Threads** table, and either stops (dry
run) or enters the per-thread loop.

`/skill-mine` loops over the threads whose `gate_decision()` passes (section 14, demo-aware) and
hands each to `run_thread` under one `RunContext` per run. A thread the gate refuses is not
handed to `run_thread`, but `pipeline.record_gate_refusal` writes its gate-only decision report
(the gate record, one `gate skip` stage, the refusal text as `stop_message`; section 20) before
the loop, so a run in which every thread is refused prints `Nothing to mine: …` and still leaves
one report per thread. The header of each mined thread is
`[i/n] thread <short id> — score S, N episodes, K incidents, <describe()>`. A run costs at most
`llm_call_bound(threads, max_plans, max_llm_calls=, expand=)` LLM calls (section 19; printed as
`Mining N threads (up to B LLM calls)`) and writes at most `max_plans_per_thread` proposals per
thread. Plans are rendered, reviewed and written one by one: a plan that fails (twice invalid,
review failed, secrets, budget, a transport error) is a render error of that plan and the next
plan still runs; an exception outside the plan loop (classify) fails the thread — printed as
`thread <id>: <exc>`. Names a new skill may not reuse (`RunContext.taken`) are the library plus
every create proposal written earlier in the run, across threads. The summary line keeps its
shape — `Mined N threads: P proposals, G skipped by the gate, Z nothing to save, F failed` —
and, when anything besides clean proposals happened, a second line reports `Plans: R render
errors, Q rejected targets, D deferred`. The pool is the agent's newest `cross_session_limit`
(20) trajectories and each target goes **first** into `collect_episodes`, so the kept
`repeated_procedure` episodes belong to that thread; `ThreadStats.supporting` holds the pool
trajectories their evidence cites (the second session of each shared procedure), `ThreadStats.
notes` the demo detector's drop reasons, and the bundle indexes `[target, *supporting]`.

Selection (`select_trajectories`): one `load_trajectories(dir, agent=..., limit=max(N,
cross_session_limit))` pass feeds both the pool (first `cross_session_limit`) and the targets, so
no file is parsed twice. `--since` compares **file mtime**, not `Trajectory.started_at`: mtime
always exists and means last activity, while `started_at` is the first event's `ts` and can be
empty, which would force a keep-or-drop fallback on unparsable data. Since the listing is
already mtime-descending, the window is a contiguous prefix and `--threads N` means "the newest
N inside the window". Mtime does not survive copying files between machines — the documented
cost of that choice. `--thread` resolves through `find_trajectory_file` (unique prefixes
included) and reuses the pool object when it is there, so a thread older than the newest 20, or
belonging to another agent (the synthetic demo thread, for instance), still works. Empty
selections get three distinct warnings: nothing recorded, nothing for this agent, nothing
inside the window.

Argument parsing is hand-rolled, not argparse: argparse reports errors with `sys.exit`, and
`SystemExit` is a `BaseException` that neither `CommandDispatcher.dispatch` nor `Session._main_loop`
catches, so a mistyped flag would end the user's session. `--since` accepts `<count>` plus `h`,
`d` or `w`; `m` is **rejected** as ambiguous between minutes and months. `--thread` combined with
`--threads` or `--since` is an error, not a precedence rule, and a repeated flag is an error —
silently ignoring a flag the user typed is the same masking pattern the project rules forbid.
`--threads` defaults to 5. `MineArgsError` is an alias of `cli_args.CliArgsError`.

### 17.2 The dry run: the detector debugger

`--dry-run` stops after feature extraction and never constructs an LLM, writes no proposal and
no decision report. The single construction site is `LazyLlm.get()`, reached inside
`run_thread` only after a thread passes both gates, so a real run in which every thread fails
the gate creates no LLM either (it still writes the gate-only decision reports of section 20).
`test_dry_run_never_creates_an_llm` pins this by making
`initializer.llm_factory.create` and `load_llm_config` raise;
`test_dry_run_with_overrides_prints_effective_config_and_changes_nothing` adds that the config
file bytes are unchanged and no `decisions/` directory appears.

Two tables (`/skill-mine`), because one cannot carry both "why nothing fired" and "what fired":

- **Threads** (both modes): thread, turns, tools, ai, episodes, incidents, score, gate, reason;
  the threshold is in the title (`Threads (min_evidence_score 1.00)` or `Threads
  (min_evidence_score 1.00, demo override)`). `incidents`, `score`, `gate` (`pass` / `skip`) and
  `reason` come from one `gate_decision()` per thread, so a thread admitted by the strong
  correction rule at score 0.90 reads `pass` with `strong user correction`, one admitted by the
  overlay reads `pass` with `demo_override; observed_procedure has required evidence`, and a
  skipped one says whether it had `no episodes` or a `score < min_evidence_score` (`; no
  episode has complete required evidence` under demo). A zero `tools` count is styled
  `warning` — it is the most diagnostic number in the table, because every detector that needs
  tool calls is then dead.
- **Episodes** (dry run only): thread, kind, incident, weight, tool sequence, evidence seq (the
  own-thread seqs via `Episode.evidence_seq`, for display — refs identify events by file and
  line), grouped per thread with a `subtotal` row (empty `incident` cell). `incident` labels the
  episodes of one thread that describe the same events — `I1`, `I2`, … in `group_incidents`
  order — so the rows sharing a label are the ones the score counts once. Rows keep **detector
  order**, not weight order: the bundle sorts by weight for the model, while a human wants to
  know which detector fired. Both list columns carry the true element count in brackets before
  any clipping (5 tool names, 8 seqs, then `… +k`).

After the tables, both commands print the same per-thread muted block — `gate_lines()` and
`bundle_preview()`: the gate decision with its reason, what the bundle would hold (`bundle: F
fragments, C chars (cap M); episodes shown S, trimmed T, excluded X`, one `excluded: <kind>
(weight W)` line per exclusion), the observed chains (`observed_procedure: read_file → bash →
bash (7 events)`) and, under demo, `observed_procedure candidates: N selected; dropped:
<reason>=<count>, …` (or `none`). `/skill-mine` ends on one line: `Dry run: N threads, E
episodes, K incidents, total evidence score S; P threads would reach the LLM (up to
llm_call_bound(P, max_plans, max_llm_calls=, expand=) LLM calls). Nothing was written and no LLM
was created.`; `/direct-skill-generation` on `Dry run: nothing was written, no LLM was created,
no decision report saved.`. The real output of the synthetic demo thread:

```text
/direct-skill-generation thread-demo-synthetic --dry-run --demo --policy reusable_workflow

Use /skill-mine for trajectory-based generation.
Requested policy: reusable_workflow (--policy)
Demo mode: true (--demo)
Effective selection: demo_workflow
Demo overlay: gate bypass (demo_override), observed_procedure extraction, target create only, names demo-<task-class>
Dry run: no LLM will be created.
Evidence score: 0.00 (min_evidence_score 1.00; 1 episodes, 1 incidents)
Gate: pass (demo_override; observed_procedure has required evidence)
bundle: 7 fragments, 1980 chars (cap 30000); episodes shown 1, trimmed 0, excluded 0
observed_procedure: read_file → bash → bash (7 events)
observed_procedure candidates: 1 selected; dropped: none
Dry run: nothing was written, no LLM was created, no decision report saved.
```

The same thread with `--no-demo` reads `Evidence score: 0.00 (min_evidence_score 1.00; 0
episodes, 0 incidents)` / `Gate: skip (no episodes)`: no detector fires, so the LLM is never
reached. A full run adds the classify and plan lines the task specification lists as its
diagnostic example — `Decision E1: accept (accepted)`, `Candidate '<title>': accepted (trivial
procedure allowed)`, `Skill proposal saved to …`, `Quality review: passed`, `Verification:
evidence_supported; not executed by generator`, `Proposal: saved, inactive, DEMO` — and ends
with the muted `Prompts: classify=<path>, render=<path>, review=<path>` line.

Colour comes from column-level styles, never inline markup, and every data-derived cell and
line is passed through `rich.markup.escape` or wrapped in `rich.text.Text`, so a tool named
`[bold]` renders literally (`test_episodes_table_shows_markup_literally`).

Files recorded before the recorder's `ignore_agent` fix carry no `tool.*` events at all
(`ARCHITECTURE_trajectory_recorder.md` section 11), so **every** detector yields nothing on them —
`user_correction` included, because it needs an observed change of the agent's actions between
turn groups. That is the realistic first run on existing data, so a note under the Threads
table names the cause once (not per row) whenever any selected thread shows zero tool calls.

### 17.3 Failure policy

Three tiers, which is how "the analyzer must fail loudly" and a usable multi-thread loop coexist:

- Configuration and prompts are checked **before** anything runs: `SkillEvolverConfigError`
  and `PromptContractError` are printed by `report_config_error` (every problem, then the fix
  hint) and the command returns; no LLM, no report. `/skill-review` prints the config error
  lines the same way and returns.
- Parsing, selection, loading and detection have **no `try` at all**. They reach the
  top-level guard in `handle()`, which prints and `logger.exception`s, and the run stops. One
  corrupt neighbouring file raises `TrajectoryReadError` naming itself, printed as `Trajectory
  unreadable (<exc>); the LLM was not called`.
- Per-thread generation is guarded inside `run_thread` and around it: a plan failure is
  printed with its label and counted as a render error, the loop continues with the next plan;
  a classify parse/contract error or any other exception fails the thread — printed with the
  thread id, the traceback logged, the id repeated in the summary, the summary switched to
  `print_error` — and the loop continues with the next thread, because aborting would discard
  the remaining threads after earlier ones already wrote files. `LlmBudgetExhausted` inside a
  plan writes nothing for that plan and defers the remaining plans of the thread. The
  decision report is written in `finally` in every case. `KeyboardInterrupt` is caught by no
  tier.

Every run ends on one fixed-shape line: threads mined, proposals, skipped by the gate, nothing to save,
failed.

### 17.4 `/skill-review`

Proposals are read from `<root>/.proposals/<thread>/<name>/` — `SkillFactory.load_skills` skips
dot-directories, so `/skills` never shows them and the review command must scan the tree itself.
`<root>` is `config.output_root(cfg, working_dir)`, the same resolution the generating
commands use, so a relative `generation.output_dir` means the same folder for all three
commands. The directory name is not always the skill name (collisions get `-2`), so `accept`
takes the destination name from the frontmatter via `skill_name()`, which is what the loader
reads. A proposal is addressed by bare name when that name exists in exactly one batch, and by
`<thread>/<name>` otherwise; an ambiguous bare name lists the qualified candidates instead of
guessing. A proposal whose `provenance.json` is missing or unparsable is **listed with its error**
rather than skipped — a half-written folder must stay visible to the person who has to decide.

The table has a `demo` column after `action`: `DEMO` when `provenance["demo"] is True`, empty
otherwise (an old proposal without the key is not a demo), `?` on an unreadable row;
`Proposal.policy` carries `provenance.policy.selection` for the confirmation text.

`accept` re-runs `validate_skill_md` because the file may have been edited by hand — with
`required_prefix="demo-"` for a demo proposal, so a demo skill renamed without the prefix is
refused (`name: '…' must start with 'demo-' (demo proposal)`), and with the secret patterns, so
a hand-added credential is refused too — refuses when `<root>/<category>/<name>/` already exists
(the only guard against shadowing a library skill), and, **for a demo proposal only**, prints
the warning `demo proposal (<selection>): a teaching, possibly trivial skill` and asks `Accept demo
proposal '<name>' into the library anyway?`; anything but an explicit yes prints `Cancelled;
nothing was moved` and leaves the folder in place. An ordinary proposal is never asked
(`test_accept_non_demo_never_asks`). Then it moves the whole folder with `shutil.move` so
`provenance.json` travels with it, and prunes the emptied batch directory; nothing under the
private state directory (the decision reports, section 20) is read, copied or moved. `reject`
deletes after an explicit confirmation. Both confirmations are `SkillReviewHandler._confirm`
(a `PromptSession` with a yes/no completer, shaped like `InterruptHandler._prompt_choice`) —
there is no y/N helper anywhere else in the CLI, and being a method is what makes it patchable
in tests. Anything but an explicit yes, and `Ctrl+C`, mean no.

**Update proposals are refused, not moved** (an addition beyond the task, which describes only the
create case): an `update` carries the name of a library skill, so moving it into
`<root>/<category>/<name>/` would create a second skill with that name in another category. The
command prints `provenance.target.existing_path` and says the file must be replaced by hand, which
matches the activation hint the generator already prints and section 12.

Note that `skill_review.py` imports the generator handler **inside** `_root()`: `handlers/__init__`
loads this module before the generator, which re-enters the package for `session_history`. A
module-level import would widen the pre-existing cycle documented in
`test_direct_skill_generation.py`.

Verification: `pytest tests/ut/cli tests/ut/skill_evolver -q`. The mining tests cover the parsers
(every error message, the shared-flag errors, `30m` and `0d` rejected, `7D` accepted, the
`--thread` conflicts, duplicate flags, defaults incl. `policy is None` / `demo is None`), the
formatters (the `[n]` prefix always equals the true length), the tables (markup shown
literally, subtotals, the demo override reason), the dry run (no LLM, the zero-tool note, the
three empty selections, `--thread`, `--threads`, the call bound, overrides changing nothing on
disk), the real run on the `skill_evolver_signals.jsonl` fixture with a scripted LLM (one
proposal written with its v4 provenance, `nothing` verdict, a failing thread reported while the
loop continues, the review and bound lines) and on the demo fixture (`--thread
thread-demo-synthetic` with and without `--demo`), and `/skill-review` (list incl. the `DEMO`
marker, accept, category, hand-broken SKILL.md, occupied destination, update refusal, ambiguity,
qualified names, the demo confirmation given and declined, no question for ordinary
proposals, a demo proposal without the prefix refused, the private state left untouched by
accept, reject with and without confirmation).

## 18. Selection policies and demo mode

What deserves a candidate is decided by exactly one **policy block** per prompt stage, inserted
by code (`policy.py`); the fixed prompt texts carry the evidence standard and the output
contract only. `classification.policy` (or `--policy`) names the ordinary policy; `demo_mode`
(or `--demo` / `--no-demo`) is an overlay that replaces the classify block and switches the
render and review texts. The **effective selection** (`EffectiveRules.selection`) is therefore
one of three values, and `test_policy.py::test_every_classify_assembly_has_exactly_one_policy_heading`
pins that every assembled classify prompt carries exactly one `# Selection policy: <selection>`
heading.

| Selection | When | What the classifier is told (`selection_policy_block`) |
|---|---|---|
| `strict_knowledge` | default | Save only proven, transferable, durable knowledge that is not an obvious use of ordinary tools and is not already covered by the library — a non-obvious rule, condition, limit, mechanism or a substantial correction. One episode is enough when it carries direct confirmation (error → fixed call → result, an explicit correction, a visible mechanism); several sessions are not required. A matching tool sequence is not proof of usefulness (`routine_activity`); a rule the library already states is `already_covered` → `reference`, or `update` only when the evidence changes it; `no_transferable_rule`, `transient_observation`, `unsafe_procedure` as defined; keep conditions, parameterise one-time values, cite the proving excerpts |
| `reusable_workflow` | `classification.policy: reusable_workflow` / `--policy reusable_workflow` | Save a useful, re-applicable **procedure**: its steps may be standard tools, the value may lie in their selection, order, branching, applicability conditions or completion criteria — a new technical discovery is not required. A bare list of tools without inputs, decision logic and a checked result is `routine_activity`; every kept step must be visible in the excerpts (arguments and results), a step whose result is missing, failed or only claimed is `insufficient_evidence`; a procedure the library already describes is `already_covered`; keep conditions, order and limits as observed, parameterise, state the observed completion criterion |
| `demo_workflow` | `demo_mode: true` / `--demo`, on top of either policy | Demonstration mode: the rule does not have to be new, surprising or absent from the library; a trivial but confirmed procedure is a valid candidate, including one the library already covers. Accept an episode (typically `observed_procedure`) when the excerpts show the task context, the real arguments of each operation and its real, meaningful result (`accepted`). A covered procedure is still a **separate teaching candidate**: `target.action` must be `create`, `existing_skill` null, the covering skill named in `covered_by`; `update` / `reference` are never answered in this mode. **The evidence standard is unchanged**: an orphan call, a missing or failing result, `status: ok` without a meaningful output, an unconfirmed claim of success is `insufficient_evidence`; destructive or credential-exposing steps are `unsafe_procedure` |

The render and review stages follow the effective demo flag, not the requested one:
`render_policy_block(demo)` is `RENDER_POLICY_NORMAL` (*the candidates passed the evidence gate
on their own merit; name a new skill after its task class*) or `RENDER_POLICY_DEMO` (*a
confirmed procedure that may be trivial or common knowledge; write it anyway as a teaching
skill — concrete inputs with their format, the exact operation with its real argument shape,
the observed result and how to check it, never generic advice; the name MUST start with `demo-`
followed by the task class and must not reuse a library name; always a new skill; do not
mention demo mode inside the file*); `review_policy_block(demo)` is `REVIEW_POLICY_NORMAL`
(*judge fidelity, not value; triviality, novelty and library overlap are not issues*) or
`REVIEW_POLICY_DEMO` (the same, plus `name_policy` when the name lacks the `demo-` prefix or the
proposal is not a new skill). The blocks contain no braces and are inserted with `str.replace`
or the one-pass regex, never `str.format`; `policy.py` takes plain strings and booleans and
imports nothing from the config, so it cannot become a second source of truth for the rules.

The principle is **lower novelty, never lower evidence**, and the console says so up front:
`Demo overlay: gate bypass (demo_override), observed_procedure extraction, target create only,
names demo-<task-class>`. Precisely:

**Barriers removed by demo (only with `demo=True`):**

- the non-obviousness / novelty test — it lives in the `strict_knowledge` block, and the
  `demo_workflow` block replaces it;
- cross-session repetition as a source of evidence (`repeated_procedure` needs two sessions;
  an `observed_procedure` needs one);
- the requirement of a human correction or an observed failure (the ordinary detectors all
  fire on something going wrong or being corrected);
- the `min_evidence_score` threshold at **both** gates — the first gate over the thread's
  episodes and the second over `bundle.kept` — through `demo_override`, granted only when at
  least one episode has its complete `REQUIRED_ROLES` set;
- the library-coverage veto: a procedure a library skill already covers becomes a separate
  `demo-<task-class>` proposal with `covering_skill` in its provenance instead of a
  `reference` / `update`;
- the "some detector must fire" requirement: the `observed_procedure` detector is enabled and
  admits a confirmed chain of `ok` calls at weight 0.0.

**Checks that stay mandatory under demo (identical to an ordinary run):**

- a non-empty evidence package with the roles `task`, `step` and `result` (`REQUIRED_ROLES`),
  every ref resolving to a physical line of the recorded file, `tool.start` present for every
  step (its arguments are the knowledge), no `ERROR_MARKERS` text in any output and a non-empty
  final output — AI narration is never evidence;
- the classify contract with its one corrective retry: consistent verdict and candidates,
  known `E<n>` and `ev<k>` ids only, `create`-only targets (an `update` or `reference` reply is a
  contract violation, then a `ClassifyContractError`, never a silent swap);
- code-side rejection of every candidate whose refs are empty or not shown (`invalid_evidence`);
- name and path safety plus the `demo-` prefix with a task class after it, checked by the one
  shared `validate_skill_md` (in the render correction loop, at write time and again at
  `/skill-review accept`); the `SKILL.md` structure rules of section 16;
- the semantic review — faithfulness to candidates and evidence, no unsupported additions,
  no claimed test that did not run, prohibitions only when the evidence shows the failure —
  with one correction, and the `quality_review` / `verification` records in provenance;
- the secrets scan of the **whole package** (`SKILL.md` and `provenance.json`) and the
  redaction of evidence text;
- the budgets: `max_plans_per_thread`, `max_llm_calls_per_thread` through `CountingLlm`,
  `bundle_max_chars`, `excerpt_max_chars` and the context budget — demo raises none of them
  (`test_demo_keeps_limits_unchanged`);
- inactive `.proposals/` only, never a library category, never an existing skill file, never
  the analysed session; acceptance needs the explicit extra confirmation;
- no execution of any command from the trajectory — the generator observes outputs, it never
  reproduces them (`verification.level = evidence_supported`).

A demo run on the synthetic fixture is refused at the first gate without `--demo`
(`Gate: skip (no episodes)`), and refused *with* `--demo` when the chain is incomplete: an
orphan call, an AI message claiming success without a completed call, or an `ok` output that
contains a traceback all yield no `observed_procedure`, hence `no episode has complete required
evidence` (`test_demo_pipeline.py`, `test_features.py`).

## 19. Budgets and LLM-call accounting

Every `ainvoke` of a thread goes through one `CountingLlm(inner, limit=max_llm_calls_per_thread)`
(`budget.py`). It increments `calls_used` **before** the transport call — a call that failed in
transport still spent its slot — and raises `LlmBudgetExhausted(used, limit)` (`LLM call budget
exhausted (U/L calls per thread)`) as soon as `calls_used >= limit`. Transport retries inside the
LLM client are invisible to it and are **not** counted; the decision report says so literally
(`llm.note = "transport retries not counted"`), so the counter is never presented as an
accounting of network retries. When the budget is exhausted the pipeline stops making new calls:
the current plan is rejected with the code `budget_exhausted` (`plan <label>: LLM call budget
exhausted (U/L calls per thread); SKILL.md not written` — nothing unverified is ever written,
because a proposal is written only after its review passed), every remaining plan is deferred
(`Deferred plan: <label> — llm call budget exhausted (U/L)`), and `budget_exhausted: true`
lands in the report. With the default limit of 16 the classify calls (at most 4 with the expand
round) always fit and at least one plan can complete in the worst case; a limit below the
classify cost makes `LlmBudgetExhausted` propagate out of the classify stage, which fails the
thread like any other exception (section 17.3).

The per-thread ceiling and the run bound:

```python
CLASSIFY_CALLS = 2        # classify + one corrective retry
EXPAND_CALLS = 2          # the expand_context_once round (same shape)
PLAN_CALLS = 6            # render 2 (first + validator correction) + review 2 (first + parse retry)
                          # + revise 1 + review 1
thread_call_ceiling(max_plans, *, expand) = CLASSIFY_CALLS + (EXPAND_CALLS if expand else 0) + PLAN_CALLS * max_plans
llm_call_bound(threads, max_plans, *, max_llm_calls=16, expand=True) = threads * min(max_llm_calls, ceiling)
```

With the packaged defaults (`max_plans_per_thread: 3`, `max_llm_calls_per_thread: 16`,
`on_nothing: expand_context_once`) the ceiling is 22 and the bound 16 per thread —
`llm_call_bound(1, 3) == 16`, `(1, 1) → 10`, `(1, 2) → 16`, `(5, 3) → 80`, `(0, 3) → 0`
(`test_budget.py`, `test_skill_mining.py::test_llm_call_bound`); `/skill-mine` prints it as
`(up to B LLM calls)` in the dry-run summary and the `Mining …` line, the report as `llm.bound`.
A thread that stops early (gate, `nothing`, budget) spends less; the bound is a ceiling, not a
cost estimate.

**Context budget.** `ContextBudget.for_window(context_window)` bounds the whole prompt text by
the model window when `LLMConfig.context_window` is known:

```python
CHARS_PER_TOKEN = 4; USABLE_RATIO = 0.6; REPLY_RESERVE_TOKENS = 2048; MIN_BUNDLE_CHARS = 500
available_chars = int(context_window * CHARS_PER_TOKEN * USABLE_RATIO) - REPLY_RESERVE_TOKENS * CHARS_PER_TOKEN
# 128000 tokens -> 299008 chars; 32000 -> 68608; 8192 -> 11468; 1000 -> 0; None or <= 0 -> not enforced
room_for(overhead_chars) = max(0, available_chars - overhead_chars)   # None when not enforced
describe() = "context window N tokens -> M chars for the prompt" | "context window unknown; not enforced"
```

It is applied twice. Before classify, the pipeline computes the overhead of the substituted
classify template (library snapshot and policy block included) plus the exgraph appendix and
fits the bundle **once**: if `bundle.text` exceeds the room, the bundle is rebuilt with
`max_chars = room` (`Bundle cut to R chars to fit the model context (…)`; excluded episodes and
the second gate apply again), unless the room is below `MIN_BUNDLE_CHARS` (500), in which case
the thread stops with `Nothing to save: the classify prompt does not fit the model context
(<describe()>); bundle needs N chars, R available` and the code rejection
`insufficient_context_budget`. Before each plan, `render_plan` derives
`evidence_budget_chars = room_for(max(render overhead, review overhead))` — the render and
review templates, their policy texts, the candidate blocks without evidence and the existing
skill text — and `select_plan_evidence` places the required fragments first; the review
payload additionally carries the SKILL.md itself, which the 2048-token reply reserve covers.
The scripted fixtures use `context_window=128_000`; with the value of 1000 the evolver tests used
before, the usable budget is 0 and every thread would stop with `insufficient_context_budget`,
which is the intended behaviour for a window that small. An unknown window (`None`, the
`LLMConfig` default) enforces nothing — see section 12.

## 20. Diagnostics: the decision report

Every thread of a non-dry run leaves a JSON record of what was decided and why, written by
`report.write_report` from the payload `pipeline.build_report` assembles: in `run_thread`'s
`finally` — also when the gate refused the thread (`/direct-skill-generation` hands every thread
to `run_thread`), when the classifier said `nothing`, when a plan was rejected and when the
thread failed with an exception — and, for a thread `/skill-mine` refuses before its loop, in
`pipeline.record_gate_refusal`, which fills the same payload with the gate record, a single
`gate skip` stage and the refusal message as `stop_message` (no bundle, no LLM). Location:

```text
<ProjectPaths.root>/skill-evolver/decisions/<batch_dir_name(thread_id)>-<UTC %Y%m%dT%H%M%S%fZ>.json
```

i.e. the project's **private state** (`initializer.get_project_paths(working_dir).root`, the
folder that also holds `trajectories/`), created 0700 with 0600 files, never the skills root
and never the library. `report_dir` is `None` — and nothing is written — for a dry run and when
`diagnostics.save_decision_report` is `false`. A name collision gets a `-2`, `-3` suffix
(`O_EXCL`), the file is written atomically (temp + fsync + `os.replace`), non-JSON values are
stringified rather than lost, and a failure to write the report is logged as a warning without
failing the thread. `/skill-review accept` never reads, copies or moves anything from this
folder: the reports stay where they are when a proposal is promoted.

Payload keys (`REPORT_VERSION = 1`; `report_version` and `evidence_text_file` are stamped by
`write_report`):

| Key | Content |
|---|---|
| `command`, `generated_at`, `thread_id`, `source` | `skill-mine` / `direct-skill-generation`; UTC ISO 8601; the thread and its JSONL file name |
| `synthetic` | `is_synthetic(agent, working_dir)`: the agent name or a component of the working directory contains `synthetic` (case-insensitive) — the demo fixture's `SyntheticDemo` / `/synthetic/demo`; the header's `"synthetic": true` is not consulted |
| `policy` | `{requested, selection, demo_mode, overrides[]}` |
| `config` | `{requested: DirectSkillGenerationConfig.as_record(), effective: EffectiveRules.as_record()}` (section 5) |
| `prompts` | `{contract_version, variants{classify, render, review}, hashes{… "sha256:<hex>"}}` (section 6) |
| `episodes` | `{total, counts{kind: n}, detector_notes[{detector, reason, detail, seqs}]}` — the demo detector's drop reasons included |
| `gate`, `second_gate` | `{score, min_evidence_score, passes, reason, detail, incidents}`; `second_gate` is `null` when nothing was excluded |
| `bundles[]` | per bundle build (round 1, the context cut, the expand round): `{round, fragments, chars, max_chars, shown, trimmed, excluded, excluded_kinds}` |
| `stages[]` | `{stage, status, detail}` in order: `gate pass\|skip`, `bundle ok\|stop`, `context_fit cut\|stop`, `classify save\|nothing\|failed`, `expand retried\|skipped`, `plan <label> written\|rejected\|deferred\|error`, `thread failed` |
| `classifier` | `{verdict, candidates: n, decisions[Decision], rejected[{title, reason, evidence_refs}]}` — decisions and rejections of both rounds accumulate, verdict and count are the final round's |
| `code_rejections[]` | `{code, subject, detail}` — `insufficient_context_budget` (per excluded episode / the thread / a plan), `invalid_evidence`, `parse_error`, `contract_error`, `invalid_target`, `ambiguous_target`, `render_invalid`, `quality_review_failed`, `budget_exhausted`, `secrets_detected`, `existing_skill_unreadable` |
| `plans` | `{rendered, proposals, render_errors, rejected_targets, deferred}` |
| `quality_review[]` | per reviewed plan: `{plan, verdict, issues, unknown_refs, calls, corrected, initial_issues}` |
| `coverage[]` | per `reference` candidate: `{skill, title, coverage: text_read \| unverified, sha256}` |
| `proposals[]` | the SKILL.md paths written |
| `llm` | `{model, context_window, calls_used, limit, bound, budget_exhausted, note: "transport retries not counted"}` |
| `stop_message`, `failed` | the `Nothing to save: …` line when the thread stopped early; `<ExceptionType>: <message>` when it failed |

`diagnostics.save_evidence_text: true` additionally writes `<stem>.evidence.md` next to the
report — `# Round N` followed by `bundle.text` for every bundle build, **without** the exgraph
appendix, without model replies and without `<think>` blocks — and records its name in
`evidence_text_file` (`null` otherwise). The report never contains hidden reasoning either:
decisions carry the model's `explanation` (≤ 500 chars) and nothing else of its output.

The report answers the questions a user asks after an empty run without re-running anything:
which policy and config were in force, why the gate refused or admitted the thread, how much of
the bundle was cut, what the classifier decided about every `E<n>` and with which reason code,
which candidate was dropped by code and why, how many LLM calls were spent and whether the
budget stopped the run (`test_report.py`, `test_pipeline.py::test_run_thread_writes_report_even_when_failing_and_reraises`).

## 21. Real-LLM smoke test

The scripted-LLM tests prove flow control, never that a real model follows the prompts. The
smoke test runs the synthetic demo thread through the real pipeline with the model of the
current session (`config.llms.yml`) and records what came back. Procedure:

1. Copy the fixture into the project's private trajectory store under the name the reader
   expects for the fixture's agent and thread:
   `cp tests/fixtures/trajectories/skill_evolver_demo_success.jsonl
   <state>/trajectories/SyntheticDemo_thread-demo-synthetic.jsonl`
   (`<state>` = `initializer.get_project_paths(<working dir>).root`).
2. Dry run first — no LLM, nothing written:
   `/direct-skill-generation thread-demo-synthetic --dry-run --demo --policy reusable_workflow`
   must print the block of section 17.2 (`Gate: pass (demo_override; observed_procedure has
   required evidence)`, `observed_procedure candidates: 1 selected`).
3. Real run: `/direct-skill-generation thread-demo-synthetic --demo --policy reusable_workflow`.
   Expected shape: `Decision E1: accept (accepted)`, `Candidate '<title>': accepted (trivial
   procedure allowed)`, `Skill proposal saved to <root>/.proposals/thread-demo-synthetic/demo-<task-class>/SKILL.md`,
   `Quality review: passed[ after one correction]`, `Verification: evidence_supported; not executed
   by generator`, `Proposal: saved, inactive, DEMO`.
4. `/skill-review list` shows the proposal with `DEMO` in the `demo` column; `accept` must ask
   `Accept demo proposal '<name>' into the library anyway?` (answer no — the library stays clean).
5. Record from `provenance.json` and the decision report: the model id (`model`), the three
   `prompt_hashes`, `config.effective`, the proposal path, `quality_review` (verdict, issues,
   `corrected`), `llm.calls_used`; and verify the rendered example **by hand** on the CSV of the
   fixture — create `input/sales.csv` with `id,amount / 1,10 / 2,20 / 3,30`, substitute the
   `<parameters>` of the SKILL.md `## Workflow` command and check that it prints `60.0` (the
   check step `TOTAL_OK`). The generator never runs this itself; the automated counterpart is
   `test_demo_fixture_executable.py`, which executes only an allowlisted `python3 -c <code>
   <operands>` shape in a sandbox (`-I`, empty `PATH`, no network, a temp working dir; every
   name token of the code on a fixed allowlist, `csv.DictReader` / `sys.argv` the only attribute
   accesses, no backslash in a string, plain relative operands) and refuses anything else.
6. If the run cannot be executed (no model, no key, no network), the status below says
   literally `not executed` with the reason; a scripted-LLM run is never substituted for it.

Steps 1-3 are automated by `scripts/skill_evolver_smoke_demo.py`: it seeds the storage layout
like the CLI, copies the fixture into the private trajectory store of an isolated working
directory (outside the repository), runs the real `/skill-mine --thread thread-demo-synthetic
--demo --policy reusable_workflow` handler with the session's model (`--model <alias>`,
default `default`) and prints the proposal path, the provenance summary (model, `prompt_hashes`,
`config.effective`, `quality_review`, `verification`, `target`, `versions`), the SKILL.md and the
decision-report path. `--dry-run` runs the same path without creating an LLM.

Packaged prompt hashes of this build (`prompt_sha256` of the template before substitution):
classify `sha256:83bc316e95db04d2fb36f8732b15d0a3112b997cd0125450512659ab26824089`, render
`sha256:a2b88a82791d42ba5be007de7380d732e03adc5eb05c0db30793e9b2ad9b04eb`, review
`sha256:4bbf85808a6ca2183565b4a19ab7d73da5c5941932c6b2786d79ee1e76a82f35`.

Status (2026-09-09): real-LLM run **not executed** in the implementation environment — no LLM
credentials were available to the automated session (no `OPENAI_API_KEY`, no `.env`, no user
`config.llms.yml`; the packaged default model is `openai/gpt-4o-mini`). The dry run of the same
driver was executed on 2026-09-08 and produced the section 17.2 block (`Gate: pass (demo_override;
observed_procedure, skill_gap have required evidence)`, `observed_procedure candidates: 1 selected;
dropped: none`, `up to 16 LLM calls`, nothing written). To complete the smoke test, run
`.venv/bin/python scripts/skill_evolver_smoke_demo.py` with the CLI's credentials in the
environment and replace this paragraph with the recorded model id, prompt hashes,
`config.effective`, proposal path, `quality_review`, `llm.calls_used` and the hand-check result of
the rendered example on the CSV.
