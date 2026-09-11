#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""The per-thread Skill Evolver pipeline shared by /skill-mine and /direct-skill-generation.

:func:`run_thread` is the only implementation of the stage sequence; both
commands build a :class:`RunContext` once per run and hand it a
:class:`ThreadInput` per thread. The stages, in order:

1. gate 1 (``features.gate_decision``, demo-aware) with the muted
   ``Evidence score`` / ``Gate`` lines;
2. the evidence bundle under the configured budgets, the second gate over
   the kept episodes (``report_bundle``);
3. the stored experience-graph appendix (fail-open);
4. the LLM behind a :class:`~msagent.skill_evolver.budget.CountingLlm` and
   the :class:`~msagent.skill_evolver.budget.ContextBudget` of its window;
5. the classify template with the library snapshot and exactly one
   selection-policy block substituted (``str.replace``);
6. one context fit: the bundle is cut once to the room the prompt leaves,
   or the thread stops with ``insufficient_context_budget``;
7. classify round 1 (contract v2) with the muted ``Decision`` /
   ``Candidate`` lines and the ``invalid_evidence`` code rejections;
8. at most one ``expand_context_once`` round, only when the model said
   ``nothing`` for ``insufficient_evidence`` and the expanded bundle shows a
   strict superset of the fragments shown before;
9. the ``Nothing to save`` message by cause;
10. render plans, with the referenced skills' text read for the coverage
    note (the classifier's claim is never confirmed);
11. the plan loop: :func:`render_plan` (render, validate, semantic review,
    one correction, provenance v4, proposal) under the LLM call budget.

A decision report is written in ``finally`` for every non-dry run
(``RunContext.report_dir``), also when the gate refused the thread or the
thread failed; a thread /skill-mine refused before the loop gets its
gate-only report through :func:`record_gate_refusal`. Classify
parse/contract errors are recorded as code rejections and re-raised, so the
handlers count the thread as failed.

Every console string that existed before this module lives here unchanged
for non-demo runs. Pre-gate lines are muted (``sink.print``), never
``print_info``. This module never imports ``mining`` or
``direct_skill_generation``.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from collections.abc import Awaitable, Callable, Collection, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.markup import escape

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import theme
from msagent.core.logging import get_logger
from msagent.skill_evolver.budget import (
    MIN_BUNDLE_CHARS,
    ContextBudget,
    CountingLlm,
    LlmBudgetExhausted,
    llm_call_bound,
)
from msagent.skill_evolver.bundle import EXCLUDED_CODE, BundleEpisode, EvidenceBundle, build_evidence_bundle
from msagent.skill_evolver.classify import (
    BUNDLE_PLACEHOLDER,
    CODE_CONTRACT_ERROR,
    CODE_INVALID_EVIDENCE,
    CODE_PARSE_ERROR,
    LIBRARY_PLACEHOLDER,
    POLICY_PLACEHOLDER,
    Candidate,
    Classification,
    ClassifyContractError,
    ClassifyParseError,
    classify,
)
from msagent.skill_evolver.config import (
    CONTRACT_VERSION,
    SCHEMA_VERSION,
    STAGES,
    DirectSkillGenerationConfig,
    EffectiveRules,
    SkillEvolverConfigError,
)
from msagent.skill_evolver.exgraph_context import attach_stored_graph
from msagent.skill_evolver.features import (
    GATE_NO_EPISODES,
    DetectorNote,
    Episode,
    GateDecision,
    expand_episode_context,
    extract_episodes,
    gate_decision,
    mine_cross_session,
)
from msagent.skill_evolver.policy import (
    DEMO_NAME_PREFIX,
    render_policy_block,
    review_policy_block,
    selection_policy_block,
)
from msagent.skill_evolver.prompts import PromptContractError, PromptText, StagePrompts, prompt_sha256
from msagent.skill_evolver.render import (
    INSUFFICIENT_CONTEXT_BUDGET,
    NO_EXISTING_SKILL,
    EvidenceSelection,
    RenderPlan,
    RenderPlans,
    format_candidates,
    format_existing_skill,
    plan_render,
    resolve_library_skill,
)
from msagent.skill_evolver.report import draft_file_names, is_synthetic, write_report
from msagent.skill_evolver.retrieval import BM25Index
from msagent.skill_evolver.review import (
    QUALITY_REVIEW_FAILED,
    RENDER_INVALID,
    VERIFICATION_EVIDENCE_SUPPORTED,
    format_review_candidates,
    render_and_review,
)
from msagent.skill_evolver.validator import skill_name
from msagent.skill_evolver.writer import SKILL_FILE, SecretsDetected, build_provenance, scan_package, write_proposal
from msagent.skills.factory import Skill
from msagent.trajectory_recorder.reader import Trajectory

logger = get_logger(__name__)

REJECTED = "SKILL.md rejected after one correction; nothing written:"
# Characters of an existing skill handed to the renderer and the reviewer.
EXISTING_SKILL_MAX_CHARS = 20000
# Code rejections raised by this module (the others come from classify, render, review, bundle).
CODE_BUDGET_EXHAUSTED = "budget_exhausted"
CODE_SECRETS_DETECTED = "secrets_detected"
CODE_EXISTING_SKILL_UNREADABLE = "existing_skill_unreadable"
# Coverage of a ``reference`` candidate: the skill text was read, or could not be; never "covered".
COVERAGE_TEXT_READ = "text_read"
COVERAGE_UNVERIFIED = "unverified"
CONFIG_FILE_SOURCE = "config.skill.evolver.yml"
CONFIG_HINT = (
    "Fix ~/.msagent/config/config.skill.evolver.yml (schema_version: 2) or delete it to use the packaged defaults; "
    "nothing was run."
)
DEMO_OVERLAY_LINE = (
    "Demo overlay: gate bypass (demo_override), observed_procedure extraction, target create only, "
    "names demo-<task-class>"
)
EXPAND_NOT_RETRIED = "expand_context_once: no new context could be shown; not retried"
LLM_NOTE = "transport retries not counted"

PromptLoader = Callable[[Path, DirectSkillGenerationConfig, str], Awaitable[tuple[str, str]]]


# ------------------------------------------------------------------- data


@dataclass(slots=True)
class LazyLlm:
    """Builds the LLM on first use, so a dry or fully gated run creates none."""

    session: Any
    llm: Any = None
    model: str = ""
    context_window: int | None = None

    async def get(self) -> Any:
        """The analysis model of the session, created once per run."""
        if self.llm is None:
            ctx = self.session.context
            config = await initializer.load_llm_config(ctx.model, ctx.working_dir)
            self.llm = initializer.llm_factory.create(config)
            self.model = config.model
            self.context_window = getattr(config, "context_window", None)
        return self.llm


@dataclass(frozen=True, slots=True)
class RunContext:
    """What every thread of one run shares."""

    # "skill-mine" | "direct-skill-generation"
    command: str
    working_dir: Path
    state_dir: Path
    requested: DirectSkillGenerationConfig
    rules: EffectiveRules
    prompts: StagePrompts
    skills: list[Skill]
    llm_slot: LazyLlm
    # Names a new skill must not reuse; mutated by render_plan across threads.
    taken: set[str]
    # The handler's console at call time (patched spies are honoured).
    sink: Any
    # None -> no decision report (dry run or diagnostics.save_decision_report false).
    report_dir: Path | None


@dataclass(frozen=True, slots=True)
class ThreadInput:
    """One thread as the code-only stage delivered it."""

    trajectory: Trajectory
    supporting: list[Trajectory]
    episodes: list[Episode]
    notes: list[DetectorNote] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class StageOutcome:
    stage: str
    status: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CodeRejection:
    """Something code refused: ``code`` names the rule, ``subject`` the candidate, plan or thread."""

    code: str
    subject: str
    detail: str


@dataclass(frozen=True, slots=True)
class PlanContext:
    """What every render plan of one thread shares: provenance inputs and the output root."""

    thread_id: str
    thread_ids: list[str]
    bundle: EvidenceBundle
    # The thread's classify rejections, recorded in every proposal's provenance.
    rejected: list[tuple[Candidate, str]]
    sources: dict[str, str]
    model: str
    prompt_variants: dict[str, str]
    category: str
    output_root: Path
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    contract_version: int = CONTRACT_VERSION
    rules: EffectiveRules | None = None
    # ``DirectSkillGenerationConfig.as_record()`` of the requested config.
    requested: dict[str, Any] = field(default_factory=dict)
    # The library, for ``covered_by`` resolution.
    skills: Sequence[Skill] = ()


@dataclass(frozen=True, slots=True)
class PlanOutcome:
    """One plan's result: the written proposal, or why nothing was written."""

    plan: RenderPlan
    calls: int
    skill_path: Path | None = None
    name: str | None = None
    # Validator errors of the last render try (render_invalid), or the rejection's details.
    errors: list[str] = field(default_factory=list)
    rejection: CodeRejection | None = None
    # ``RenderedSkill.review_record()`` when a review happened.
    review: dict[str, Any] | None = None
    verification: dict[str, str] = field(default_factory=lambda: dict(VERIFICATION_EVIDENCE_SUPPORTED))
    # Non-fatal notes for the handler's console (render_plan itself never prints).
    warnings: list[str] = field(default_factory=list)
    # The refused plan's last SKILL.md draft (render_invalid, quality review failed); None when nothing was rendered.
    draft: str | None = None

    @property
    def written(self) -> bool:
        return self.skill_path is not None


@dataclass(slots=True)
class PlanTally:
    """Counters over the render plans of one thread (or, summed, of a run)."""

    # Plans that ran: proposals + render_errors.
    plans: int = 0
    proposals: int = 0
    # A double validation failure, a rejection or an exception inside the plan.
    render_errors: int = 0
    # Candidates whose target was unknown or ambiguous.
    rejected: int = 0
    deferred: int = 0

    @property
    def flagged(self) -> bool:
        """Whether anything besides clean proposals happened."""
        return bool(self.render_errors or self.rejected or self.deferred)

    def add(self, other: PlanTally) -> None:
        self.plans += other.plans
        self.proposals += other.proposals
        self.render_errors += other.render_errors
        self.rejected += other.rejected
        self.deferred += other.deferred

    def describe(self) -> str:
        """The plan-level part of a summary line."""
        return f"{self.render_errors} render errors, {self.rejected} rejected targets, {self.deferred} deferred"


@dataclass(slots=True)
class ThreadResult:
    """Everything one thread's run decided; the decision report is built from it."""

    thread_id: str
    tally: PlanTally = field(default_factory=PlanTally)
    stages: list[StageOutcome] = field(default_factory=list)
    code_rejections: list[CodeRejection] = field(default_factory=list)
    gate: dict[str, Any] | None = None
    second_gate: dict[str, Any] | None = None
    # One entry per bundle build (round 1, context cut, expand round).
    bundle_stats: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    classifier_verdict: str | None = None
    candidate_count: int = 0
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    quality_reviews: list[dict[str, Any]] = field(default_factory=list)
    coverage: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)
    llm_calls: int = 0
    llm_limit: int = 0
    budget_stop: bool = False
    context_window: int | None = None
    stop_message: str | None = None
    failed: str | None = None
    report_path: Path | None = None
    # bundle.text per round; kept only under diagnostics.save_evidence_text.
    evidence_texts: list[str] = field(default_factory=list)
    # ``{plan, code, content}`` of every refused plan's last SKILL.md draft (diagnostics.save_rejected_drafts);
    # written next to the decision report, whose siblings ``rejected_draft_paths`` then names.
    rejected_drafts: list[dict[str, str]] = field(default_factory=list)
    rejected_draft_paths: list[Path] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ExistingSkillText:
    """A library skill's SKILL.md as read for an update, a reference or a covering skill."""

    display_name: str
    path: Path
    # Hex sha256 of the raw bytes (``target.base_sha256`` of an update).
    sha256: str
    text: str
    truncated: bool


# ---------------------------------------------------------- code-only helpers


def collect_episodes(
    current: Trajectory,
    others: list[Trajectory],
    *,
    skill_index: BM25Index,
    demo: bool = False,
    notes: list[DetectorNote] | None = None,
) -> list[Episode]:
    """Per-trajectory episodes plus the cross-session patterns the thread supports.

    ``current`` goes first into the mining, so every shared pattern it takes
    part in cites its own events and the evidence bundle needs only this
    trajectory. ``demo`` adds the observed_procedure detector (drop reasons
    appended to ``notes``).
    """
    episodes = extract_episodes(current, skill_index=skill_index, demo=demo, notes=notes)
    if not others:
        return episodes
    shared = mine_cross_session([current, *others])
    episodes.extend(e for e in shared if e.thread_id == current.thread_id)
    return episodes


def supporting(others: list[Trajectory], episodes: list[Episode]) -> list[Trajectory]:
    """The trajectories among ``others`` that the episodes cite (a shared procedure's second session)."""
    cited = {item.ref.source for episode in episodes for item in episode.evidence}
    return [traj for traj in others if traj.source in cited]


def second_gate(bundle: EvidenceBundle, *, min_score: float, demo: bool = False) -> GateDecision | None:
    """The gate decided again over the kept episodes; None when nothing was excluded.

    Required items are never trimmed, so without an exclusion the kept set
    equals the gated set and the outcome cannot change.
    """
    if all(item.status != "excluded" for item in bundle.episodes):
        return None
    return gate_decision(bundle.kept, min_score=min_score, demo=demo)


def report_bundle(bundle: EvidenceBundle, *, min_score: float, demo: bool = False) -> tuple[list[str], str | None]:
    """One warning per excluded episode, and why nothing is saved when the kept ones fail the gate.

    An episode is excluded when even its required evidence does not fit
    the bundle budget (insufficient context). Its weight must not carry
    the thread to the LLM, so the gate is decided again on the kept
    episodes; a ``None`` second value means the pipeline goes on. The
    caller prints both (each handler has its own console).
    """
    excluded = [item.episode for item in bundle.episodes if item.status == "excluded"]
    warnings = [_exclusion_line(item) for item in bundle.episodes if item.status == "excluded"]
    decision = second_gate(bundle, min_score=min_score, demo=demo)
    if decision is None or decision.passes:
        return warnings, None
    stop = (
        f"Nothing to save: {len(excluded)} episodes excluded from the bundle; "
        f"remaining evidence score {decision.score:.2f} < min_evidence_score {min_score:.2f}"
    )
    if decision.detail:
        stop = f"{stop}; {decision.detail}"
    return warnings, stop


def _exclusion_line(item: BundleEpisode) -> str:
    """The warning for one excluded episode, with the characters it needed and the budget's leftover."""
    episode = item.episode
    line = f"Excluded {episode.kind} (weight {episode.weight:.2f}): insufficient context for the bundle budget"
    if item.needed is not None and item.room is not None:
        line += f" (needs {item.needed} chars, {item.room} left)"
    return line


def gate_lines(decision: GateDecision, *, min_score: float, episodes: Sequence[Episode]) -> list[str]:
    """The two muted lines both commands print for a gate decision (also in dry runs)."""
    counts = f"{len(episodes)} episodes, {len(decision.incidents)} incidents"
    return [
        f"Evidence score: {decision.score:.2f} (min_evidence_score {min_score:.2f}; {counts})",
        f"Gate: {'pass' if decision.passes else 'skip'} ({decision.describe()})",
    ]


def gate_refusal(decision: GateDecision, *, min_score: float, episodes: Sequence[Episode], thread_id: str) -> str:
    """The ``Nothing to save`` line of a refused gate; the demo detail is appended when present."""
    if decision.reason == GATE_NO_EPISODES:
        return f"Nothing to save: no episodes detected in thread {thread_id}"
    score = f"{decision.score:.2f} < min_evidence_score {min_score:.2f}"
    counts = f"{len(episodes)} episodes, {len(decision.incidents)} incidents"
    message = f"Nothing to save: evidence score {score} ({counts})"
    return f"{message}; {decision.detail}" if decision.detail else message


def cited_threads(current: Trajectory, episodes: list[Episode]) -> list[str]:
    """The analysed thread first, then every thread a shared procedure relies on."""
    others: set[str] = set()
    for episode in episodes:
        others.update(episode.facts.get("thread_ids", []))
    others.discard(current.thread_id)
    return [current.thread_id, *sorted(others)]


def report_plans(plans: RenderPlans, coverage: Sequence[str] = ()) -> tuple[list[str], list[str]]:
    """Warnings (rejected targets) and notes (references, deferred plans) for the console.

    ``coverage`` is aligned with ``plans.references`` (COVERAGE_TEXT_READ or
    COVERAGE_UNVERIFIED); a reference is the classifier's claim and is never
    reported as verified coverage. The caller prints both.
    """
    warnings = [f"Rejected target '{item.candidate.title}': {item.code} — {item.detail}" for item in plans.rejected]
    notes: list[str] = []
    for index, (candidate, skill) in enumerate(plans.references):
        state = coverage[index] if index < len(coverage) else COVERAGE_UNVERIFIED
        read = "read" if state == COVERAGE_TEXT_READ else "unreadable"
        notes.append(
            f"reference {skill.display_name}: {candidate.title} "
            f"(classifier's claim; skill text {read}, coverage not verified)"
        )
    notes.extend(f"Deferred plan: {plan.label} — {reason}" for plan, reason in plans.deferred)
    return warnings, notes


def skill_library_snapshot(skills: Sequence[Skill]) -> str:
    """Programmatic inventory injected as {skill_library} (replaces a tool call)."""
    if not skills:
        return "The skill library is currently empty."
    lines: list[str] = []
    for skill in sorted(skills, key=lambda s: s.display_name.casefold()):
        description = skill.description or "no description"
        lines.append(f"- {skill.display_name}: {description}")
    return "\n".join(lines)


def target_record(plan: RenderPlan, base_sha256: str | None = None) -> dict[str, str | None]:
    """Provenance ``target``: new skill, or the library skill being revised (with the hash of its file)."""
    if plan.existing is None:
        return {"action": "create", "existing_skill": None, "existing_path": None}
    return {
        "action": "update",
        "existing_skill": plan.existing.display_name,
        "existing_path": str(plan.existing.path),
        "base_sha256": base_sha256,
    }


def activation_hint(skill_path: Path, name: str, plan: RenderPlan, library_dir: Path) -> str:
    """Tell the user the proposal is inactive and how to promote it by hand."""
    folder = skill_path.parent
    if plan.existing is None:
        return f"Not active: review it, then move {folder} to {library_dir / name}"
    return f"Not active: review it, then replace {plan.existing.path} with it"


def read_existing_skill(skill: Skill, *, max_chars: int = EXISTING_SKILL_MAX_CHARS) -> ExistingSkillText | None:
    """The skill's SKILL.md: sha256 of the raw bytes and its text clipped to ``max_chars``; None when unreadable."""
    try:
        raw = skill.path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        logger.warning("Cannot read skill %s at %s", skill.display_name, skill.path, exc_info=True)
        return None
    truncated = len(text) > max_chars
    if truncated:
        extra = len(text) - max_chars
        text = text[:max_chars] + f"\n… [truncated: {extra} more characters]"
    return ExistingSkillText(skill.display_name, skill.path, hashlib.sha256(raw).hexdigest(), text, truncated)


def covering_skill_record(covered_by: str | None, skills: Sequence[Skill]) -> tuple[dict[str, Any] | None, str | None]:
    """Provenance ``covering_skill`` for a candidate's ``covered_by``, plus a warning when it names no skill."""
    if not covered_by:
        return None, None
    skill = resolve_library_skill(covered_by, skills)
    if skill is None:
        warning = f"covered_by '{covered_by}' names no library skill; recorded as the classifier's claim"
        return {"display_name": covered_by, "path": None, "sha256": None}, warning
    existing = read_existing_skill(skill)
    sha256 = existing.sha256 if existing is not None else None
    return {"display_name": skill.display_name, "path": str(skill.path), "sha256": sha256}, None


def reference_coverage(references: Sequence[tuple[Candidate, Skill]]) -> list[dict[str, Any]]:
    """Read every referenced skill's text; coverage stays the classifier's claim (report ``coverage``)."""
    rows: list[dict[str, Any]] = []
    for candidate, skill in references:
        existing = read_existing_skill(skill)
        rows.append(
            {
                "skill": skill.display_name,
                "title": candidate.title,
                "coverage": COVERAGE_TEXT_READ if existing is not None else COVERAGE_UNVERIFIED,
                "sha256": existing.sha256 if existing is not None else None,
            },
        )
    return rows


# ------------------------------------------------------------- run helpers


def muted(sink: Any, text: str) -> None:
    """A muted console line (never an ``info`` entry: tests count those)."""
    sink.print(f"[muted]{escape(text)}[/muted]")


def status(sink: Any, text: str):
    """Spinner shown while a pipeline stage runs."""
    color = theme.spinner_color
    return sink.console.status(f"[{color}]{text}[/{color}]")


async def load_prompts(loader: PromptLoader, root: Path, cfg: DirectSkillGenerationConfig) -> StagePrompts:
    """The three stage prompts through the handler's (patchable) loader; hashes of the returned text."""
    texts: dict[str, PromptText] = {}
    for stage in STAGES:
        text, source = await loader(root, cfg, stage)
        texts[stage] = PromptText(stage, text, source, prompt_sha256(text), cfg.contract_version)
    return StagePrompts(**texts)


def print_policy_block(sink: Any, requested: DirectSkillGenerationConfig, rules: EffectiveRules) -> None:
    """The muted lines that say which policy runs and where the values came from."""
    overrides = list(requested.overrides)
    policy_source = "--policy" if any(o.startswith("--policy") for o in overrides) else CONFIG_FILE_SOURCE
    demo_source = next((o for o in overrides if o in ("--demo", "--no-demo")), CONFIG_FILE_SOURCE)
    muted(sink, f"Requested policy: {rules.policy} ({policy_source})")
    muted(sink, f"Demo mode: {'true' if rules.demo else 'false'} ({demo_source})")
    muted(sink, f"Effective selection: {rules.selection}")
    if rules.demo:
        muted(sink, DEMO_OVERLAY_LINE)
    for note in requested.notes:
        muted(sink, f"Config note: {note}")


def report_config_error(sink: Any, exc: SkillEvolverConfigError | PromptContractError) -> None:
    """Print a config or prompt-contract problem and the fix hint; the caller returns without running."""
    lines = exc.lines() if isinstance(exc, SkillEvolverConfigError) else [str(exc)]
    for line in lines:
        sink.print_error(escape(line))
    muted(sink, CONFIG_HINT)


def bundle_preview(thread: ThreadInput, rules: EffectiveRules) -> list[str]:
    """Dry-run lines: what the bundle would hold, the excluded episodes, the observed procedures."""
    bundle = build_evidence_bundle(
        thread.episodes,
        [thread.trajectory, *thread.supporting],
        max_chars=rules.bundle_max_chars,
        excerpt_chars=rules.excerpt_max_chars,
        demo=rules.demo,
    )
    statuses = Counter(item.status for item in bundle.episodes)
    lines = [
        f"bundle: {len(bundle.shown)} fragments, {len(bundle.text)} chars (cap {rules.bundle_max_chars}); "
        f"episodes shown {statuses['shown']}, trimmed {statuses['trimmed']}, excluded {statuses['excluded']}",
    ]
    lines.extend(
        f"excluded: {item.episode.kind} (weight {item.episode.weight:.2f})"
        for item in bundle.episodes
        if item.status == "excluded"
    )
    observed = [episode for episode in thread.episodes if episode.kind == "observed_procedure"]
    lines.extend(
        f"observed_procedure: {' → '.join(episode.tool_sequence)} ({len(episode.evidence)} events)"
        for episode in observed
    )
    if rules.demo:
        dropped = Counter(note.reason for note in thread.notes)
        summary = ", ".join(f"{reason}={count}" for reason, count in sorted(dropped.items())) or "none"
        lines.append(f"observed_procedure candidates: {len(observed)} selected; dropped: {summary}")
    return lines


# ------------------------------------------------------------ render_plan


def _evidence_room(
    budget: ContextBudget,
    prompts: StagePrompts,
    candidates: Sequence[Candidate],
    existing_text: str | None,
    render_policy: str,
    review_policy: str,
) -> int | None:
    """Characters the quoted evidence may take: the larger of the render and review prompts' overhead.

    The review prompt also carries the SKILL.md itself, which the budget's
    reply reserve covers; None when the window is unknown.
    """
    if budget.available_chars is None:
        return None
    empty = [EvidenceSelection([], [], []) for _ in candidates]
    existing = existing_text or NO_EXISTING_SKILL
    render_overhead = (
        len(prompts.render.text)
        + len(render_policy)
        + len(format_candidates(candidates, selections=empty))
        + len(existing)
    )
    review_overhead = (
        len(prompts.review.text) + len(review_policy) + len(format_review_candidates(candidates)) + len(existing)
    )
    return budget.room_for(max(render_overhead, review_overhead))


async def render_plan(
    plan: RenderPlan,
    *,
    llm: Any,
    prompts: StagePrompts,
    context: PlanContext,
    taken: set[str],
    budget: ContextBudget,
    rules: EffectiveRules,
) -> PlanOutcome:
    """Render one plan, review it, write its proposal; never prints (the handlers do).

    A create checks its name against ``taken`` and, once written, adds the
    new name to it; an update passes ``expected_name`` instead, leaves
    ``taken`` alone and records the hash of the file it revises. Code
    refusals come back as ``rejection`` (nothing written); a SKILL.md the
    validator refused twice comes back with ``errors`` only (the caller
    prints REJECTED). :class:`LlmBudgetExhausted` propagates.
    """
    existing_text: str | None = None
    expected_name: str | None = None
    taken_names: Collection[str] = ()
    base_sha256: str | None = None
    if plan.existing is None:
        taken_names = taken
    else:
        existing = await asyncio.to_thread(read_existing_skill, plan.existing)
        if existing is None:
            detail = (
                f"existing skill {plan.existing.display_name} at {plan.existing.path} is unreadable "
                f"({CODE_EXISTING_SKILL_UNREADABLE}); nothing written"
            )
            rejection = CodeRejection(CODE_EXISTING_SKILL_UNREADABLE, plan.label, detail)
            return PlanOutcome(plan, 0, errors=[detail], rejection=rejection)
        existing_text = format_existing_skill(plan.existing.display_name, existing.text)
        expected_name = plan.existing.name
        base_sha256 = existing.sha256

    render_policy = render_policy_block(rules.demo)
    review_policy = review_policy_block(rules.demo)
    rendered = await render_and_review(
        plan.candidates,
        llm=llm,
        render_template=prompts.render.text,
        review_template=prompts.review.text,
        render_policy=render_policy,
        review_policy=review_policy,
        evidence=context.bundle.shown,
        existing_skill=existing_text,
        existing_skill_text=existing_text,
        expected_name=expected_name,
        taken_names=taken_names,
        required_prefix=DEMO_NAME_PREFIX if rules.demo and plan.existing is None else None,
        evidence_budget_chars=_evidence_room(
            budget, prompts, plan.candidates, existing_text, render_policy, review_policy
        ),
    )
    if rendered.code == RENDER_INVALID:
        return PlanOutcome(plan, rendered.calls, errors=list(rendered.errors), draft=rendered.draft)
    review = rendered.review_record() if rendered.review is not None else None
    if rendered.code is not None:
        rejection = CodeRejection(rendered.code, plan.label, "; ".join(rendered.errors))
        return PlanOutcome(
            plan, rendered.calls, errors=list(rendered.errors), rejection=rejection, review=review, draft=rendered.draft
        )

    name = skill_name(rendered.content)
    covered_by = next((c.covered_by for c in plan.candidates if c.covered_by), None)
    covering, warning = await asyncio.to_thread(covering_skill_record, covered_by, context.skills)
    warnings = [warning] if warning else []
    overrides = context.requested.get("overrides", [])
    policy_source = "--policy" if any(str(o).startswith("--policy") for o in overrides) else "config"
    provenance = build_provenance(
        thread_ids=context.thread_ids,
        bundle=context.bundle,
        candidates=plan.candidates,
        rejected=context.rejected,
        sources=context.sources,
        model=context.model,
        prompt_variants=context.prompt_variants,
        category=context.category,
        target=target_record(plan, base_sha256),
        policy={"requested": rules.policy, "selection": rules.selection, "source": policy_source},
        demo=rules.demo,
        config={"requested": context.requested, "effective": rules.as_record()},
        versions={
            "config_schema": int(context.requested.get("schema_version", SCHEMA_VERSION)),
            "prompts_contract": context.contract_version,
        },
        prompt_hashes=context.prompt_hashes,
        quality_review=rendered.review_record(),
        verification=VERIFICATION_EVIDENCE_SUPPORTED,
        covering_skill=covering,
        render_evidence=rendered.render_evidence,
        render_evidence_omitted=rendered.render_evidence_omitted,
    )
    try:
        skill_path = await asyncio.to_thread(
            write_proposal,
            rendered.content,
            root=context.output_root,
            name=name,
            provenance=provenance,
            thread_id=context.thread_id,
        )
    except SecretsDetected as exc:
        rejection = CodeRejection(CODE_SECRETS_DETECTED, plan.label, str(exc))
        return PlanOutcome(
            plan, rendered.calls, errors=[str(exc)], rejection=rejection, review=review, warnings=warnings
        )
    if plan.existing is None:
        taken.add(name)
    return PlanOutcome(plan, rendered.calls, skill_path=skill_path, name=name, review=review, warnings=warnings)


def rejection_lines(label: str, outcome: PlanOutcome) -> list[str]:
    """The error lines of a rejected plan, by rejection code."""
    rejection = outcome.rejection
    if rejection is None:
        return []
    if rejection.code == QUALITY_REVIEW_FAILED:
        invalid = [e.removeprefix("reviewer reply invalid: ") for e in outcome.errors if e.startswith("reviewer reply")]
        if invalid:
            return [f"plan {label}: quality reviewer reply invalid; nothing written: {'; '.join(invalid)}"]
        head = f"plan {label}: quality review failed after one correction; nothing written:"
        return [head, *(f"  - {error}" for error in outcome.errors)]
    if rejection.code == INSUFFICIENT_CONTEXT_BUDGET:
        head = f"plan {label}: required evidence does not fit the render budget ({INSUFFICIENT_CONTEXT_BUDGET}); nothing written:"
        return [head, *(f"  - {error}" for error in outcome.errors)]
    return [f"plan {label}: {rejection.detail}"]


# -------------------------------------------------------------- run_thread


def _gate_record(decision: GateDecision, min_score: float) -> dict[str, Any]:
    return {
        "score": decision.score,
        "min_evidence_score": min_score,
        "passes": decision.passes,
        "reason": decision.reason,
        "detail": decision.detail,
        "incidents": len(decision.incidents),
    }


def _bundle_stats(bundle: EvidenceBundle, *, round_number: int, max_chars: int) -> dict[str, Any]:
    statuses = Counter(item.status for item in bundle.episodes)
    return {
        "round": round_number,
        "fragments": len(bundle.shown),
        "chars": len(bundle.text),
        "max_chars": max_chars,
        "shown": statuses["shown"],
        "trimmed": statuses["trimmed"],
        "excluded": statuses["excluded"],
        "excluded_kinds": [item.episode.kind for item in bundle.episodes if item.status == "excluded"],
    }


class _ThreadRun:
    """The stages of one thread; state that several stages share lives here."""

    def __init__(self, run: RunContext, thread: ThreadInput, result: ThreadResult) -> None:
        self.run = run
        self.thread = thread
        self.result = result
        self.rules = run.rules
        self.sink = run.sink
        self.trajectories = [thread.trajectory, *thread.supporting]
        self.max_chars = run.rules.bundle_max_chars
        self.rounds = 0
        self.llm: CountingLlm | None = None
        self.budget = ContextBudget.for_window(None)
        self.template = ""

    @property
    def thread_id(self) -> str:
        return self.thread.trajectory.thread_id

    def _stop(self, message: str) -> None:
        self.sink.print_info(message)
        self.result.stop_message = message

    async def run_stages(self) -> None:
        rules, episodes = self.rules, self.thread.episodes
        decision = gate_decision(episodes, min_score=rules.min_evidence_score, demo=rules.demo)
        self.result.gate = _gate_record(decision, rules.min_evidence_score)
        for line in gate_lines(decision, min_score=rules.min_evidence_score, episodes=episodes):
            muted(self.sink, line)
        self.result.stages.append(StageOutcome("gate", "pass" if decision.passes else "skip", decision.describe()))
        if not decision.passes:
            self._stop(
                gate_refusal(decision, min_score=rules.min_evidence_score, episodes=episodes, thread_id=self.thread_id),
            )
            return

        built = self._admit(self._build(episodes))
        if built is None:
            return
        raw = await self.run.llm_slot.get()
        self.llm = CountingLlm(raw, limit=rules.max_llm_calls)
        self.result.llm_limit = rules.max_llm_calls
        self.result.context_window = self.run.llm_slot.context_window
        self.budget = ContextBudget.for_window(self.run.llm_slot.context_window)
        try:
            await self._llm_stages(*built)
        finally:
            self.result.llm_calls = self.llm.calls_used

    # ------------------------------------------------------------ bundle

    def _build(self, episodes: list[Episode]) -> EvidenceBundle:
        return build_evidence_bundle(
            episodes,
            self.trajectories,
            max_chars=self.max_chars,
            excerpt_chars=self.rules.excerpt_max_chars,
            demo=self.rules.demo,
        )

    def _admit(self, bundle: EvidenceBundle) -> tuple[EvidenceBundle, str] | None:
        """Record a bundle, warn about exclusions, re-gate; the text with the graph appendix, or None on stop."""
        rules, result = self.rules, self.result
        self.rounds += 1
        result.bundle_stats.append(_bundle_stats(bundle, round_number=self.rounds, max_chars=self.max_chars))
        if rules.save_evidence_text:
            result.evidence_texts.append(bundle.text)
        warnings, stop = report_bundle(bundle, min_score=rules.min_evidence_score, demo=rules.demo)
        for line in warnings:
            self.sink.print_warning(escape(line))
        for item in bundle.episodes:
            if item.status == "excluded":
                detail = "required evidence does not fit the bundle budget"
                rejection = CodeRejection(EXCLUDED_CODE, item.episode.kind, detail)
                # An episode excluded again after the cut or in the expand round is recorded once.
                if rejection not in result.code_rejections:
                    result.code_rejections.append(rejection)
        regate = second_gate(bundle, min_score=rules.min_evidence_score, demo=rules.demo)
        if regate is not None:
            result.second_gate = _gate_record(regate, rules.min_evidence_score)
        detail = f"{len(bundle.shown)} fragments, {len(bundle.text)} chars"
        result.stages.append(StageOutcome("bundle", "stop" if stop is not None else "ok", detail))
        if stop is not None:
            self._stop(stop)
            return None
        text = attach_stored_graph(
            bundle.text,
            self.thread.trajectory,
            working_dir=self.run.working_dir,
            state_dir=self.run.state_dir,
        )
        return bundle, text

    def _fit(self, bundle: EvidenceBundle, text: str, episodes: list[Episode]) -> tuple[EvidenceBundle, str] | None:
        """Cut the bundle once to the room the classify prompt leaves, or stop the thread."""
        overhead = len(self.template) - len(BUNDLE_PLACEHOLDER) + (len(text) - len(bundle.text))
        room = self.budget.room_for(overhead)
        if room is None or len(bundle.text) <= room:
            return bundle, text
        if room < MIN_BUNDLE_CHARS:
            detail = f"bundle needs {len(bundle.text)} chars, {room} available"
            self.result.code_rejections.append(CodeRejection(INSUFFICIENT_CONTEXT_BUDGET, "thread", detail))
            self.result.stages.append(StageOutcome("context_fit", "stop", detail))
            described = self.budget.describe()
            self._stop(f"Nothing to save: the classify prompt does not fit the model context ({described}); {detail}")
            return None
        self.sink.print_warning(
            escape(f"Bundle cut to {room} chars to fit the model context ({self.budget.describe()})")
        )
        self.result.stages.append(StageOutcome("context_fit", "cut", f"max_chars {room}"))
        self.max_chars = room
        return self._admit(self._build(episodes))

    # ---------------------------------------------------------- classify

    async def _llm_stages(self, bundle: EvidenceBundle, text: str) -> None:
        template = self.run.prompts.classify.text
        if POLICY_PLACEHOLDER not in template:
            raise ValueError(f"classify template has no {POLICY_PLACEHOLDER} placeholder")
        library = skill_library_snapshot(self.run.skills)
        policy = selection_policy_block(self.rules.selection)
        self.template = template.replace(LIBRARY_PLACEHOLDER, library).replace(POLICY_PLACEHOLDER, policy)

        fitted = self._fit(bundle, text, self.thread.episodes)
        if fitted is None:
            return
        bundle, text = fitted
        classification = await self._classify(bundle, text)
        if self._expand_wanted(classification):
            expanded = await self._expand(classification, bundle)
            if self.result.stop_message is not None:
                return
            if expanded is not None:
                classification, bundle = expanded
        if classification.verdict == "nothing":
            self._stop(self._nothing_message(classification))
            return
        await self._plans(classification, bundle)

    async def _classify(self, bundle: EvidenceBundle, text: str) -> Classification:
        rules, result, sink = self.rules, self.result, self.sink
        with status(sink, "Classifying evidence..."):
            classification = await classify(
                text,
                bundle.shown,
                self.llm,
                self.template,
                episode_ids=bundle.episode_ids,
                demo=rules.demo,
            )
        result.classifier_verdict = classification.verdict
        result.candidate_count = len(classification.candidates)
        result.decisions.extend(decision.model_dump() for decision in classification.decisions)
        result.rejected_candidates.extend(
            {"title": candidate.title, "reason": reason, "evidence_refs": list(candidate.evidence_refs)}
            for candidate, reason in classification.rejected
        )
        detail = f"{len(classification.candidates)} candidates, {classification.calls} calls"
        result.stages.append(StageOutcome("classify", classification.verdict, detail))
        for decision in classification.decisions:
            muted(sink, f"Decision {','.join(decision.episode_ids)}: {decision.decision} ({decision.reason_code})")
        suffix = " (trivial procedure allowed)" if rules.demo else ""
        for candidate in classification.candidates:
            muted(sink, f"Candidate '{candidate.title}': accepted{suffix}")
        for candidate, reason in classification.rejected:
            sink.print_warning(escape(f"Rejected '{candidate.title}': {reason}"))
            result.code_rejections.append(CodeRejection(CODE_INVALID_EVIDENCE, candidate.title, reason))
        return classification

    def _expand_wanted(self, classification: Classification) -> bool:
        return (
            classification.verdict == "nothing"
            and classification.insufficient_evidence
            and self.rules.on_nothing == "expand_context_once"
        )

    async def _expand(
        self,
        classification: Classification,
        bundle: EvidenceBundle,
    ) -> tuple[Classification, EvidenceBundle] | None:
        """One more round on the context-expanded episodes; None when not retried or stopped (stop_message set)."""
        rules, result, sink = self.rules, self.result, self.sink
        episodes = self.thread.episodes
        flagged = {
            episode_id
            for decision in classification.decisions
            if decision.reason_code == "insufficient_evidence"
            for episode_id in decision.episode_ids
        }
        only = {
            index
            for index, episode in enumerate(episodes)
            if any(bundle.episode_ids.get(episode_id) is episode for episode_id in flagged)
        }
        expanded = expand_episode_context(
            episodes,
            self.thread.trajectory,
            surrounding_events=rules.surrounding_events,
            only=only,
        )
        probe = self._build(expanded)
        before = {fragment.ref for fragment in bundle.shown.values()}
        # Cheap early exit on the uncut probe; the retry is decided on the bundle the model will see.
        if not {fragment.ref for fragment in probe.shown.values()} > before:
            sink.print_info(EXPAND_NOT_RETRIED)
            result.stages.append(StageOutcome("expand", "skipped", "no new context could be shown"))
            return None
        admitted = self._admit(probe)
        if admitted is None:
            return None
        fitted = self._fit(*admitted, expanded)
        if fitted is None:
            return None
        bundle2, text2 = fitted
        after = {fragment.ref for fragment in bundle2.shown.values()}
        if not after > before:
            sink.print_info(EXPAND_NOT_RETRIED)
            result.stages.append(StageOutcome("expand", "skipped", "no new context fits the model context"))
            return None
        added = len(after - before)
        sink.print_info(
            f"Insufficient evidence: one more classification round with {added} added events "
            f"({rules.surrounding_events} per side)",
        )
        result.stages.append(StageOutcome("expand", "retried", f"{added} new events"))
        return await self._classify(bundle2, text2), bundle2

    def _nothing_message(self, classification: Classification) -> str:
        if classification.insufficient_evidence:
            return f"Nothing to save: insufficient evidence in thread {self.thread_id} (classifier)"
        if classification.rejected:
            return f"Nothing to save: no candidate with valid evidence refs in thread {self.thread_id} ({CODE_INVALID_EVIDENCE})"
        return f"Nothing to save: no durable learning found in thread {self.thread_id}"

    # ------------------------------------------------------------- plans

    def _keep_draft(self, label: str, code: str, outcome: PlanOutcome) -> None:
        """Keep a refused plan's last draft for the decision report (diagnostics.save_rejected_drafts).

        A draft carrying a possible secret is dropped with a warning: the
        rule the proposal writer applies (nothing with a secret is written).
        """
        if outcome.draft is None or not self.rules.save_rejected_drafts:
            return
        findings = scan_package({SKILL_FILE: outcome.draft})
        if findings:
            detail = "; ".join(findings)
            self.sink.print_warning(escape(f"plan {label}: rejected draft not kept, possible secrets: {detail}"))
            return
        self.result.rejected_drafts.append({"plan": label, "code": code, "content": outcome.draft})

    async def _plans(self, classification: Classification, bundle: EvidenceBundle) -> None:
        run, rules, result, sink = self.run, self.rules, self.result, self.sink
        plans = plan_render(classification.candidates, run.skills, max_plans=rules.max_plans)
        result.coverage = await asyncio.to_thread(reference_coverage, plans.references)
        warnings, notes = report_plans(plans, [row["coverage"] for row in result.coverage])
        for line in warnings:
            sink.print_warning(escape(line))
        for line in notes:
            sink.print_info(escape(line))
        result.code_rejections.extend(
            CodeRejection(item.code, item.candidate.title, item.detail) for item in plans.rejected
        )
        tally = result.tally
        tally.rejected = len(plans.rejected)
        tally.deferred = len(plans.deferred)
        if not plans.plans:
            self._stop("Nothing to save: no candidate left to render")
            return

        context = PlanContext(
            thread_id=self.thread_id,
            thread_ids=cited_threads(self.thread.trajectory, self.thread.episodes),
            bundle=bundle,
            rejected=classification.rejected,
            sources={traj.source: str(traj.path) for traj in self.trajectories},
            model=run.llm_slot.model,
            prompt_variants=run.prompts.variants(),
            category=rules.category,
            output_root=rules.output_root,
            prompt_hashes=run.prompts.hashes(),
            # The same source as the report's prompts.contract_version.
            contract_version=run.prompts.classify.contract_version,
            rules=rules,
            requested=run.requested.as_record(),
            skills=run.skills,
        )
        library_dir = rules.output_root / rules.category
        llm = self.llm
        total = len(plans.plans)
        for position, plan in enumerate(plans.plans, start=1):
            label = plan.label
            if llm.exhausted:
                tally.deferred += 1
                reason = f"llm call budget exhausted ({llm.calls_used}/{llm.limit})"
                sink.print_info(escape(f"Deferred plan: {label} — {reason}"))
                result.stages.append(StageOutcome(f"plan {label}", "deferred", reason))
                continue
            tally.plans += 1
            try:
                with status(sink, f"Rendering SKILL.md (plan {position}/{total}: {escape(label)})..."):
                    outcome = await render_plan(
                        plan,
                        llm=llm,
                        prompts=run.prompts,
                        context=context,
                        taken=run.taken,
                        budget=self.budget,
                        rules=rules,
                    )
            except LlmBudgetExhausted as exc:
                tally.render_errors += 1
                result.budget_stop = True
                result.code_rejections.append(CodeRejection(CODE_BUDGET_EXHAUSTED, label, str(exc)))
                sink.print_error(escape(f"plan {label}: {exc}; SKILL.md not written"))
                result.stages.append(StageOutcome(f"plan {label}", "rejected", CODE_BUDGET_EXHAUSTED))
                continue
            except Exception as exc:
                tally.render_errors += 1
                sink.print_error(escape(f"plan {label}: {exc}"))
                logger.exception("Rendering plan %s of thread %s failed", label, self.thread_id)
                result.stages.append(StageOutcome(f"plan {label}", "error", str(exc)))
                continue
            for line in outcome.warnings:
                sink.print_warning(escape(line))
            if outcome.review is not None:
                result.quality_reviews.append({"plan": label, **outcome.review})
            if outcome.rejection is not None:
                tally.render_errors += 1
                result.code_rejections.append(outcome.rejection)
                for line in rejection_lines(label, outcome):
                    sink.print_error(escape(line))
                result.stages.append(StageOutcome(f"plan {label}", "rejected", outcome.rejection.code))
                self._keep_draft(label, outcome.rejection.code, outcome)
                continue
            if not outcome.written:
                tally.render_errors += 1
                result.code_rejections.append(CodeRejection(RENDER_INVALID, label, "; ".join(outcome.errors)))
                sink.print_error(escape(f"plan {label}: {REJECTED}"))
                for error in outcome.errors:
                    sink.print_error(escape(f"  - {error}"))
                result.stages.append(StageOutcome(f"plan {label}", "rejected", RENDER_INVALID))
                self._keep_draft(label, RENDER_INVALID, outcome)
                continue
            tally.proposals += 1
            result.proposals.append(str(outcome.skill_path))
            sink.print_success(escape(f"Skill proposal saved to {outcome.skill_path}"))
            muted(sink, activation_hint(outcome.skill_path, outcome.name, plan, library_dir))
            corrected = bool(outcome.review and outcome.review.get("corrected"))
            muted(sink, "Quality review: passed after one correction" if corrected else "Quality review: passed")
            muted(sink, f"Verification: {outcome.verification['level']}; {outcome.verification['note']}")
            muted(sink, "Proposal: saved, inactive, DEMO" if rules.demo else "Proposal: saved, inactive")
            result.stages.append(StageOutcome(f"plan {label}", "written", str(outcome.skill_path)))


def build_report(run: RunContext, thread: ThreadInput, result: ThreadResult) -> dict[str, Any]:
    """The decision-report payload (see :mod:`msagent.skill_evolver.report` for the key list)."""
    rules, trajectory, tally = run.rules, thread.trajectory, result.tally
    expand = rules.on_nothing == "expand_context_once"
    return {
        "command": run.command,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thread_id": trajectory.thread_id,
        "source": trajectory.source,
        "synthetic": is_synthetic(trajectory.agent, str(trajectory.working_dir)),
        "policy": {
            "requested": rules.policy,
            "selection": rules.selection,
            "demo_mode": rules.demo,
            "overrides": list(run.requested.overrides),
        },
        "config": {"requested": run.requested.as_record(), "effective": rules.as_record()},
        "prompts": {
            "contract_version": run.prompts.classify.contract_version,
            "variants": run.prompts.variants(),
            "hashes": run.prompts.hashes(),
        },
        "episodes": {
            "total": len(thread.episodes),
            "counts": dict(sorted(Counter(episode.kind for episode in thread.episodes).items())),
            "detector_notes": [asdict(note) for note in thread.notes],
        },
        "gate": result.gate,
        "second_gate": result.second_gate,
        "bundles": list(result.bundle_stats),
        "stages": [asdict(stage) for stage in result.stages],
        "classifier": {
            "verdict": result.classifier_verdict,
            "candidates": result.candidate_count,
            "decisions": list(result.decisions),
            "rejected": list(result.rejected_candidates),
        },
        "code_rejections": [asdict(rejection) for rejection in result.code_rejections],
        "plans": {
            "rendered": tally.plans,
            "proposals": tally.proposals,
            "render_errors": tally.render_errors,
            "rejected_targets": tally.rejected,
            "deferred": tally.deferred,
        },
        "quality_review": list(result.quality_reviews),
        "coverage": list(result.coverage),
        "proposals": list(result.proposals),
        "llm": {
            "model": run.llm_slot.model,
            "context_window": result.context_window,
            "calls_used": result.llm_calls,
            "limit": rules.max_llm_calls,
            "bound": llm_call_bound(1, rules.max_plans, max_llm_calls=rules.max_llm_calls, expand=expand),
            "budget_exhausted": result.budget_stop,
            "note": LLM_NOTE,
        },
        "stop_message": result.stop_message,
        "failed": result.failed,
    }


def _write_report(run: RunContext, thread: ThreadInput, result: ThreadResult) -> None:
    evidence_text: str | None = None
    if run.rules.save_evidence_text and result.evidence_texts:
        rounds = enumerate(result.evidence_texts, start=1)
        evidence_text = "\n\n".join(f"# Round {number}\n\n{text}" for number, text in rounds)
    drafts = list(result.rejected_drafts)
    try:
        result.report_path = write_report(
            run.report_dir,
            thread_id=result.thread_id,
            payload=build_report(run, thread, result),
            evidence_text=evidence_text,
            rejected_drafts=drafts,
        )
        names = draft_file_names(result.report_path.stem, drafts)
        result.rejected_draft_paths = [result.report_path.with_name(name) for name in names]
    except Exception:
        logger.warning("Decision report of thread %s not written", result.thread_id, exc_info=True)


def draft_lines(result: ThreadResult) -> list[str]:
    """The muted lines naming the rejected-draft files a thread left next to its decision report."""
    return [f"Rejected SKILL.md draft kept for inspection: {path}" for path in result.rejected_draft_paths]


async def record_gate_refusal(run: RunContext, thread: ThreadInput) -> ThreadResult:
    """The gate-only result of a thread the caller refused before :func:`run_thread`; prints nothing.

    /skill-mine decides gate 1 for its threads table and hands only the
    passing threads to ``run_thread``; a refused one still leaves the
    decision report ``run_thread`` would have written at stage 1.
    """
    rules, episodes = run.rules, thread.episodes
    result = ThreadResult(thread_id=thread.trajectory.thread_id)
    decision = gate_decision(episodes, min_score=rules.min_evidence_score, demo=rules.demo)
    result.gate = _gate_record(decision, rules.min_evidence_score)
    result.stages.append(StageOutcome("gate", "skip", decision.describe()))
    result.stop_message = gate_refusal(
        decision, min_score=rules.min_evidence_score, episodes=episodes, thread_id=result.thread_id
    )
    if run.report_dir is not None:
        await asyncio.to_thread(_write_report, run, thread, result)
    return result


async def run_thread(run: RunContext, thread: ThreadInput) -> ThreadResult:
    """Run every stage for one thread; the decision report is written in ``finally``.

    Classify parse/contract errors are recorded (``parse_error`` /
    ``contract_error``) and re-raised; an LLM call budget spent before the
    plans (classify or the expand round) ends the thread with nothing
    written; any other exception marks the thread failed and propagates to
    the handler's guard.
    """
    result = ThreadResult(thread_id=thread.trajectory.thread_id)
    try:
        await _ThreadRun(run, thread, result).run_stages()
    except (ClassifyParseError, ClassifyContractError) as exc:
        code = CODE_PARSE_ERROR if isinstance(exc, ClassifyParseError) else CODE_CONTRACT_ERROR
        result.failed = f"{type(exc).__name__}: {exc}"
        result.code_rejections.append(CodeRejection(code, "classify", str(exc)))
        result.stages.append(StageOutcome("classify", "failed", str(exc)))
        raise
    except LlmBudgetExhausted as exc:
        result.budget_stop = True
        result.code_rejections.append(CodeRejection(CODE_BUDGET_EXHAUSTED, "classify", str(exc)))
        result.stages.append(StageOutcome("classify", CODE_BUDGET_EXHAUSTED))
        message = (
            f"LLM call budget exhausted ({exc.used}/{exc.limit} calls per thread) before classification finished; "
            "nothing written"
        )
        run.sink.print_error(escape(message))
        result.stop_message = message
    except Exception as exc:
        result.failed = f"{type(exc).__name__}: {exc}"
        result.stages.append(StageOutcome("thread", "failed", result.failed))
        raise
    finally:
        if run.report_dir is not None:
            await asyncio.to_thread(_write_report, run, thread, result)
    return result
