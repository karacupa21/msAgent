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

"""/skill-mine: mine several recorded threads into SKILL.md proposals.

Up to ``max_plans`` proposals per thread: every selected trajectory is turned
into episodes by code, gated by ``features.gate_decision`` (incident score
against ``min_evidence_score``, plus the strong correction rule and, in demo
mode, the observed-procedure override) and only then handed to
:func:`msagent.skill_evolver.pipeline.run_thread`, the stage sequence shared
with /direct-skill-generation; a refused thread still leaves its gate-only
decision report. ``--dry-run`` stops after the code-only stage
and prints the gate decision with its reason, the episode table with
incident labels and a bundle preview, so detector behaviour can be inspected
without spending a token. ``--policy``, ``--demo`` and ``--no-demo`` override
the config for one run.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from rich import box
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import console
from msagent.core.constants import SKILL_EVOLVER_CONFIG_FOLDER_NAME
from msagent.core.logging import get_logger
from msagent.skill_evolver.budget import llm_call_bound
from msagent.skill_evolver.cli_args import POLICY_USAGE, CliArgsError, SharedFlags, take_shared_flag
from msagent.skill_evolver.config import (
    CROSS_SESSION_LIMIT,
    DirectSkillGenerationConfig,
    EffectiveRules,
    SkillEvolverConfigError,
    apply_overrides,
    effective_rules,
)
from msagent.skill_evolver.direct_skill_generation import DirectSkillGenerationHandler
from msagent.skill_evolver.features import (
    DetectorNote,
    Episode,
    GateDecision,
    evidence_score,
    gate_decision,
    group_incidents,
)
from msagent.skill_evolver.pipeline import (
    LazyLlm,
    PlanTally,
    RunContext,
    ThreadInput,
    bundle_preview,
    collect_episodes,
    draft_lines,
    gate_lines,
    load_prompts,
    print_policy_block,
    record_gate_refusal,
    report_config_error,
    run_thread,
    status,
    supporting,
)
from msagent.skill_evolver.prompts import PromptContractError
from msagent.skill_evolver.report import decisions_dir
from msagent.skill_evolver.retrieval import BM25Index, SkillDoc
from msagent.skills.factory import Skill
from msagent.trajectory_recorder.export import (
    find_trajectory_file,
    resolve_trajectories_dir,
    workspace_filter,
)
from msagent.trajectory_recorder.reader import (
    Trajectory,
    TrajectoryReadError,
    load_trajectories,
    load_trajectory,
)

logger = get_logger(__name__)

USAGE = "/skill-mine [--threads N] [--since 7d] [--dry-run] [--thread <id>] " + POLICY_USAGE
# Threads mined by default: at most ``llm_call_bound(1, max_plans, ...)`` LLM
# calls each, and the evidence gate usually cuts that further. Cross-session
# support is unaffected by this number.
DEFAULT_THREADS = 5
# Thread ids are uuid4; this prefix stays unique in practice and is still a
# valid --thread argument (find_trajectory_file resolves unique prefixes).
THREAD_ID_WIDTH = 12
MAX_TOOLS_SHOWN = 5
MAX_SEQ_SHOWN = 8

# --since grammar. Minutes/months ('m') are rejected on purpose: the unit is
# ambiguous and guessing would mask a typo instead of reporting it.
_SINCE_RE = re.compile(r"^(\d+)([hdw])$")
_SINCE_SECONDS = {"h": 3600.0, "d": 86400.0, "w": 604800.0}

_NO_TOOL_EVENTS_NOTE = " ".join(
    (
        "Threads showing 0 tools have no tool.* events recorded.",
        "Files written before the recorder's ignore_agent fix contain only",
        "turn.*, message.ai and llm.* events, so every detector that needs",
        "tool calls yields nothing",
        "(ARCHITECTURE_trajectory_recorder.md, section 11).",
        "Record a new session to get mineable evidence.",
    ),
)

# The shared flags' violations keep the historical name.
MineArgsError = CliArgsError


class MineSelectionError(ValueError):
    """No trajectory matches the requested selection."""


@dataclass(frozen=True, slots=True)
class MineOptions:
    """Parsed /skill-mine flags."""

    threads: int = DEFAULT_THREADS
    since: timedelta | None = None
    dry_run: bool = False
    thread: str | None = None
    policy: str | None = None
    demo: bool | None = None


@dataclass(slots=True)
class ThreadStats:
    """One selected trajectory with its shape counters and mined episodes."""

    trajectory: Trajectory
    turns: int
    tool_calls: int
    ai_messages: int
    episodes: list[Episode] = field(default_factory=list)
    # Other trajectories the episodes cite (a shared procedure's second
    # session); the evidence bundle indexes them next to ``trajectory``.
    supporting: list[Trajectory] = field(default_factory=list)
    # Why observed_procedure candidates were dropped (demo only).
    notes: list[DetectorNote] = field(default_factory=list)

    @property
    def thread_id(self) -> str:
        """Thread the trajectory was recorded for."""
        return self.trajectory.thread_id

    @property
    def score(self) -> float:
        """Incident score of the episodes mined from this thread."""
        return evidence_score(self.episodes)

    def gate(self, min_score: float, *, demo: bool = False) -> GateDecision:
        """Whether this thread reaches the LLM stage, and why."""
        return gate_decision(self.episodes, min_score=min_score, demo=demo)


# ------------------------------------------------------------ argument parsing


def _parse_threads(raw: str) -> int:
    """Positive thread count of --threads."""
    try:
        value = int(raw)
    except ValueError:
        raise MineArgsError(
            f"--threads: expected a whole number, got '{raw}'",
        ) from None
    if value < 1:
        raise MineArgsError(f"--threads: must be at least 1, got '{raw}'")
    return value


def _parse_since(raw: str) -> timedelta:
    """Age window of --since: <count> followed by h, d or w."""
    match = _SINCE_RE.match(raw.strip().lower())
    if match is None:
        expected = "expected <count><unit> with unit h, d or w (24h, 7d, 2w)"
        raise MineArgsError(f"--since: {expected}, got '{raw}'")
    count = int(match.group(1))
    if count < 1:
        raise MineArgsError(f"--since: the count must be at least 1, got '{raw}'")
    return timedelta(seconds=count * _SINCE_SECONDS[match.group(2)])


def parse_mine_args(args: list[str]) -> MineOptions:
    """Parse the flags of /skill-mine; every violation raises MineArgsError.

    Hand-rolled on purpose: argparse reports errors with ``sys.exit``, and
    ``SystemExit`` is a ``BaseException`` that no guard in the CLI catches, so
    a mistyped flag would end the user's session.
    """
    threads: int | None = None
    since: timedelta | None = None
    thread: str | None = None
    shared = SharedFlags()

    index = 0
    while index < len(args):
        consumed = take_shared_flag(args, index, shared)
        if consumed is not None:
            index = consumed
            continue
        token = args[index]
        if token in ("--threads", "--since", "--thread"):
            if index + 1 >= len(args):
                raise MineArgsError(f"{token} requires a value")
            value = args[index + 1]
            if token == "--threads":
                if threads is not None:
                    raise MineArgsError("--threads given twice")
                threads = _parse_threads(value)
            elif token == "--since":
                if since is not None:
                    raise MineArgsError("--since given twice")
                since = _parse_since(value)
            else:
                if thread is not None:
                    raise MineArgsError("--thread given twice")
                thread = value.strip()
            index += 2
            continue
        raise MineArgsError(f"unknown argument '{token}'")

    if thread is not None and (threads is not None or since is not None):
        raise MineArgsError(
            " ".join(
                (
                    "--thread selects one thread and cannot be combined",
                    "with --threads or --since",
                ),
            ),
        )
    return MineOptions(
        threads=DEFAULT_THREADS if threads is None else threads,
        since=since,
        dry_run=shared.dry_run,
        thread=thread,
        policy=shared.policy,
        demo=shared.demo,
    )


# ---------------------------------------------------------------- formatting


def short_thread(thread_id: str) -> str:
    """Clip a thread id to a still-resolvable prefix."""
    if len(thread_id) <= THREAD_ID_WIDTH:
        return thread_id
    return f"{thread_id[:THREAD_ID_WIDTH]}…"


def format_since(since: timedelta | None) -> str:
    """Human form of the --since window."""
    if since is None:
        return ""
    hours = int(since.total_seconds() // 3600)
    if hours % 168 == 0:
        return f"{hours // 168}w"
    if hours % 24 == 0:
        return f"{hours // 24}d"
    return f"{hours}h"


def format_tool_sequence(names: list[str]) -> str:
    """Tool names of an episode, prefixed with their true count."""
    if not names:
        return "[0] (none)"
    shown = " → ".join(names[:MAX_TOOLS_SHOWN])
    extra = len(names) - MAX_TOOLS_SHOWN
    suffix = f" … +{extra}" if extra > 0 else ""
    return f"[{len(names)}] {shown}{suffix}"


def format_evidence_seq(seqs: list[int]) -> str:
    """Cited event seqs of an episode, prefixed with their true count."""
    shown = ", ".join(str(seq) for seq in seqs[:MAX_SEQ_SHOWN])
    extra = len(seqs) - MAX_SEQ_SHOWN
    suffix = f" … +{extra}" if extra > 0 else ""
    return f"[{len(seqs)}] {shown}{suffix}"


def count_shape(trajectory: Trajectory) -> tuple[int, int, int]:
    """Turns, tool calls and AI messages of one trajectory."""
    turns = len(trajectory.turns)
    tool_calls = sum(len(turn.tool_calls) for turn in trajectory.turns)
    ai_messages = sum(len(turn.ai_messages) for turn in trajectory.turns)
    return turns, tool_calls, ai_messages


def failed_turn_lines(stats: list[ThreadStats]) -> list[str]:
    """One line per thread that recorded neither tool calls nor model replies because its turns ended in an error.

    Such a file (an authentication failure at the first LLM call, for
    instance) is not a pre-``ignore_agent`` recording: the tool-less note
    would name the wrong cause.
    """
    lines: list[str] = []
    for item in stats:
        if item.tool_calls or item.ai_messages:
            continue
        errors = [turn.error_type or "error" for turn in item.trajectory.turns if turn.status == "error"]
        if not errors:
            continue
        kinds = ", ".join(dict.fromkeys(errors))
        lines.append(
            f"Thread {short_thread(item.thread_id)}: {len(errors)} turn(s) ended with {kinds} before any tool call "
            "or model reply; nothing to mine."
        )
    return lines


def build_threads_table(stats: list[ThreadStats], *, min_score: float, demo: bool = False) -> Table:
    """One row per selected thread: shape counters, score and the gate."""
    title = f"Threads (min_evidence_score {min_score:.2f}"
    title += ", demo override)" if demo else ")"
    table = Table(
        box=box.SIMPLE_HEAD,
        title=title,
        title_style="accent",
        title_justify="left",
        header_style="accent",
        border_style="border",
        expand=False,
        pad_edge=False,
    )
    table.add_column("thread", style="command", no_wrap=True)
    table.add_column("turns", justify="right", style="muted")
    table.add_column("tools", justify="right")
    table.add_column("ai", justify="right", style="muted")
    table.add_column("episodes", justify="right", style="secondary")
    table.add_column("incidents", justify="right", style="secondary")
    table.add_column("score", justify="right", style="primary")
    table.add_column("gate", no_wrap=True)
    table.add_column("reason", style="muted")
    for item in stats:
        # A zero tool count is the single most diagnostic number here: it
        # means no detector that needs tool calls could ever fire.
        tools_style = "warning" if item.tool_calls == 0 else "muted"
        decision = item.gate(min_score, demo=demo)
        gate = Text("pass", style="success") if decision.passes else Text("skip", style="muted")
        table.add_row(
            escape(short_thread(item.thread_id)),
            str(item.turns),
            Text(str(item.tool_calls), style=tools_style),
            str(item.ai_messages),
            str(len(item.episodes)),
            str(len(decision.incidents)),
            f"{decision.score:.2f}",
            gate,
            escape(decision.describe()),
        )
    return table


def build_episodes_table(stats: list[ThreadStats]) -> Table:
    """Every mined episode, grouped by thread, in detector order.

    Detector order, not weight order: the bundle sorts by weight for the
    model, while a human reading this table wants to know which detector
    fired. The ``incident`` column labels the episodes of one thread that
    describe the same events (``I1``, ``I2``, … in ``group_incidents``
    order), so it is visible which of them count once in the score.
    """
    table = Table(
        box=box.SIMPLE_HEAD,
        title="Episodes",
        title_style="accent",
        title_justify="left",
        header_style="accent",
        border_style="border",
        expand=False,
        pad_edge=False,
    )
    table.add_column("thread", style="command", no_wrap=True)
    table.add_column("kind", style="secondary", no_wrap=True)
    table.add_column("incident", style="muted", no_wrap=True)
    table.add_column("weight", justify="right", style="primary", no_wrap=True)
    table.add_column(
        "tool sequence",
        style="default",
        max_width=48,
        overflow="ellipsis",
    )
    table.add_column("evidence seq", style="muted", max_width=30, overflow="ellipsis")

    first_group = True
    for item in stats:
        if not item.episodes:
            continue
        if not first_group:
            table.add_section()
        first_group = False
        # Episodes hold lists and dicts, so they are not hashable: key by id.
        incident_labels = {
            id(episode): f"I{number}"
            for number, incident in enumerate(group_incidents(item.episodes), start=1)
            for episode in incident
        }
        for position, episode in enumerate(item.episodes):
            label = escape(short_thread(item.thread_id)) if position == 0 else ""
            table.add_row(
                label,
                escape(episode.kind),
                incident_labels[id(episode)],
                f"{episode.weight:.2f}",
                escape(format_tool_sequence(episode.tool_sequence)),
                escape(format_evidence_seq(episode.evidence_seq)),
            )
        table.add_row(
            "",
            Text("subtotal", style="muted.bold"),
            "",
            Text(f"{item.score:.2f}", style="muted.bold"),
            "",
            "",
        )
    return table


# ----------------------------------------------------------------- selection


def _where(trajectories_dir: Path, workspace: Path | None) -> str:
    """Where a selection looked, naming the workspace in the shared store."""
    if workspace is None:
        return f"in {trajectories_dir}"
    return f"in {trajectories_dir} for workspace {workspace}"


def select_trajectories(
    trajectories_dir: Path,
    *,
    agent: str,
    options: MineOptions,
    now: float,
    cross_session_limit: int = CROSS_SESSION_LIMIT,
    workspace: Path | None = None,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Pick the threads to mine plus the cross-session pool behind them.

    The pool is the agent's newest ``cross_session_limit`` trajectories, the
    same set the single-thread command builds, so repeated procedures keep
    their support. ``--since`` compares file mtime, not ``started_at``: mtime
    is always present and means last activity, while ``started_at`` is the
    first event's timestamp and may be empty. Because the listing is already
    mtime-descending, the filter is a contiguous prefix. Note that mtime does
    not survive copying files between machines. ``workspace`` narrows targets
    and pool to one workspace of the shared store (``workspace_filter``).
    """
    if not trajectories_dir.is_dir() or not any(trajectories_dir.glob("*.jsonl")):
        raise MineSelectionError(
            f"No trajectories recorded in {trajectories_dir}; run a session first",
        )

    listing = load_trajectories(
        trajectories_dir,
        agent=agent,
        workspace=workspace,
        limit=max(options.threads, cross_session_limit),
    )
    pool = listing[:cross_session_limit]

    if options.thread is not None:
        target = _resolve_thread(
            trajectories_dir,
            options.thread,
            pool,
            workspace=workspace,
        )
        return [target], pool

    if not listing:
        hint = "use --thread <id> for another agent's session"
        where = _where(trajectories_dir, workspace)
        raise MineSelectionError(
            f"No trajectories for agent '{agent}' {where}; {hint}",
        )

    candidates = listing
    if options.since is not None:
        oldest = now - options.since.total_seconds()
        candidates = []
        for trajectory in listing:
            if trajectory.path.stat().st_mtime >= oldest:
                candidates.append(trajectory)
        if not candidates:
            window = format_since(options.since)
            detail = f"was modified within {window}; widen --since or drop it"
            raise MineSelectionError(f"No thread of agent '{agent}' {detail}")
    return candidates[: options.threads], pool


def _resolve_thread(
    trajectories_dir: Path,
    thread: str,
    pool: list[Trajectory],
    *,
    workspace: Path | None = None,
) -> Trajectory:
    """Resolve --thread to one trajectory; the id may be a unique prefix."""
    path = find_trajectory_file(trajectories_dir, thread, workspace=workspace)
    if path is None:
        # find_trajectory_file returns None both for "no match" and for an
        # ambiguous prefix, so one message has to cover both.
        hint = "give a full id or a unique prefix"
        where = _where(trajectories_dir, workspace)
        raise MineSelectionError(
            f"No recorded trajectory for thread '{thread}' {where}; {hint}",
        )
    for trajectory in pool:
        if trajectory.path == path:
            return trajectory
    return load_trajectory(path)


def mine_stats(
    targets: list[Trajectory],
    pool: list[Trajectory],
    skills: list[Skill],
    *,
    demo: bool = False,
) -> list[ThreadStats]:
    """Run the code-only detectors over every selected thread."""
    docs = [SkillDoc(skill.display_name, skill.description) for skill in skills]
    index = BM25Index(docs)
    stats: list[ThreadStats] = []
    for target in targets:
        others = [item for item in pool if item.thread_id != target.thread_id]
        # The target goes in first: mine_cross_session attributes an n-gram to
        # the first thread it sees, so the kept episodes belong to this thread
        # and cite its events plus those of one supporting session, which the
        # bundle must index too (``supporting``).
        notes: list[DetectorNote] = []
        episodes = collect_episodes(target, others, skill_index=index, demo=demo, notes=notes)
        turns, tool_calls, ai_messages = count_shape(target)
        stats.append(
            ThreadStats(
                trajectory=target,
                turns=turns,
                tool_calls=tool_calls,
                ai_messages=ai_messages,
                episodes=episodes,
                supporting=supporting(others, episodes),
                notes=notes,
            ),
        )
    return stats


class SkillMiningHandler:
    """Mine recorded threads into SKILL.md proposals (up to max_plans per thread)."""

    def __init__(self, session) -> None:
        self.session = session
        # Only used for its prompt loader, which is an instance method. Its
        # __init__ just stores the session, and reaching _load_config /
        # _load_stage_prompt through the class keeps both commands seeing the
        # same (patchable) class attributes.
        self._generator = DirectSkillGenerationHandler(session)

    async def handle(self, args: list[str]) -> None:
        """Mine recent threads for SKILL.md proposals: --dry-run shows episodes."""
        try:
            options = parse_mine_args(args)
        except MineArgsError as exc:
            console.print_error(escape(str(exc)))
            console.print(f"[muted]{escape(USAGE)}[/muted]")
            console.print("")
            return

        try:
            await self._run(options)
        except (SkillEvolverConfigError, PromptContractError) as exc:
            report_config_error(console, exc)
            console.print("")
        except MineSelectionError as exc:
            console.print_warning(escape(str(exc)))
            console.print("")
        except TrajectoryReadError as exc:
            console.print_warning(escape(f"Trajectory unreadable ({exc}); the LLM was not called"))
            console.print("")
        except Exception as exc:
            console.print_error(escape(f"Error mining skills: {exc}"))
            console.print("")
            logger.exception("Skill mining failed")

    async def _run(self, options: MineOptions) -> None:
        """Select threads, extract evidence, then report or mine."""
        ctx = self.session.context
        cfg = apply_overrides(DirectSkillGenerationHandler._load_config(), policy=options.policy, demo=options.demo)
        rules = effective_rules(cfg, working_dir=Path(ctx.working_dir))
        state_dir = initializer.get_project_paths(Path(ctx.working_dir)).root
        trajectories_dir = resolve_trajectories_dir(state_dir=state_dir)
        workspace = workspace_filter(Path(ctx.working_dir))
        skills = await self._load_skills()
        now = time.time()

        with self._status("Extracting evidence..."):
            stats = await asyncio.to_thread(
                self._gather,
                trajectories_dir,
                ctx.agent,
                options,
                skills,
                now,
                rules,
                workspace,
            )

        self._report_selection(stats, options, agent=ctx.agent)
        print_policy_block(console, cfg, rules)
        console.console.print(
            build_threads_table(stats, min_score=rules.min_evidence_score, demo=rules.demo),
        )
        # The note explains a file with model replies but no tool events; a
        # file with neither (a turn that died at its first LLM call) gets its own line.
        if any(item.tool_calls == 0 and item.ai_messages > 0 for item in stats):
            console.print(f"[muted]{_NO_TOOL_EVENTS_NOTE}[/muted]")
        for line in failed_turn_lines(stats):
            console.print(f"[muted]{escape(line)}[/muted]")

        if options.dry_run:
            self._report_dry_run(stats, rules)
            return

        await self._mine(stats, cfg=cfg, rules=rules, skills=skills)

    @staticmethod
    def _gather(
        trajectories_dir: Path,
        agent: str,
        options: MineOptions,
        skills: list[Skill],
        now: float,
        rules: EffectiveRules,
        workspace: Path | None = None,
    ) -> list[ThreadStats]:
        """Blocking part of a run: read the JSONL files and detect episodes."""
        targets, pool = select_trajectories(
            trajectories_dir,
            agent=agent,
            options=options,
            now=now,
            cross_session_limit=rules.cross_session_limit,
            workspace=workspace,
        )
        return mine_stats(targets, pool, skills, demo=rules.demo)

    @staticmethod
    def _report_selection(
        stats: list[ThreadStats],
        options: MineOptions,
        *,
        agent: str,
    ) -> None:
        """Say what was selected before the tables are printed."""
        prefix = "Dry run: no LLM will be created. " if options.dry_run else ""
        if options.thread is not None:
            scope = f"Selected thread {options.thread}"
        else:
            window = format_since(options.since)
            scope = f"Selected the newest {len(stats)} of agent '{agent}'"
            if window:
                scope = f"{scope} modified within {window}"
        console.print_info(f"{prefix}{escape(scope)}")

    @staticmethod
    def _report_dry_run(stats: list[ThreadStats], rules: EffectiveRules) -> None:
        """Episode table, per-thread gate and bundle preview, then the totals; no LLM exists at this point."""
        min_score = rules.min_evidence_score
        episodes = sum(len(item.episodes) for item in stats)
        if episodes:
            console.console.print(build_episodes_table(stats))
        else:
            console.print_info(f"No episodes detected in {len(stats)} threads.")

        decisions = [item.gate(min_score, demo=rules.demo) for item in stats]
        for item, decision in zip(stats, decisions):
            console.print(f"[muted]thread {escape(short_thread(item.thread_id))}:[/muted]")
            thread = ThreadInput(item.trajectory, item.supporting, item.episodes, item.notes)
            for line in (
                *gate_lines(decision, min_score=min_score, episodes=item.episodes),
                *bundle_preview(thread, rules),
            ):
                console.print(f"[muted]  {escape(line)}[/muted]")
        incidents = sum(len(decision.incidents) for decision in decisions)
        total = sum(decision.score for decision in decisions)
        passing = sum(1 for decision in decisions if decision.passes)
        bound = llm_call_bound(
            passing,
            rules.max_plans,
            max_llm_calls=rules.max_llm_calls,
            expand=rules.on_nothing == "expand_context_once",
        )
        summary = " ".join(
            (
                f"Dry run: {len(stats)} threads, {episodes} episodes,",
                f"{incidents} incidents, total evidence score {total:.2f};",
                f"{passing} threads would reach the LLM (up to",
                f"{bound} LLM calls).",
                "Nothing was written and no LLM was created.",
            ),
        )
        console.print_info(summary)
        console.print("")

    async def _mine(
        self,
        stats: list[ThreadStats],
        *,
        cfg: DirectSkillGenerationConfig,
        rules: EffectiveRules,
        skills: list[Skill],
    ) -> None:
        """Run the LLM stages for every thread that passes the evidence gate; a refused one gets its gate-only report."""
        decisions = [item.gate(rules.min_evidence_score, demo=rules.demo) for item in stats]
        passing = [item for item, decision in zip(stats, decisions) if decision.passes]
        refused = [item for item, decision in zip(stats, decisions) if not decision.passes]
        below = len(refused)

        root = initializer.app_paths.home / SKILL_EVOLVER_CONFIG_FOLDER_NAME / "prompts"
        prompts = await load_prompts(self._generator._load_stage_prompt, root, cfg)
        work = Path(self.session.context.working_dir)
        state_dir = initializer.get_project_paths(work).root
        run = RunContext(
            command="skill-mine",
            working_dir=work,
            state_dir=state_dir,
            requested=cfg,
            rules=rules,
            prompts=prompts,
            skills=skills,
            llm_slot=LazyLlm(self.session),
            # Names a new skill must not reuse: the library plus every create
            # proposal written earlier in this run, across threads.
            taken={s.name for s in skills} | {s.display_name for s in skills},
            sink=console,
            report_dir=decisions_dir(state_dir) if rules.save_decision_report else None,
        )
        for item in refused:
            await record_gate_refusal(run, ThreadInput(item.trajectory, item.supporting, item.episodes, item.notes))
        if not passing:
            threshold = f"{rules.min_evidence_score:.2f}"
            detail = f"skipped by the gate (min_evidence_score {threshold})"
            console.print_info(f"Nothing to mine: {below} threads {detail}")
            console.print("")
            return

        bound = llm_call_bound(
            len(passing),
            rules.max_plans,
            max_llm_calls=rules.max_llm_calls,
            expand=rules.on_nothing == "expand_context_once",
        )
        console.print_info(f"Mining {len(passing)} threads (up to {bound} LLM calls)")
        total = PlanTally()
        nothing = 0
        # Threads whose every plan was refused (invalid twice, review failed,
        # secrets, budget, transport): they neither saved nor "had nothing".
        unwritten = 0
        failed: list[str] = []
        for position, item in enumerate(passing, start=1):
            tally = await self._mine_thread(item, run=run, position=position, total=len(passing))
            if tally is None:
                failed.append(item.thread_id)
                continue
            total.add(tally)
            if tally.plans == 0:
                nothing += 1
            elif tally.proposals == 0:
                unwritten += 1

        # Every thread lands in exactly one of: with a proposal, skipped by the
        # gate, nothing to save, rejected at generation, failed (proposals are counted, not threads).
        summary = (
            f"Mined {len(stats)} threads: {total.proposals} proposals,"
            f" {below} skipped by the gate, {nothing} nothing to save,"
            f" {unwritten} rejected at generation, {len(failed)} failed"
        )
        if failed:
            report = console.print_error
        elif total.generation_errors:
            report = console.print_warning
        elif total.proposals:
            report = console.print_success
        else:
            report = console.print_info
        report(summary)
        if failed:
            console.print_error(escape(f"Failed threads: {', '.join(failed)}"))
        if total.flagged:
            line = f"Plans: {total.describe()}"
            (console.print_warning if total.generation_errors else console.print_info)(line)
        console.print("")

    async def _mine_thread(
        self,
        stats: ThreadStats,
        *,
        run: RunContext,
        position: int,
        total: int,
    ) -> PlanTally | None:
        """The shared per-thread pipeline for one thread.

        Returns the plan tally, or ``None`` when the thread failed before or
        outside its plans. A failure is printed and logged with its thread
        id, and the loop continues: aborting would discard the remaining
        threads after earlier ones already wrote files.
        """
        thread_id = stats.thread_id
        decision = stats.gate(run.rules.min_evidence_score, demo=run.rules.demo)
        header = (
            f"[{position}/{total}] thread {short_thread(thread_id)} —"
            f" score {decision.score:.2f}, {len(stats.episodes)} episodes,"
            f" {len(decision.incidents)} incidents, {decision.describe()}"
        )
        console.print_info(escape(header))
        try:
            thread = ThreadInput(stats.trajectory, stats.supporting, stats.episodes, stats.notes)
            result = await run_thread(run, thread)
        except Exception as exc:
            console.print_error(escape(f"thread {thread_id}: {exc}"))
            logger.exception("Mining thread %s failed", thread_id)
            return None
        for line in draft_lines(result):
            console.print(f"[muted]{escape(line)}[/muted]")
        return result.tally

    async def _load_skills(self) -> list[Skill]:
        """The skill catalogue of the current agent (rescanned, project first)."""
        ctx = self.session.context
        return await initializer.refresh_cached_skills(
            agent=ctx.agent,
            working_dir=ctx.working_dir,
        )

    @staticmethod
    def _status(text: str):
        """Spinner shown while a pipeline stage runs."""
        return status(console, text)
