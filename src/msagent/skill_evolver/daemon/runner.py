#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""One tick of the background miner.

A tick is a pure batch: it takes the state of the disk, the config and the clock, and
runs the **existing** per-thread pipeline over the trajectories nobody has mined yet.
It is a third thin caller of ``pipeline.run_thread`` beside ``/skill-mine`` and
``/direct-skill-generation`` — the stage sequence itself is not duplicated here.

Two guards keep an unattended run honest:

* the LLM budget is spent against ``llm_call_bound``, the **worst case** of a thread,
  so the tick stops before it could exceed ``max_llm_calls_per_tick`` rather than
  after;
* every outcome lands in the ledger, so the next tick pays for nothing twice.

Proposals stay inactive: the writer puts them under ``.proposals/`` exactly as the
interactive command does, and promotion is still a human decision.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# The handlers package must be initialized before the evolver handlers are imported
# directly: msagent.cli.handlers imports DirectSkillGenerationHandler, which imports
# msagent.cli.handlers.session_history. The interactive CLI enters through the handlers
# package and never trips this; a standalone entry point does. The tests do the same.
import msagent.cli.handlers  # noqa: F401
from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.core.context import Context
from msagent.cli.theme import console
from msagent.core.constants import SKILL_EVOLVER_CONFIG_FOLDER_NAME
from msagent.core.logging import get_logger
from msagent.core.paths import AppPaths
from msagent.core.storage_layout import validate_and_initialize_storage_layout
from msagent.skill_evolver.budget import llm_call_bound
from msagent.skill_evolver.config import EffectiveRules, apply_overrides, effective_rules
from msagent.skill_evolver.daemon import inbox
from msagent.skill_evolver.daemon.config import (
    SkillDaemonConfig,
    daemon_state_dir,
    load_daemon_config,
)
from msagent.skill_evolver.daemon.discovery import (
    Candidate,
    ProjectRef,
    ScanResult,
    discover_projects,
    iter_scans,
    iter_shared_scans,
)
from msagent.skill_evolver.daemon.ledger import (
    STATUS_FAILED,
    STATUS_GATE_SKIP,
    STATUS_MINED,
    STATUS_NOTHING,
    STATUS_REJECTED,
    Ledger,
    content_sha256,
    methodology_fingerprint,
)
from msagent.skill_evolver.daemon.lock import tick_lock
from msagent.skill_evolver.direct_skill_generation import DirectSkillGenerationHandler
from msagent.skill_evolver.features import FEATURES_VERSION
from msagent.skill_evolver.mining import SkillMiningHandler, mine_stats
from msagent.skill_evolver.pipeline import LazyLlm, RunContext, ThreadInput, load_prompts, record_gate_refusal
from msagent.skill_evolver.report import decisions_dir
from msagent.skill_evolver.writer import batch_dir_name
from msagent.trajectory_recorder.config import load_trajectory_config
from msagent.trajectory_recorder.export import (
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

# The command label stamped into every decision report this module produces, so a
# background verdict is distinguishable from one the user asked for. Nothing branches
# on RunContext.command; it is a report field only.
COMMAND = "skill-daemon"

STATUS_OK = "ok"
STATUS_DISABLED = "disabled"
STATUS_BUSY = "busy"
STATUS_TOO_SOON = "too_soon"


@dataclass(slots=True)
class TickResult:
    """What one tick did; the CLI turns it into an exit code and a summary line."""

    status: str = STATUS_OK
    detail: str = ""
    projects: int = 0
    threads: int = 0
    proposals: int = 0
    gate_skips: int = 0
    nothing: int = 0
    rejected: int = 0
    failures: int = 0
    llm_calls_reserved: int = 0
    stopped_by_budget: bool = False
    proposal_paths: list[Path] = field(default_factory=list)

    @property
    def ran(self) -> bool:
        return self.status == STATUS_OK

    def summary(self) -> str:
        if self.status == STATUS_DISABLED:
            return "skill-daemon is disabled; nothing was scanned"
        if self.status == STATUS_BUSY:
            return "another tick is already running"
        if self.status == STATUS_TOO_SOON:
            return f"too soon after the previous tick ({self.detail})"
        line = (
            f"Tick over {self.projects} projects: {self.threads} threads, {self.proposals} proposals,"
            f" {self.gate_skips} skipped by the gate, {self.nothing} nothing to save,"
            f" {self.rejected} rejected at generation, {self.failures} failed"
        )
        if self.stopped_by_budget:
            line = f"{line}; stopped by the LLM budget"
        return line


@dataclass(slots=True)
class _Budget:
    """The LLM allowance of one tick, spent against the worst case of each thread."""

    remaining: int
    reserved: int = 0
    exhausted: bool = False

    def afford(self, ceiling: int) -> bool:
        if self.remaining < ceiling:
            self.exhausted = True
            return False
        self.remaining -= ceiling
        self.reserved += ceiling
        return True


class _BudgetStop(Exception):
    """Raised inside the project loop to end the whole tick at the budget."""


# ------------------------------------------------------------------- helpers


def _fingerprint(rules: EffectiveRules, prompts: Any) -> str:
    """Everything that decides *how* a thread would be mined, as one hash."""
    return methodology_fingerprint(
        {
            "features_version": FEATURES_VERSION,
            "classify": prompts.classify.sha256,
            "generate": prompts.generate.sha256,
            "review": prompts.review.sha256,
            "rules": sorted(rules.as_record().items()),
        }
    )


def _status_of(tally: Any) -> str:
    """Map the plan tally onto the ledger vocabulary, using the /skill-mine categories."""
    if tally is None:
        return STATUS_FAILED
    if tally.plans == 0:
        return STATUS_NOTHING
    if tally.proposals == 0:
        return STATUS_REJECTED
    return STATUS_MINED


def _latest_report(report_dir: Path | None, thread_id: str, *, since: float) -> dict[str, Any] | None:
    """The decision report this run just wrote for ``thread_id``, if it wrote one.

    The report is the pipeline's own contract (REPORT_VERSION 2); reading it back is
    how the daemon learns the real LLM cost and the proposal paths without changing a
    line of the pipeline.
    """
    if report_dir is None or not report_dir.is_dir():
        return None
    stem = batch_dir_name(thread_id)
    fresh = [path for path in report_dir.glob(f"{stem}-*.json") if path.stat().st_mtime >= since]
    if not fresh:
        return None
    newest = max(fresh, key=lambda path: path.stat().st_mtime)
    try:
        payload = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("skill-daemon: cannot read the decision report %s", newest, exc_info=True)
        return None
    return payload if isinstance(payload, dict) else None


def _pool_and_targets(
    trajectories_dir: Path,
    agent: str,
    candidates: list[Candidate],
    *,
    cross_session_limit: int,
    workspace: Path | None = None,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """The agent's cross-session pool plus the selected targets, each parsed once.

    Same construction as ``mining.select_trajectories``: repeated procedures keep
    their support because the pool is the agent's newest ``cross_session_limit``
    trajectories, and a target already in the pool is reused rather than re-read.
    In the shared store ``workspace`` keeps the pool to the targets' own workspace,
    so a background verdict matches /skill-mine run in that workspace.
    """
    pool = load_trajectories(
        trajectories_dir,
        agent=agent,
        workspace=workspace,
        limit=cross_session_limit,
    )
    by_path = {trajectory.path: trajectory for trajectory in pool}
    targets = [by_path.get(candidate.path) or load_trajectory(candidate.path) for candidate in candidates]
    return pool, targets


# ---------------------------------------------------------------------- tick


async def run_once(
    *,
    now: float | None = None,
    dry_run: bool = False,
    force: bool = False,
    project_filter: str | None = None,
) -> TickResult:
    """Run one batch. Safe to call on a timer: it locks, budgets and records itself."""
    moment = time.time() if now is None else now
    # Seed first: on a fresh home the config the user is meant to edit does not exist
    # yet, and reading it before seeding would silently fall back to the packaged file.
    app_paths = AppPaths.resolve()
    validate_and_initialize_storage_layout(app_paths)

    config = load_daemon_config(force_reload=True)
    if not config.is_active:
        return TickResult(status=STATUS_DISABLED)
    # Where the trajectories live is the recorder's call: a --watch process must see a
    # scope change without a restart, like the daemon config above.
    load_trajectory_config(force_reload=True)

    state = daemon_state_dir()

    with tick_lock(state) as acquired:
        if not acquired:
            return TickResult(status=STATUS_BUSY)
        with Ledger.open(state) as ledger:
            waited = moment - ledger.last_tick()
            if not force and waited < config.schedule.min_interval_seconds:
                detail = f"{int(waited)}s of {config.schedule.min_interval_seconds}s"
                return TickResult(status=STATUS_TOO_SOON, detail=detail)
            result = await _run_projects(
                app_paths,
                config,
                ledger,
                now=moment,
                dry_run=dry_run,
                project_filter=project_filter,
            )
            if not dry_run:
                ledger.set_last_tick(moment)
    return result


async def _run_projects(
    app_paths: AppPaths,
    config: SkillDaemonConfig,
    ledger: Ledger,
    *,
    now: float,
    dry_run: bool,
    project_filter: str | None,
) -> TickResult:
    """Scan every project in scope and mine what the ledger has not seen."""
    projects = discover_projects(app_paths, config)
    if project_filter is not None:
        wanted = Path(project_filter).expanduser().resolve()
        projects = [item for item in projects if item.working_dir == wanted or item.project_id == project_filter]
    result = TickResult()
    budget = _Budget(remaining=config.schedule.max_llm_calls_per_tick)
    remaining_threads = config.schedule.max_threads_per_tick

    if load_trajectory_config().is_shared:
        # One directory serves every project: scan it once, give each file its origin.
        scans = iter_shared_scans(resolve_trajectories_dir(), projects, config, now=now)
    else:
        scans = iter_scans(projects, config, now=now)
    for scan in scans:
        result.projects += 1
        for path, reason in scan.skipped:
            logger.info("skill-daemon: skipping %s (%s)", path, reason)
        if remaining_threads <= 0:
            continue
        try:
            used = await _run_project(
                scan,
                config,
                ledger,
                result=result,
                budget=budget,
                max_threads=remaining_threads,
                dry_run=dry_run,
            )
        except _BudgetStop:
            result.stopped_by_budget = True
            break
        finally:
            # Stamp even an empty scan: the banner uses last_tick to tell
            # "the daemon has never run here" from "it ran and found nothing".
            if config.notify.inbox and not dry_run:
                inbox.stamp_tick(scan.project.state_dir)
        remaining_threads -= used

    result.llm_calls_reserved = budget.reserved
    result.stopped_by_budget = result.stopped_by_budget or budget.exhausted
    return result


async def _run_project(
    scan: ScanResult,
    config: SkillDaemonConfig,
    ledger: Ledger,
    *,
    result: TickResult,
    budget: _Budget,
    max_threads: int,
    dry_run: bool,
) -> int:
    """Mine one project, agent by agent. Returns how many threads it consumed."""
    used = 0
    for agent, candidates in scan.by_agent().items():
        if used >= max_threads:
            break
        try:
            used += await _run_agent(
                scan.project,
                agent,
                candidates,
                config,
                ledger,
                result=result,
                budget=budget,
                max_threads=max_threads - used,
                dry_run=dry_run,
            )
        except TrajectoryReadError as exc:
            # The cross-session pool is the agent's newest trajectories, and one corrupt
            # file among them makes load_trajectories abort. /skill-mine ends the command
            # there, which is right for a user watching it; a timer must not lose every
            # other project to it forever. Loud, per agent, every tick until it is fixed.
            result.failures += 1
            logger.error(
                "skill-daemon: unreadable trajectory in the pool of agent %s (project %s): %s",
                agent,
                scan.project.project_id,
                exc,
            )
    return used


async def _run_agent(
    project: ProjectRef,
    agent: str,
    candidates: list[Candidate],
    config: SkillDaemonConfig,
    ledger: Ledger,
    *,
    result: TickResult,
    budget: _Budget,
    max_threads: int,
    dry_run: bool,
) -> int:
    """Mine the candidates of one (project, agent) pair under one RunContext."""
    context = await _context_for(agent, project, config)
    if context is None:
        return 0
    session = SimpleNamespace(context=context, graph=None)
    handler = SkillMiningHandler(session)

    evolver_config = apply_overrides(
        DirectSkillGenerationHandler._load_config(),
        policy=config.mining.policy,
        demo=config.mining.demo,
    )
    rules = effective_rules(evolver_config, working_dir=project.working_dir)
    skills = await handler._load_skills()
    prompts_root = initializer.app_paths.home / SKILL_EVOLVER_CONFIG_FOLDER_NAME / "prompts"
    prompts = await load_prompts(handler._generator._load_stage_prompt, prompts_root, evolver_config)
    fingerprint = _fingerprint(rules, prompts)

    wanted = _due(candidates, ledger, project=project, config=config, fingerprint=fingerprint)
    if not wanted:
        return 0
    wanted = wanted[:max_threads]

    ceiling = llm_call_bound(
        1,
        rules.max_plans,
        max_llm_calls=rules.max_llm_calls,
        expand=rules.on_nothing == "expand_context_once",
    )
    report_dir = decisions_dir(project.state_dir) if rules.save_decision_report and not dry_run else None
    run = RunContext(
        command=COMMAND,
        working_dir=project.working_dir,
        state_dir=project.state_dir,
        requested=evolver_config,
        rules=rules,
        prompts=prompts,
        skills=skills,
        llm_slot=LazyLlm(session),
        taken={skill.name for skill in skills} | {skill.display_name for skill in skills},
        sink=console,
        report_dir=report_dir,
    )

    pool, targets = await asyncio.to_thread(
        _pool_and_targets,
        project.trajectories_dir,
        agent,
        [item for item, _ in wanted],
        cross_session_limit=rules.cross_session_limit,
        workspace=workspace_filter(project.working_dir),
    )
    stats = await asyncio.to_thread(mine_stats, targets, pool, skills, demo=rules.demo)

    consumed = 0
    for position, (candidate, sha) in enumerate(wanted, start=1):
        item = stats[position - 1]
        consumed += 1
        await _run_thread(
            item,
            candidate,
            sha,
            handler=handler,
            run=run,
            ledger=ledger,
            config=config,
            fingerprint=fingerprint,
            result=result,
            budget=budget,
            ceiling=ceiling,
            position=position,
            total=len(wanted),
            dry_run=dry_run,
        )
    return consumed


async def _context_for(agent: str, project: ProjectRef, config: SkillDaemonConfig) -> Any | None:
    """Resolve agent and model the way the CLI does, or ``None`` when the agent is gone.

    Trajectories outlive the agents that recorded them: one renamed or removed agent
    must cost its own threads, not the whole tick. Nothing is written to the ledger —
    the trajectory is fine, the configuration moved, and restoring the agent should
    make the threads mineable again rather than leaving them permanently skipped.
    """
    try:
        return await Context.create(
            agent=agent,
            model=config.mining.model,
            approval_mode=None,
            working_dir=project.working_dir,
            stream_output=False,
        )
    except ValueError as exc:
        logger.warning(
            "skill-daemon: skipping agent %s of project %s; its threads stay unmined (%s)",
            agent,
            project.project_id,
            exc,
        )
        return None


def _due(
    candidates: list[Candidate],
    ledger: Ledger,
    *,
    project: ProjectRef,
    config: SkillDaemonConfig,
    fingerprint: str,
) -> list[tuple[Candidate, str]]:
    """Candidates the ledger says are worth an LLM budget, with their content hash."""
    due: list[tuple[Candidate, str]] = []
    for candidate in candidates:
        sha = content_sha256(candidate.path)
        entry = ledger.entry(project.project_id, candidate.thread_id)
        verdict = ledger.verdict(
            entry,
            content_sha=sha,
            fingerprint=fingerprint,
            remine_on=config.mining.remine_on,
        )
        if verdict.mine:
            due.append((candidate, sha))
        else:
            logger.info("skill-daemon: %s not due (%s)", candidate.thread_id, verdict.reason)
    return due


async def _run_thread(
    item: Any,
    candidate: Candidate,
    sha: str,
    *,
    handler: SkillMiningHandler,
    run: RunContext,
    ledger: Ledger,
    config: SkillDaemonConfig,
    fingerprint: str,
    result: TickResult,
    budget: _Budget,
    ceiling: int,
    position: int,
    total: int,
    dry_run: bool,
) -> None:
    """One thread: the gate, the pipeline, the ledger row and the inbox entry."""
    project = candidate.project
    record = _recorder(ledger, candidate, sha, fingerprint=fingerprint, config=config)

    decision = item.gate(run.rules.min_evidence_score, demo=run.rules.demo)
    if not decision.passes:
        result.threads += 1
        result.gate_skips += 1
        if not dry_run:
            await record_gate_refusal(run, ThreadInput(item.trajectory, item.supporting, item.episodes, item.notes))
            record(STATUS_GATE_SKIP, detail=decision.describe())
        return

    if dry_run:
        result.threads += 1
        logger.info("skill-daemon: would mine %s (score %.2f)", candidate.thread_id, decision.score)
        return

    if not budget.afford(ceiling):
        logger.warning(
            "skill-daemon: stopping the tick before %s; %d calls left, a thread may need %d",
            candidate.thread_id,
            budget.remaining,
            ceiling,
        )
        raise _BudgetStop

    result.threads += 1
    started = time.time()
    tally = await handler._mine_thread(item, run=run, position=position, total=total)
    status = _status_of(tally)
    report = _latest_report(run.report_dir, candidate.thread_id, since=started)
    proposals = [Path(entry) for entry in (report or {}).get("proposals", []) if isinstance(entry, str)]
    llm_calls = int(((report or {}).get("llm") or {}).get("calls_used") or 0)

    if status == STATUS_MINED:
        result.proposals += tally.proposals
        result.proposal_paths.extend(proposals)
    elif status == STATUS_NOTHING:
        result.nothing += 1
    elif status == STATUS_REJECTED:
        result.rejected += 1
    else:
        result.failures += 1

    record(
        status,
        proposals=tally.proposals if tally is not None else 0,
        llm_calls=llm_calls,
        detail=str((report or {}).get("stop_message") or (report or {}).get("failed") or ""),
    )
    if proposals and config.notify.inbox:
        inbox.record_proposals(project.state_dir, thread_id=candidate.thread_id, proposals=proposals)


def _recorder(ledger: Ledger, candidate: Candidate, sha: str, *, fingerprint: str, config: SkillDaemonConfig):
    """A closure that writes this candidate's ledger row with the fixed fields filled in."""

    def record(status: str, *, proposals: int = 0, llm_calls: int = 0, detail: str = "") -> None:
        ledger.record(
            project_id=candidate.project.project_id,
            thread_id=candidate.thread_id,
            source_path=candidate.path,
            size_bytes=candidate.size_bytes,
            mtime_ns=candidate.mtime_ns,
            content_sha=sha,
            fingerprint=fingerprint,
            status=status,
            proposals=proposals,
            llm_calls=llm_calls,
            detail=detail or None,
            max_attempts=config.mining.max_attempts,
        )

    return record
