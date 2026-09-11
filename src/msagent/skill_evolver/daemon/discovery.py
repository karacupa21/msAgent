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

"""Which projects exist and which of their trajectories are ready to be mined.

There is no ``session.end`` event: the recorder opens the file lazily, appends a line
and closes it again — no sentinel, no rename, no finalizer
(ARCHITECTURE_trajectory_recorder.md, section on session wiring). Adding one would edit
the SCHEMA_VERSION 1 write path, which the project rules forbid. Completeness is
therefore **inferred**:

    candidate = quiescent AND complete AND eligible

``quiescent`` is file mtime, the same signal ``/skill-mine --since`` already trusts;
``complete`` is the reader's own ``Turn.status`` and ``ToolCall.status`` markers.

In the shared store (``output.scope: shared`` in the recorder config) every project
resolves the same directory. :func:`iter_shared_scans` then reads it once per tick and
hands each file to the project whose working directory its ``recorder.attach`` names,
so a thread is mined once, in the context of the workspace it was recorded in.

Stdlib only (plus the recorder's langchain-free reader), so the whole selection stage
runs in tests without an LLM.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from msagent.core.logging import get_logger
from msagent.core.paths import AppPaths
from msagent.skill_evolver.daemon.config import SkillDaemonConfig
from msagent.trajectory_recorder.export import resolve_trajectories_dir
from msagent.trajectory_recorder.reader import (
    TrajectoryReadError,
    file_workspace,
    iter_events,
    load_trajectory,
    workspace_key,
)
from msagent.trajectory_recorder.model import Trajectory

logger = get_logger(__name__)

PROJECT_METADATA_FILE = "project.json"


class PoolNotReady(RuntimeError):
    """A trajectory directory holds a file that would abort the whole selection.

    ``load_trajectories`` filters on each file's first event through ``_peek_header``,
    which raises ``TrajectoryReadError`` for a file with no events at all — and it is
    called for **every** ``*.jsonl`` in the directory. A session that has just created
    its file but not yet written to it would therefore kill the run for the whole
    project. That is transient, so the project is deferred to the next tick rather
    than failed.
    """


@dataclass(frozen=True, slots=True)
class ProjectRef:
    """One project of the msAgent home."""

    project_id: str
    working_dir: Path
    state_dir: Path
    trajectories_dir: Path


@dataclass(frozen=True, slots=True)
class Candidate:
    """A trajectory file the daemon is willing to mine."""

    project: ProjectRef
    path: Path
    thread_id: str
    agent: str
    size_bytes: int
    mtime_ns: int


@dataclass(slots=True)
class ScanResult:
    """What one project offered this tick."""

    project: ProjectRef
    candidates: list[Candidate] = field(default_factory=list)
    # Files that exist but are not ready yet, with the reason; logged, never mined.
    skipped: list[tuple[Path, str]] = field(default_factory=list)

    def by_agent(self) -> dict[str, list[Candidate]]:
        """Candidates grouped by agent: the skill catalogue and the cross-session pool are agent-scoped."""
        grouped: dict[str, list[Candidate]] = {}
        for candidate in self.candidates:
            grouped.setdefault(candidate.agent, []).append(candidate)
        return grouped


# ------------------------------------------------------------------- projects


def discover_projects(app_paths: AppPaths, config: SkillDaemonConfig) -> list[ProjectRef]:
    """Every project of the home that the configured scope admits.

    A project is a ``project.json`` under ``<home>/state/projects/``; its
    ``working_dir`` is what the mining pipeline needs to resolve skills and output.
    """
    projects_dir = app_paths.projects_dir
    if not projects_dir.is_dir():
        return []
    allowed = _allowed_dirs(config)
    found: list[ProjectRef] = []
    for metadata in sorted(projects_dir.glob(f"*/{PROJECT_METADATA_FILE}")):
        working_dir = _working_dir_of(metadata)
        if working_dir is None:
            continue
        if allowed is not None and working_dir not in allowed:
            continue
        state_dir = metadata.parent
        found.append(
            ProjectRef(
                project_id=state_dir.name,
                working_dir=working_dir,
                state_dir=state_dir,
                trajectories_dir=resolve_trajectories_dir(state_dir=state_dir),
            )
        )
    return found


def _allowed_dirs(config: SkillDaemonConfig) -> set[Path] | None:
    """The resolved allow-list, or ``None`` when every project is in scope."""
    if config.scope.projects == "all":
        return None
    return {Path(entry).expanduser().resolve() for entry in config.scope.project_dirs}


def _working_dir_of(metadata: Path) -> Path | None:
    """The project's working directory, or ``None`` when the metadata is unusable."""
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("skill-daemon: cannot read %s; skipping the project", metadata, exc_info=True)
        return None
    raw = payload.get("working_dir") if isinstance(payload, dict) else None
    if not isinstance(raw, str) or not raw:
        logger.warning("skill-daemon: %s has no working_dir; skipping the project", metadata)
        return None
    return Path(raw).expanduser().resolve()


# ---------------------------------------------------------------- trajectories


def preflight(trajectories_dir: Path) -> None:
    """Raise ``PoolNotReady`` when a file would abort ``load_trajectories``.

    Cheap: it stops at the first event of each file. Broken *lines* inside a non-empty
    file are fine — the reader skips them and keeps the physical line numbers, so
    evidence refs stay valid while the file is still being appended to.
    """
    for path in sorted(trajectories_dir.glob("*.jsonl")):
        if _first_event(path) is None:
            raise PoolNotReady(f"{path} holds no events yet")


def _first_event(path: Path) -> dict | None:
    """The first readable event of a file, or ``None`` when it has none."""
    try:
        for event in iter_events(path):
            return event
    except OSError:
        logger.warning("skill-daemon: cannot read %s", path, exc_info=True)
        return None
    return None


def scan_project(project: ProjectRef, config: SkillDaemonConfig, *, now: float) -> ScanResult:
    """Select the finished, in-scope trajectories of one project.

    Raises ``PoolNotReady`` when the directory is not safe to enumerate yet.
    """
    result = ScanResult(project=project)
    directory = project.trajectories_dir
    if not directory.is_dir():
        return result
    preflight(directory)
    for path in sorted(directory.glob("*.jsonl")):
        _examine(path, result, config, now=now)
    return result


def _examine(
    path: Path,
    result: ScanResult,
    config: SkillDaemonConfig,
    *,
    now: float,
) -> None:
    """Add a file of ``result.project`` as a candidate, or as a skip with a reason."""
    stat = path.stat()
    if now - stat.st_mtime < config.schedule.quiet_period_seconds:
        result.skipped.append((path, "still active"))
        return
    try:
        trajectory = load_trajectory(path)
    except TrajectoryReadError as exc:
        # Quiescent and still unreadable: a real defect, not a race. The caller
        # records it against max_attempts so it stops being retried forever.
        result.skipped.append((path, f"unreadable: {exc}"))
        logger.error("skill-daemon: %s is quiescent but unreadable: %s", path, exc)
        return
    agents = set(config.scope.agents)
    if agents and trajectory.agent not in agents:
        result.skipped.append((path, f"agent {trajectory.agent} out of scope"))
        return
    incomplete = incompleteness(trajectory)
    if incomplete is not None:
        result.skipped.append((path, incomplete))
        return
    result.candidates.append(
        Candidate(
            project=result.project,
            path=path,
            thread_id=trajectory.thread_id,
            agent=trajectory.agent,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
    )


def iter_shared_scans(
    store_dir: Path,
    projects: list[ProjectRef],
    config: SkillDaemonConfig,
    *,
    now: float,
) -> Iterator[ScanResult]:
    """Scan the shared store once and hand each file to the project it was recorded in.

    Every project resolves ``store_dir``, so per-project scans would parse each file
    once per project and mine each thread once per project. A file belongs to the
    project whose working directory its first event (``recorder.attach``) names; a
    file whose origin is unknown, relative or not among ``projects`` (out of scope,
    or not selected by ``--project``) is logged and left alone. Every project gets a
    ``ScanResult``, empty or not, because the runner stamps its inbox from it; a store
    that is not safe to enumerate yet defers the whole tick.
    """
    results = [ScanResult(project=project) for project in projects]
    owners = {workspace_key(result.project.working_dir): result for result in results}
    if store_dir.is_dir():
        try:
            preflight(store_dir)
        except PoolNotReady as exc:
            logger.warning(
                "skill-daemon: deferring every project to the next tick: %s",
                exc,
            )
            return
        for path in sorted(store_dir.glob("*.jsonl")):
            origin = file_workspace(path)
            owner = owners.get(origin) if origin is not None else None
            if owner is None:
                reason = "unknown origin workspace"
                if origin is not None:
                    reason = f"origin {origin} is not in this tick"
                logger.info("skill-daemon: skipping %s (%s)", path, reason)
                continue
            _examine(path, owner, config, now=now)
    yield from results


def incompleteness(trajectory: Trajectory) -> str | None:
    """Why this trajectory is not finished, or ``None`` when it is.

    A thread cut short by ``recorder.limit`` counts as finished: nothing more will
    ever be appended to it, so waiting would only make it go stale.
    """
    if not trajectory.turns:
        return "no turns recorded"
    if trajectory.truncated_by_limit:
        return None
    last = trajectory.turns[-1]
    if last.status == "truncated":
        return "last turn has no turn.end"
    if any(call.status == "orphan" for turn in trajectory.turns for call in turn.tool_calls):
        return "an orphan tool span is still open"
    return None


def iter_scans(
    projects: list[ProjectRef],
    config: SkillDaemonConfig,
    *,
    now: float,
) -> Iterator[ScanResult]:
    """Scan every project, deferring the ones whose directory is not ready."""
    for project in projects:
        try:
            yield scan_project(project, config, now=now)
        except PoolNotReady as exc:
            logger.warning("skill-daemon: deferring project %s to the next tick: %s", project.project_id, exc)
