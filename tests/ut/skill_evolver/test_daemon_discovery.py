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

"""Project enumeration and the readiness rules of one trajectory directory."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from msagent.core.paths import AppPaths
from msagent.skill_evolver.daemon.config import SkillDaemonConfig
from msagent.skill_evolver.daemon.discovery import (
    PoolNotReady,
    ProjectRef,
    discover_projects,
    incompleteness,
    iter_scans,
    preflight,
    scan_project,
)
from msagent.trajectory_recorder.reader import load_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "trajectories"

NOW = 1_800_000_000.0
QUIET = 900


def _config(**overrides) -> SkillDaemonConfig:
    return SkillDaemonConfig.model_validate(overrides)


def _project(home: Path, working_dir: Path, project_id: str = "proj-abc123") -> ProjectRef:
    """A project as the CLI would have left it: metadata plus a trajectories directory."""
    state_dir = home / "state" / "projects" / project_id
    (state_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (state_dir / "project.json").write_text(
        json.dumps({"working_dir": str(working_dir)}),
        encoding="utf-8",
    )
    return ProjectRef(
        project_id=project_id,
        working_dir=working_dir,
        state_dir=state_dir,
        trajectories_dir=state_dir / "trajectories",
    )


def _place(project: ProjectRef, fixture: str, name: str, *, age: float = QUIET + 60) -> Path:
    """Copy a fixture into the project and age it so it counts as quiescent."""
    target = project.trajectories_dir / name
    shutil.copy(FIXTURES / fixture, target)
    stamp = NOW - age
    os.utime(target, (stamp, stamp))
    return target


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return AppPaths.resolve().home


# ------------------------------------------------------------------ projects


def test_discover_projects_reads_working_dir_from_metadata(home: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    _project(home, work)
    found = discover_projects(AppPaths.resolve(), _config())
    assert [p.working_dir for p in found] == [work.resolve()]
    assert found[0].trajectories_dir.name == "trajectories"


def test_listed_scope_filters_projects(home: Path, tmp_path: Path) -> None:
    wanted = tmp_path / "wanted"
    other = tmp_path / "other"
    wanted.mkdir()
    other.mkdir()
    _project(home, wanted, "wanted-1")
    _project(home, other, "other-1")

    config = _config(scope={"projects": "listed", "project_dirs": [str(wanted)]})
    assert [p.working_dir for p in discover_projects(AppPaths.resolve(), config)] == [wanted.resolve()]


def test_unreadable_metadata_skips_only_that_project(home: Path, tmp_path: Path) -> None:
    good = tmp_path / "good"
    good.mkdir()
    _project(home, good, "good-1")
    broken = home / "state" / "projects" / "broken-1"
    broken.mkdir(parents=True)
    (broken / "project.json").write_text("{not json", encoding="utf-8")

    found = discover_projects(AppPaths.resolve(), _config())
    assert [p.project_id for p in found] == ["good-1"]


def test_missing_projects_dir_is_not_an_error(home: Path) -> None:
    assert discover_projects(AppPaths.resolve(), _config()) == []


# -------------------------------------------------------------- readiness


def test_a_still_active_trajectory_is_not_a_candidate(home: Path, tmp_path: Path) -> None:
    """Nothing tells us a session ended, so a file written seconds ago is left alone."""
    project = _project(home, tmp_path / "work")
    _place(project, "skill_evolver_signals.jsonl", "Profiler_thread-live.jsonl", age=10)
    result = scan_project(project, _config(), now=NOW)
    assert result.candidates == []
    assert [reason for _, reason in result.skipped] == ["still active"]


def test_a_quiescent_complete_trajectory_is_a_candidate(home: Path, tmp_path: Path) -> None:
    project = _project(home, tmp_path / "work")
    _place(project, "skill_evolver_signals.jsonl", "Profiler_thread-done.jsonl")
    result = scan_project(project, _config(), now=NOW)
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.agent == "Profiler"
    assert candidate.mtime_ns > 0 and candidate.size_bytes > 0


def test_an_interrupted_turn_is_not_ready(home: Path, tmp_path: Path) -> None:
    """missing_turn_end.jsonl is a Ctrl+C session: the last turn never closed."""
    project = _project(home, tmp_path / "work")
    _place(project, "missing_turn_end.jsonl", "Profiler_thread-ctrlc.jsonl")
    result = scan_project(project, _config(), now=NOW)
    assert result.candidates == []
    assert result.skipped[0][1] == "last turn has no turn.end"


def test_broken_lines_do_not_block_a_readable_trajectory(home: Path, tmp_path: Path) -> None:
    """The reader skips damaged lines and keeps the line numbers, so refs stay valid."""
    project = _project(home, tmp_path / "work")
    path = _place(project, "malformed_lines.jsonl", "Profiler_thread-damaged.jsonl")
    assert load_trajectory(path).malformed_lines > 0
    result = scan_project(project, _config(), now=NOW)
    assert [c.thread_id for c in result.candidates] == ["thread-damaged"]


def test_a_thread_cut_off_by_the_recorder_limit_is_mined(home: Path, tmp_path: Path) -> None:
    """Nothing more will ever be appended, so waiting would only let it go stale."""
    project = _project(home, tmp_path / "work")
    _place(project, "recorder_limit.jsonl", "Quantizer_thread-limit.jsonl")
    result = scan_project(project, _config(), now=NOW)
    assert [c.thread_id for c in result.candidates] == ["thread-limit"]


def test_agent_scope_filters_candidates(home: Path, tmp_path: Path) -> None:
    project = _project(home, tmp_path / "work")
    _place(project, "skill_evolver_signals.jsonl", "Profiler_thread-a.jsonl")
    _place(project, "recorder_limit.jsonl", "Quantizer_thread-limit.jsonl")

    result = scan_project(project, _config(scope={"agents": ["Quantizer"]}), now=NOW)
    assert [c.agent for c in result.candidates] == ["Quantizer"]
    assert any("out of scope" in reason for _, reason in result.skipped)


def test_candidates_group_by_agent(home: Path, tmp_path: Path) -> None:
    """The skill catalogue and the cross-session pool are both agent-scoped."""
    project = _project(home, tmp_path / "work")
    _place(project, "skill_evolver_signals.jsonl", "Profiler_thread-a.jsonl")
    _place(project, "recorder_limit.jsonl", "Quantizer_thread-limit.jsonl")

    grouped = scan_project(project, _config(), now=NOW).by_agent()
    assert sorted(grouped) == ["Profiler", "Quantizer"]


# --------------------------------------------------------------- preflight


def test_an_empty_trajectory_file_defers_the_whole_project(home: Path, tmp_path: Path) -> None:
    """load_trajectories peeks the first event of every file, so one empty file kills the run."""
    project = _project(home, tmp_path / "work")
    _place(project, "skill_evolver_signals.jsonl", "Profiler_thread-done.jsonl")
    (project.trajectories_dir / "Profiler_thread-brandnew.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(PoolNotReady):
        preflight(project.trajectories_dir)
    with pytest.raises(PoolNotReady):
        scan_project(project, _config(), now=NOW)


def test_iter_scans_defers_instead_of_failing(home: Path, tmp_path: Path) -> None:
    """A transient empty file must not abort the other projects of the tick."""
    ready = _project(home, tmp_path / "ready", "ready-1")
    _place(ready, "skill_evolver_signals.jsonl", "Profiler_thread-done.jsonl")
    blocked = _project(home, tmp_path / "blocked", "blocked-1")
    (blocked.trajectories_dir / "Profiler_thread-brandnew.jsonl").write_text("", encoding="utf-8")

    scans = list(iter_scans([ready, blocked], _config(), now=NOW))
    assert [scan.project.project_id for scan in scans] == ["ready-1"]


def test_missing_trajectories_dir_yields_nothing(home: Path, tmp_path: Path) -> None:
    project = _project(home, tmp_path / "work")
    shutil.rmtree(project.trajectories_dir)
    assert scan_project(project, _config(), now=NOW).candidates == []


# ------------------------------------------------------------ completeness


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("skill_evolver_signals.jsonl", None),
        ("missing_turn_end.jsonl", "last turn has no turn.end"),
        ("recorder_limit.jsonl", None),
        ("orphan_tool_start.jsonl", "an orphan tool span is still open"),
    ],
)
def test_incompleteness_reasons(fixture: str, expected: str | None) -> None:
    assert incompleteness(load_trajectory(FIXTURES / fixture)) == expected
