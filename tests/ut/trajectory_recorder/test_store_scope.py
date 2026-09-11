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

"""Trajectory store scope: a store per workspace, or one shared by all of them.

Drives the real hooks (instrument_config / finish_turn) with a duck-typed
context, the way the CLI dispatcher does, and reads the files back.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from msagent.core.paths import AppPaths
from msagent.trajectory_recorder import export, hooks
from msagent.trajectory_recorder.config import (
    ENV_CONFIG_PATH,
    StoreScope,
    TrajectoryRecorderConfig,
    reset_config_cache,
    store_dir,
)
from msagent.trajectory_recorder.reader import load_trajectories, load_trajectory

AGENT = "Tester"


def _config(**output: str) -> TrajectoryRecorderConfig:
    return TrajectoryRecorderConfig.model_validate({"output": output})


def _context(
    working_dir: Path,
    thread_id: str,
    *,
    state_dir: Path | None = None,
) -> SimpleNamespace:
    """What the hooks read from the CLI Context."""
    return SimpleNamespace(
        agent=AGENT,
        thread_id=thread_id,
        working_dir=working_dir,
        state_dir=state_dir,
        model="default",
        model_display="fake-model (test)",
    )


def _record_turn(context: SimpleNamespace) -> None:
    """One dispatched turn, as MessageDispatcher.dispatch() records it."""
    run_id = f"run-{context.thread_id}"
    hooks.instrument_config({}, context=context, run_id=run_id, user_message="hi")
    hooks.finish_turn(context=context, run_id=run_id)


def _shared_store() -> Path:
    return AppPaths.resolve().state_dir / "trajectories"


def _two_workspaces(tmp_path: Path) -> tuple[Path, Path]:
    """Record thread-a in ``first`` and thread-b in ``second``."""
    first, second = tmp_path / "first", tmp_path / "second"
    for work, thread_id in ((first, "thread-a"), (second, "thread-b")):
        work.mkdir()
        _record_turn(_context(work, thread_id))
    return first, second


@pytest.fixture
def scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Select the store scope through a config file, as a user would."""

    def select(value: str) -> None:
        path = tmp_path / f"config.trajectory.recorder.{value}.yml"
        path.write_text(f"output:\n  scope: {value}\n", encoding="utf-8")
        monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
        reset_config_cache()

    hooks.reset()
    yield select
    hooks.reset()
    reset_config_cache()


# ------------------------------------------------------------------ config


def test_the_workspace_scope_is_the_default() -> None:
    assert TrajectoryRecorderConfig().output.scope is StoreScope.WORKSPACE
    assert not TrajectoryRecorderConfig().is_shared
    assert _config(scope="shared").is_shared


def test_the_packaged_config_keeps_the_workspace_scope() -> None:
    """Existing installs keep recording per workspace."""
    default_dir = Path(str(files("resources") / "configs" / "default"))
    text = (default_dir / "config.trajectory.recorder.yml").read_text(encoding="utf-8")
    config = TrajectoryRecorderConfig.model_validate(yaml.safe_load(text))
    assert config.output.scope is StoreScope.WORKSPACE


def test_store_dir_per_scope(tmp_path: Path) -> None:
    state_dir = tmp_path / "project-state"
    absolute = tmp_path / "elsewhere"

    assert store_dir(_config(), state_dir=state_dir) == state_dir / "trajectories"
    assert store_dir(_config(scope="shared"), state_dir=state_dir) == _shared_store()
    assert store_dir(_config(scope="shared"), state_dir=None) == _shared_store()
    for scope in ("workspace", "shared"):
        config = _config(scope=scope, directory=str(absolute))
        assert store_dir(config, state_dir=None) == absolute
    with pytest.raises(ValueError, match="state dir"):
        store_dir(_config(), state_dir=None)


# ------------------------------------------------------------------ writer


def test_the_workspace_scope_records_into_the_project_state_dir(
    scope,
    tmp_path: Path,
) -> None:
    scope("workspace")
    work = tmp_path / "work"
    work.mkdir()
    state_dir = AppPaths.resolve().for_project(work).root

    _record_turn(_context(work, "thread-a", state_dir=state_dir))

    trajectory = load_trajectory(state_dir / "trajectories" / f"{AGENT}_thread-a.jsonl")
    assert [turn.status for turn in trajectory.turns] == ["completed"]
    assert trajectory.working_dir == str(work.resolve())
    assert not _shared_store().exists()


def test_the_shared_scope_records_every_workspace_into_one_store(
    scope,
    tmp_path: Path,
) -> None:
    scope("shared")
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_state = AppPaths.resolve().for_project(first).root

    # The CLI always sets state_dir; the shared store must not follow it.
    _record_turn(_context(first, "thread-a", state_dir=first_state))
    _record_turn(_context(second, "thread-b"))

    store = _shared_store()
    names = sorted(path.name for path in store.glob("*.jsonl"))
    assert names == [f"{AGENT}_thread-a.jsonl", f"{AGENT}_thread-b.jsonl"]
    assert load_trajectory(store / names[0]).working_dir == str(first.resolve())
    assert load_trajectory(store / names[1]).working_dir == str(second.resolve())
    assert not (first_state / "trajectories").exists()


def test_a_relative_working_dir_is_recorded_absolute(
    scope,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative -w must not reach the file: nothing could attribute it later."""
    scope("shared")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    _record_turn(_context(Path("."), "thread-rel"))

    trajectory = load_trajectory(_shared_store() / f"{AGENT}_thread-rel.jsonl")
    assert trajectory.working_dir == str(work.resolve())


# ----------------------------------------------------------------- readers


def test_readers_see_one_workspace_of_the_shared_store(
    scope,
    tmp_path: Path,
) -> None:
    scope("shared")
    first, second = _two_workspaces(tmp_path)

    store = export.resolve_trajectories_dir(working_dir=first)
    assert store == export.resolve_trajectories_dir(working_dir=second)
    assert store == _shared_store()
    workspace = export.workspace_filter(first)
    assert workspace == first.resolve()

    listed = export.list_trajectories(store, workspace=workspace)
    assert [summary.thread_id for summary in listed] == ["thread-a"]
    assert listed[0].working_dir == str(first.resolve())
    everything = sorted(item.thread_id for item in export.list_trajectories(store))
    assert everything == ["thread-a", "thread-b"]
    assert export.find_trajectory_file(store, "thread-b", workspace=workspace) is None
    found = export.find_trajectory_file(store, "thread-a", workspace=workspace)
    assert found is not None
    loaded = load_trajectories(store, workspace=workspace)
    assert [trajectory.thread_id for trajectory in loaded] == ["thread-a"]


def test_the_workspace_scope_does_not_filter(scope, tmp_path: Path) -> None:
    scope("workspace")
    assert export.workspace_filter(tmp_path) is None


def test_export_cli_lists_this_workspace_unless_asked_for_all(
    scope,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scope("shared")
    first, second = _two_workspaces(tmp_path)

    assert export.main(["list", "-w", str(first)]) == 0
    mine = capsys.readouterr().out
    assert "thread-a" in mine
    assert "thread-b" not in mine
    assert "workspace=" not in mine

    assert export.main(["list", "-w", str(first), "--all-workspaces"]) == 0
    everything = capsys.readouterr().out
    assert "thread-a" in everything
    assert f"workspace={second.resolve()}" in everything

    assert export.main(["show", "-w", str(first), "--thread", "thread-b"]) == 1
    assert "No trajectory found" in capsys.readouterr().err
