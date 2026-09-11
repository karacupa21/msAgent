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

"""Unit tests for the /trajectories command."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from msagent.cli.handlers import trajectories as module
from msagent.cli.theme import theme
from msagent.trajectory_recorder.config import reset_config_cache

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "trajectories"
SIGNALS = FIXTURES / "skill_evolver_signals.jsonl"
AGENT = "Profiler"
THREAD = "thread-signals"


class _ConsoleSpy:
    """Records the themed console calls and the renderables printed."""

    def __init__(self) -> None:
        self.info: list[str] = []
        self.error: list[str] = []
        self.plain: list[str] = []
        self.renderables: list[object] = []
        self.console = SimpleNamespace(print=self.renderables.append)

    def print(self, *args, **_kwargs) -> None:
        self.plain.extend(str(arg) for arg in args)

    def print_info(self, content: str) -> None:
        self.info.append(content)

    def print_success(self, content: str) -> None:
        self.info.append(content)

    def print_warning(self, content: str) -> None:
        self.info.append(content)

    def print_error(self, content: str) -> None:
        self.error.append(content)

    def rendered_text(self) -> str:
        """Everything printed as a renderable, as plain text."""
        recorder = Console(
            record=True,
            width=200,
            no_color=True,
            theme=theme.rich_theme,
        )
        for renderable in self.renderables:
            recorder.print(renderable)
        return recorder.export_text()


@pytest.fixture
def browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A trajectories handler over an isolated project state directory."""
    spy = _ConsoleSpy()
    monkeypatch.setattr(module, "console", spy)
    monkeypatch.delenv("MSAGENT_TRAJECTORY_CONFIG", raising=False)
    reset_config_cache()

    directory = module.initializer.get_project_paths(tmp_path).root / "trajectories"
    directory.mkdir(parents=True)
    session = SimpleNamespace(
        context=SimpleNamespace(
            agent=AGENT,
            thread_id="thread-current",
            working_dir=tmp_path,
            model="default",
        ),
    )
    state = SimpleNamespace(
        handler=module.TrajectoriesHandler(session),
        spy=spy,
        directory=directory,
    )
    yield state
    reset_config_cache()


def _place(directory: Path, thread_id: str = THREAD) -> Path:
    target = directory / f"{AGENT}_{thread_id}.jsonl"
    shutil.copy(SIGNALS, target)
    return target


def test_short_thread_clips_but_stays_resolvable() -> None:
    assert module.short_thread("abc") == "abc"
    clipped = module.short_thread("0123456789abcdef")
    assert clipped == "0123456789ab…"


@pytest.mark.asyncio
async def test_list_renders_a_row_per_thread(browser) -> None:
    _place(browser.directory)

    await browser.handler.handle([])

    assert browser.spy.error == []
    text = browser.spy.rendered_text()
    assert "Recorded trajectories" in text
    assert "thread-signa" in text
    assert AGENT in text


@pytest.mark.asyncio
async def test_list_reports_an_empty_directory(browser) -> None:
    await browser.handler.handle(["list"])

    assert any("No trajectories recorded" in line for line in browser.spy.info)
    assert browser.spy.renderables == []


@pytest.mark.asyncio
async def test_show_renders_markdown(browser) -> None:
    _place(browser.directory)

    await browser.handler.handle(["show", THREAD])

    assert browser.spy.error == []
    text = browser.spy.rendered_text()
    assert "Trajectory" in text
    assert "Turn" in text


@pytest.mark.asyncio
async def test_show_accepts_a_prefix(browser) -> None:
    _place(browser.directory)

    await browser.handler.handle(["show", "thread-sig"])

    assert browser.spy.error == []
    assert browser.spy.renderables


@pytest.mark.asyncio
async def test_show_reports_an_unknown_thread(browser) -> None:
    _place(browser.directory)

    await browser.handler.handle(["show", "nope"])

    assert any("No recorded trajectory" in line for line in browser.spy.error)
    assert browser.spy.renderables == []


@pytest.mark.asyncio
async def test_show_requires_a_thread_id(browser) -> None:
    await browser.handler.handle(["show"])

    assert any("requires a thread id" in line for line in browser.spy.error)


@pytest.mark.asyncio
async def test_unknown_subcommand(browser) -> None:
    await browser.handler.handle(["delete"])

    assert any("Unknown subcommand" in line for line in browser.spy.error)
    assert any("show <thread-id>" in line for line in browser.spy.plain)


# ------------------------------------------------------------ shared store


@pytest.fixture
def shared(browser, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The browser over the shared store (output.scope: shared)."""
    config = tmp_path / "config.trajectory.recorder.yml"
    config.write_text("output:\n  scope: shared\n", encoding="utf-8")
    monkeypatch.setenv("MSAGENT_TRAJECTORY_CONFIG", str(config))
    reset_config_cache()
    browser.directory = module.initializer.app_paths.state_dir / "trajectories"
    browser.directory.mkdir(parents=True)
    return browser


def _place_in(directory: Path, thread_id: str, working_dir: Path) -> Path:
    """The signals fixture as recorded in ``working_dir`` under ``thread_id``."""
    lines: list[str] = []
    for raw in SIGNALS.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        event = json.loads(raw)
        event["thread_id"] = thread_id
        if event.get("event") == "recorder.attach":
            event["working_dir"] = str(working_dir)
        lines.append(json.dumps(event, ensure_ascii=False))
    target = directory / f"{AGENT}_{thread_id}.jsonl"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


@pytest.mark.asyncio
async def test_shared_list_shows_only_this_workspace(shared, tmp_path: Path) -> None:
    _place_in(shared.directory, "t-mine", tmp_path)
    _place_in(shared.directory, "t-theirs", tmp_path / "elsewhere")

    await shared.handler.handle(["list"])

    assert shared.spy.error == []
    text = shared.spy.rendered_text()
    assert "t-mine" in text
    assert "t-theirs" not in text


@pytest.mark.asyncio
async def test_shared_list_of_an_idle_workspace_names_it(shared, tmp_path: Path) -> None:
    _place_in(shared.directory, "t-theirs", tmp_path / "elsewhere")

    await shared.handler.handle(["list"])

    assert any("No trajectories recorded" in line and "for workspace" in line for line in shared.spy.info)
    assert shared.spy.renderables == []


@pytest.mark.asyncio
async def test_shared_show_refuses_another_workspaces_thread(shared, tmp_path: Path) -> None:
    _place_in(shared.directory, "t-theirs", tmp_path / "elsewhere")

    await shared.handler.handle(["show", "t-theirs"])

    assert any("No recorded trajectory" in line and "for workspace" in line for line in shared.spy.error)
    assert shared.spy.renderables == []
