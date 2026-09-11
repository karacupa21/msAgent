#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""P2.2 classify A/B is deterministic and invents no evidence seqs."""

from __future__ import annotations

from pathlib import Path

import pytest

from msagent.exgraph.config import ENV_DISABLED, ENV_ENABLED, reset_config_cache


@pytest.fixture(autouse=True)
def _exgraph_opt_in(monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.delenv(ENV_DISABLED, raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


from msagent.exgraph.ab import compare_trajectory, render_markdown
from msagent.trajectory_recorder.reader import load_trajectory

REPO = Path(__file__).resolve().parents[3]
SIGNALS = REPO / "tests" / "fixtures" / "trajectories" / "skill_evolver_signals.jsonl"


def test_ab_episodes_has_no_appendix(tmp_path: Path) -> None:
    traj = load_trajectory(SIGNALS)
    work = tmp_path / "proj"
    work.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    report = compare_trajectory(
        traj, working_dir=work, state_dir=state, modes=("episodes", "hybrid")
    )
    by_mode = {snap.mode: snap for snap in report.modes}
    assert by_mode["episodes"].has_graph is False
    assert by_mode["episodes"].appendix_chars == 0
    assert "Experience graph" not in by_mode["episodes"].text
    assert by_mode["hybrid"].has_graph is True
    assert by_mode["hybrid"].chars > by_mode["episodes"].chars
    assert by_mode["hybrid"].invented_evidence is False
    assert "thread-signals" in render_markdown(report)


def test_ab_graph_puts_relations_first(tmp_path: Path) -> None:
    traj = load_trajectory(SIGNALS)
    work = tmp_path / "proj"
    work.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    report = compare_trajectory(
        traj, working_dir=work, state_dir=state, modes=("graph",)
    )
    snap = report.modes[0]
    assert snap.graph_first is True
    assert snap.invented_evidence is False
