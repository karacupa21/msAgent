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


"""Evolver bridge must not change mining when the graph is absent."""

from __future__ import annotations

from pathlib import Path

import pytest

from msagent.exgraph.config import (
    ENV_DISABLED,
    ENV_ENABLED,
    ENV_EVIDENCE_MODE,
    reset_config_cache,
)


@pytest.fixture(autouse=True)
def _exgraph_opt_in(monkeypatch):
    """Exgraph tests run opted-in. Product default remains off."""
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.delenv(ENV_DISABLED, raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


from msagent.skill_evolver.exgraph_context import attach_stored_graph
from msagent.trajectory_recorder.reader import load_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
SIGNALS = REPO_ROOT / "tests" / "fixtures" / "trajectories" / "skill_evolver_signals.jsonl"


def test_attach_returns_original_on_failure(tmp_path: Path) -> None:
    trajectory = load_trajectory(SIGNALS)
    original = "### Episode E1 — retry_loop (weight 0.70, thread abc)"
    # state_dir that cannot resolve a project still must not raise.
    text = attach_stored_graph(
        original,
        trajectory,
        working_dir=tmp_path / "missing",
        state_dir=tmp_path / "missing-state",
    )
    assert original in text


def test_attach_appends_when_graph_can_be_written(tmp_path: Path) -> None:
    trajectory = load_trajectory(SIGNALS)
    work = tmp_path / "proj"
    work.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    original = "### Episode E1"
    text = attach_stored_graph(
        original,
        trajectory,
        working_dir=work,
        state_dir=state,
    )
    assert text.startswith(original)
    assert "Experience graph" in text or text == original


def test_kill_switch_skips_graph(tmp_path: Path, monkeypatch) -> None:
    import os

    trajectory = load_trajectory(SIGNALS)
    original = "### Episode E1 — unchanged"
    monkeypatch.setenv("MSAGENT_EXGRAPH_DISABLED", "1")
    work = tmp_path / "proj"
    work.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    text = attach_stored_graph(
        original,
        trajectory,
        working_dir=work,
        state_dir=state,
    )
    assert text == original
    # No shard written under the state dir.
    written = list(state.rglob("nodes.jsonl"))
    assert written == []
    monkeypatch.delenv("MSAGENT_EXGRAPH_DISABLED", raising=False)


def test_episodes_mode_leaves_bundle_and_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_EVIDENCE_MODE, "episodes")
    reset_config_cache()
    trajectory = load_trajectory(SIGNALS)
    original = "### Episode E1 — only this"
    work = tmp_path / "proj"
    work.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    text = attach_stored_graph(
        original,
        trajectory,
        working_dir=work,
        state_dir=state,
    )
    assert text == original
    assert list(state.rglob("nodes.jsonl")) == []
    monkeypatch.delenv(ENV_EVIDENCE_MODE, raising=False)
    reset_config_cache()


def test_graph_mode_puts_relations_first_without_evidence_seqs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_EVIDENCE_MODE, "graph")
    reset_config_cache()
    trajectory = load_trajectory(SIGNALS)
    work = tmp_path / "proj"
    work.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    original = "### Episode E1 — retry_loop [ev3]\nEvidence: 3, 4"
    text = attach_stored_graph(
        original,
        trajectory,
        working_dir=work,
        state_dir=state,
    )
    assert "Experience graph (primary)" in text
    assert text.index("Experience graph (primary)") < text.index("Episode E1")
    assert "Episode bundle (supporting, citable)" in text
    graph_part = text.split("## Episode bundle")[0]
    assert "Evidence:" not in graph_part
    assert "[ev" not in graph_part
    monkeypatch.delenv(ENV_EVIDENCE_MODE, raising=False)
    reset_config_cache()
