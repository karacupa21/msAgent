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

"""P0 experience-graph ingest: cases, tree, store, skill attachments."""

from __future__ import annotations

from pathlib import Path

import pytest

from msagent.exgraph.config import ENV_DISABLED, ENV_ENABLED, reset_config_cache


@pytest.fixture(autouse=True)
def _exgraph_opt_in(monkeypatch):
    """Exgraph tests run opted-in. Product default remains off."""
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.delenv(ENV_DISABLED, raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


from msagent.exgraph.cases import build_from_path, label_outcome
from msagent.exgraph.export import render_markdown
from msagent.exgraph.schema import case_id, step_id, task_anchor_id, thread_id
from msagent.exgraph.skills import attach_skill_docs
from msagent.exgraph.store import load_graph, save_graph
from msagent.trajectory_recorder.model import Turn
from msagent.trajectory_recorder.reader import load_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "trajectories"


def test_normal_session_cases() -> None:
    graph = build_from_path(FIXTURES / "normal_subagent.jsonl")
    assert graph.thread_id == "thread-normal"
    assert graph.agent == "Profiler"
    assert thread_id("thread-normal") in graph.nodes
    assert task_anchor_id("thread-normal") in graph.nodes
    assert graph.nodes[task_anchor_id("thread-normal")].attrs["grain"] == "thread"
    assert set(graph.cases) == {case_id("run-1"), case_id("run-2")}

    first = graph.cases[case_id("run-1")]
    second = graph.cases[case_id("run-2")]
    assert first.r == "unknown"
    assert first.x == "Analyse the profile"
    assert first.sigma["tool_path"] == ["get_skill", "task", "ls"]
    assert first.sigma["skills_consulted"] == ["cluster-analysis"]
    assert "tools:b2" in first.sigma["subagents"]
    assert first.sigma["approvals"] == 1

    assert second.r == "warning"
    assert second.sigma["errors"]
    assert any(error["tool"] == "bash" for error in second.sigma["errors"])


def test_parent_and_subagent_edges() -> None:
    graph = build_from_path(FIXTURES / "normal_subagent.jsonl")
    kinds = {(edge.type, edge.src, edge.dst) for edge in graph.edges.values()}
    assert (
        "HAS_TASK",
        thread_id("thread-normal"),
        task_anchor_id("thread-normal"),
    ) in kinds
    assert (
        "CONTAINS",
        task_anchor_id("thread-normal"),
        case_id("run-1"),
    ) in kinds
    assert ("NEXT_CASE", case_id("run-1"), case_id("run-2")) in kinds
    # ls ran inside the subagent namespace captured by the reader.
    assert any(
        edge.type == "IN_SUBAGENT" and edge.src == step_id("s6")
        for edge in graph.edges.values()
    )
    assert step_id("s6") in graph.nodes
    assert graph.nodes[step_id("s6")].attrs["kind"] == "tool"


def test_outcome_override_success_skips_hard_failures() -> None:
    graph = build_from_path(FIXTURES / "normal_subagent.jsonl", outcome="success")
    assert graph.cases[case_id("run-1")].r == "golden"
    assert graph.cases[case_id("run-2")].r == "warning"


def test_outcome_override_fail() -> None:
    graph = build_from_path(FIXTURES / "normal_subagent.jsonl", outcome="fail")
    assert graph.cases[case_id("run-1")].r == "warning"
    assert graph.cases[case_id("run-2")].r == "warning"


def test_label_policy_unit() -> None:
    clean = Turn(
        run_id="r",
        seq_start=1,
        line_start=1,
        user_message="go",
        source="dispatch",
        status="completed",
    )
    assert label_outcome(clean) == "unknown"
    assert label_outcome(clean, override="success") == "golden"
    failed = Turn(
        run_id="r",
        seq_start=1,
        line_start=1,
        user_message="go",
        source="dispatch",
        status="error",
    )
    assert label_outcome(failed, override="success") == "warning"


def test_store_upsert_is_idempotent(tmp_path: Path) -> None:
    source = FIXTURES / "normal_subagent.jsonl"
    first = build_from_path(source)
    directory = save_graph(first, tmp_path)
    n_nodes, n_edges, n_cases = len(first.nodes), len(first.edges), len(first.cases)
    second = build_from_path(source)
    save_graph(second, tmp_path)
    loaded = load_graph(directory)
    assert len(loaded.nodes) == n_nodes
    assert len(loaded.edges) == n_edges
    assert len(loaded.cases) == n_cases
    assert loaded.cases[case_id("run-2")].r == "warning"


def test_show_mentions_both_cases() -> None:
    text = render_markdown(build_from_path(FIXTURES / "normal_subagent.jsonl"))
    assert "run-1" in text
    assert "run-2" in text
    assert "warning" in text


def test_skill_doc_from_proposals(tmp_path: Path) -> None:
    work = tmp_path / "proj"
    proposal = work / ".proposals" / "thread-normal" / "cluster-tune"
    proposal.mkdir(parents=True)
    (proposal / "SKILL.md").write_text("# cluster-tune\n", encoding="utf-8")
    (proposal / "provenance.json").write_text(
        '{"thread_ids": ["thread-normal"], "candidates": ["x"]}',
        encoding="utf-8",
    )
    graph = build_from_path(FIXTURES / "normal_subagent.jsonl")
    attach_skill_docs(graph, working_dir=work)
    skills = [node for node in graph.nodes.values() if node.type == "SkillDoc"]
    assert len(skills) == 1
    assert skills[0].attrs["name"] == "cluster-tune"
    assert any(edge.type == "DERIVED_SKILL" for edge in graph.edges.values())


def test_cli_build_from_path(tmp_path: Path) -> None:
    from msagent.exgraph.export import main

    code = main(
        [
            "build",
            "--path",
            str(FIXTURES / "normal_subagent.jsonl"),
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert code == 0
    saved = list(tmp_path.joinpath("exgraph").glob("*_thread-normal"))
    assert len(saved) == 1
    assert (saved[0] / "cases.jsonl").is_file()


def test_reader_fixture_still_loads() -> None:
    # Guard against accidental coupling: ingest must keep using the public model.
    trajectory = load_trajectory(FIXTURES / "normal_subagent.jsonl")
    assert [turn.run_id for turn in trajectory.turns] == ["run-1", "run-2"]
