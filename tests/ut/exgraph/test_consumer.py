#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""Classify appendix lists relations only (P1.1)."""

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


from msagent.exgraph.consumer import render_thread_context
from msagent.exgraph.schema import (
    SCHEMA_VERSION,
    CaseRecord,
    Edge,
    ExperienceGraph,
    Node,
)
from msagent.exgraph.store import save_graph


def _graph() -> ExperienceGraph:
    graph = ExperienceGraph(
        schema_version=SCHEMA_VERSION,
        thread_id="thread-signals",
        agent="Profiler",
        source_path="",
    )
    graph.add_node(Node(id="thread:thread-signals", type="Thread", attrs={}))
    graph.add_case(
        CaseRecord(
            id="case:run-1",
            thread_id="thread-signals",
            agent="Profiler",
            run_id="run-1",
            x="profile",
            y="retry",
            r="warning",
            sigma={
                "tool_path": ["bash", "read_file"],
                "errors": [{"tool": "bash", "error": "unknown option --collect"}],
            },
            evidence=[2, 5],
        )
    )
    graph.add_edge(
        Edge(
            id="e1",
            type="FIXED_BY",
            src="case:run-1",
            dst="case:run-2",
            attrs={"via": "user_correction"},
        )
    )
    graph.add_node(
        Node(
            id="skill:proposal:thread-signals:msprof-kernel-profile",
            type="SkillDoc",
            attrs={
                "name": "msprof-kernel-profile",
                "status": "proposal",
                "path": "/tmp/SKILL.md",
            },
        )
    )
    graph.add_node(
        Node(
            id="episode:1",
            type="Episode",
            attrs={"kind": "user_correction", "weight": 0.9},
        )
    )
    return graph


def test_appendix_has_relations_not_episode_echo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MSAGENT_EXGRAPH_DISABLED", raising=False)
    save_graph(_graph(), tmp_path / "exgraph")
    text = render_thread_context(
        "thread-signals",
        working_dir=tmp_path,
        state_dir=tmp_path,
    )
    assert "Experience graph" in text
    assert "outcome=warning" in text
    assert "err=bash:" in text
    assert "fixed_by" in text
    assert "SkillDoc proposal msprof-kernel-profile" in text
    assert "Episodes:" not in text
    assert "user_correction weight" not in text
    assert "Evidence:" not in text
    assert len(text) <= 1600


def test_appendix_empty_when_killed(tmp_path: Path, monkeypatch) -> None:
    save_graph(_graph(), tmp_path / "exgraph")
    monkeypatch.setenv("MSAGENT_EXGRAPH_DISABLED", "1")
    text = render_thread_context(
        "thread-signals",
        working_dir=tmp_path,
        state_dir=tmp_path,
    )
    assert text == ""
