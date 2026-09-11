#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""SIMILAR_TO is same-agent, cross-thread, no embeddings."""

from __future__ import annotations

import pytest

from msagent.exgraph.config import ENV_DISABLED, ENV_ENABLED, reset_config_cache
from msagent.exgraph.schema import CaseRecord, ExperienceGraph
from msagent.exgraph.similar import pair_scores, similar_pairs


@pytest.fixture(autouse=True)
def _exgraph_opt_in(monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.delenv(ENV_DISABLED, raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


def _case(thread: str, run: str, agent: str, x: str, tools: list[str]) -> CaseRecord:
    return CaseRecord(
        id=f"case:{run}",
        thread_id=thread,
        agent=agent,
        run_id=run,
        x=x,
        y="",
        r="unknown",
        sigma={"tool_path": tools},
        evidence=[],
    )


def _graph(thread: str, agent: str, record: CaseRecord) -> ExperienceGraph:
    graph = ExperienceGraph(schema_version=3, thread_id=thread, agent=agent, source_path="")
    graph.add_case(record)
    return graph


def test_profiler_sessions_are_similar() -> None:
    left = _case(
        "thread-signals",
        "run-1",
        "Profiler",
        "Profile the training run and find the bottleneck",
        ["bash", "read_file"],
    )
    right = _case(
        "thread-reuse",
        "run-r1",
        "Profiler",
        "Profile the training run and find the kernel-level bottleneck",
        ["bash", "read_file", "grep"],
    )
    tools, tokens = pair_scores(left, right)
    assert tools >= 0.5
    assert tokens >= 0.25
    pairs = similar_pairs(
        [_graph("thread-signals", "Profiler", left), _graph("thread-reuse", "Profiler", right)]
    )
    assert len(pairs) == 1


def test_accuracy_is_not_similar_to_profiler() -> None:
    left = _case("t-p", "run-1", "Profiler", "Profile the training run bottleneck", ["bash", "read_file"])
    right = _case("t-a", "run-a1", "Accuracy", "Loss became NaN after step 1200", ["read_file", "grep"])
    pairs = similar_pairs(
        [_graph("t-p", "Profiler", left), _graph("t-a", "Accuracy", right)],
        min_tools=0.3,
        min_tokens=0.1,
    )
    assert pairs == []


def test_bm25_matches_han_where_legacy_jaccard_does_not() -> None:
    from msagent.exgraph.similar import tokenize as legacy_tokenize

    left = _case("t1", "r1", "Profiler", "查找训练性能瓶颈", ["bash", "read_file"])
    right = _case("t2", "r2", "Profiler", "分析训练任务的性能瓶颈", ["bash", "read_file"])
    assert legacy_tokenize(left.x) == set() or not (
        legacy_tokenize(left.x) & legacy_tokenize(right.x)
    )
    tools, tokens = pair_scores(left, right, text_backend="bm25")
    assert tools == 1.0
    assert tokens >= 0.25
    j_tools, j_tokens = pair_scores(left, right, text_backend="jaccard")
    assert j_tools == 1.0
    assert j_tokens == 0.0


def test_english_profiler_still_similar_under_bm25() -> None:
    left = _case(
        "thread-signals",
        "run-1",
        "Profiler",
        "Profile the training run and find the bottleneck",
        ["bash", "read_file"],
    )
    right = _case(
        "thread-reuse",
        "run-r1",
        "Profiler",
        "Profile the training run and find the kernel-level bottleneck",
        ["bash", "read_file", "grep"],
    )
    tools, tokens = pair_scores(left, right, text_backend="bm25")
    assert tools >= 0.5
    assert tokens >= 0.25
