#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""Insight requires an accepted SkillDoc plus a supported recipe."""

from __future__ import annotations

import pytest

from msagent.exgraph.config import ENV_DISABLED, ENV_ENABLED, reset_config_cache
from msagent.exgraph.insight import insight_nodes
from msagent.exgraph.schema import ExperienceGraph, Node


@pytest.fixture(autouse=True)
def _exgraph_opt_in(monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.delenv(ENV_DISABLED, raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


def test_no_insight_without_accepted_skill() -> None:
    graph = ExperienceGraph(schema_version=3, thread_id="t1", agent="Profiler", source_path="")
    recipes = [
        {
            "id": "recipe:bash>read_file",
            "type": "Recipe",
            "ngram": ["bash", "read_file"],
            "support": 2,
            "thread_ids": ["t1"],
        }
    ]
    nodes, edges = insight_nodes(recipes, [graph], min_support=2)
    assert nodes == []
    assert edges == []


def test_insight_links_recipe_to_accepted_skill() -> None:
    graph = ExperienceGraph(schema_version=3, thread_id="t1", agent="Profiler", source_path="")
    graph.add_node(
        Node(
            id="skill:msprof-kernel-profile",
            type="SkillDoc",
            attrs={"name": "msprof-kernel-profile", "status": "accepted"},
        )
    )
    recipes = [
        {
            "id": "recipe:bash>read_file",
            "type": "Recipe",
            "ngram": ["bash", "read_file"],
            "support": 2,
            "thread_ids": ["t1"],
        }
    ]
    nodes, edges = insight_nodes(recipes, [graph], min_support=2)
    assert len(nodes) == 1
    assert nodes[0].type == "Insight"
    assert "msprof-kernel-profile" in nodes[0].attrs["skills"]
    kinds = {edge.type for edge in edges}
    assert "INSIGHT_OF" in kinds
    assert "INSIGHT_SKILL" in kinds


class _FakeLlm:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def ainvoke(self, payload):
        self.calls += 1
        return type("Reply", (), {"content": self.text})()


def test_compose_strips_evidence_markers() -> None:
    import asyncio

    from msagent.exgraph.insight import compose_insight_text

    llm = _FakeLlm("Prefer --type=operator. Evidence: 12 [ev3] leftover")
    text = asyncio.run(
        compose_insight_text(llm, {"ngram": ["bash"], "skills": ["s"], "support": 2})
    )
    assert "Evidence" not in text
    assert "[ev" not in text
    assert "Prefer --type=operator." in text
    assert llm.calls == 1


def test_fill_overlay_writes_text(tmp_path) -> None:
    import asyncio
    import json

    from msagent.exgraph.insight import fill_overlay_insights

    overlay = tmp_path / "workspace"
    overlay.mkdir()
    (overlay / "nodes.jsonl").write_text(
        '{"id":"insight:bash","type":"Insight","ngram":["bash"],"skills":["s"],"support":2}\n',
        encoding="utf-8",
    )
    llm = _FakeLlm("Reuse the working flag set.")
    assert asyncio.run(fill_overlay_insights(llm, overlay)) == 1
    row = json.loads((overlay / "nodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["text"] == "Reuse the working flag set."
    assert asyncio.run(fill_overlay_insights(llm, overlay)) == 0
    assert llm.calls == 1
