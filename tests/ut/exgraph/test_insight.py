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
