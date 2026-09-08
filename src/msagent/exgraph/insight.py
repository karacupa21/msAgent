#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""Insight nodes: accepted SkillDoc + high-support Recipe. No generated prose."""

from __future__ import annotations

from typing import Any, Iterable

from msagent.exgraph.schema import Edge, ExperienceGraph, Node, edge_id, insight_id


def accepted_skills(graphs: Iterable[ExperienceGraph]) -> list[Node]:
    found: list[Node] = []
    for graph in graphs:
        for node in graph.nodes.values():
            if node.type == "SkillDoc" and node.attrs.get("status") == "accepted":
                found.append(node)
    return found


def insight_nodes(
    recipe_rows: list[dict[str, Any]],
    graphs: Iterable[ExperienceGraph],
    *,
    min_support: int = 2,
) -> tuple[list[Node], list[Edge]]:
    """One Insight per recipe that meets support and has an accepted skill."""
    graph_list = list(graphs)
    skills = accepted_skills(graph_list)
    if not skills:
        return [], []
    thread_to_skills: dict[str, list[Node]] = {}
    for graph in graph_list:
        names = [
            node
            for node in graph.nodes.values()
            if node.type == "SkillDoc" and node.attrs.get("status") == "accepted"
        ]
        if names:
            thread_to_skills[graph.thread_id] = names
    nodes: list[Node] = []
    edges: list[Edge] = []
    for row in recipe_rows:
        if row.get("type") != "Recipe":
            continue
        support = int(row.get("support") or 0)
        if support < min_support:
            continue
        thread_ids = [str(tid) for tid in (row.get("thread_ids") or [])]
        linked = [skill for tid in thread_ids for skill in thread_to_skills.get(tid, [])]
        if not linked:
            continue
        ngram = list(row.get("ngram") or [])
        rid = str(row.get("id") or "")
        iid = insight_id(rid)
        skill_names = list(dict.fromkeys(str(s.attrs.get("name") or s.id) for s in linked))
        nodes.append(
            Node(
                id=iid,
                type="Insight",
                attrs={
                    "recipe": rid,
                    "ngram": ngram,
                    "support": support,
                    "skills": skill_names,
                    "thread_ids": thread_ids,
                },
            )
        )
        edges.append(
            Edge(
                id=edge_id("INSIGHT_OF", iid, rid),
                type="INSIGHT_OF",
                src=iid,
                dst=rid,
            )
        )
        for skill in linked:
            edges.append(
                Edge(
                    id=edge_id("INSIGHT_SKILL", iid, skill.id),
                    type="INSIGHT_SKILL",
                    src=iid,
                    dst=skill.id,
                )
            )
    return nodes, edges
