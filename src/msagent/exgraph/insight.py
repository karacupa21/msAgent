#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""Insight nodes: accepted SkillDoc + high-support Recipe.

Structural nodes are deterministic. Optional ``text`` is filled later by
the same CountingLlm the evolver already owns — never inside rebuild_overlay.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from msagent.exgraph.schema import Edge, ExperienceGraph, Node, edge_id, insight_id

logger = logging.getLogger(__name__)

INSIGHT_PROMPT = (
    "Write one or two short sentences: the reusable lesson of this agent experience.\n"
    "Do not cite event numbers, Evidence lines, or [evN] ids. No markdown heading.\n"
    "Recipe tools: {ngram}\n"
    "Accepted skills: {skills}\n"
    "Threads: {threads}\n"
    "Support: {support}\n"
)
INSIGHT_TEXT_CAP = 400


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


def insight_prompt(attrs: dict[str, Any]) -> str:
    ngram = ">".join(str(part) for part in (attrs.get("ngram") or []))
    skills = ", ".join(str(name) for name in (attrs.get("skills") or []))
    threads = ", ".join(str(tid) for tid in (attrs.get("thread_ids") or []))
    return INSIGHT_PROMPT.format(
        ngram=ngram or attrs.get("recipe") or attrs.get("id") or "",
        skills=skills or "(none)",
        threads=threads or "(none)",
        support=attrs.get("support") or 0,
    )


def _clean_insight_text(raw: str) -> str:
    text = " ".join(str(raw).split())
    for marker in ("Evidence:", "[ev"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text[:INSIGHT_TEXT_CAP]


async def compose_insight_text(llm: Any, attrs: dict[str, Any]) -> str:
    """One CountingLlm ainvoke. Failures propagate to the caller."""
    from msagent.skill_evolver.classify import reply_text

    raw = reply_text(await llm.ainvoke(insight_prompt(attrs)))
    return _clean_insight_text(raw)


async def fill_overlay_insights(llm: Any, overlay_dir: Path, *, limit: int = 2) -> int:
    """Write ``text`` onto Insight rows that lack it. Returns how many were filled."""
    path = overlay_dir / "nodes.jsonl"
    if not path.is_file() or limit < 1:
        return 0
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    filled = 0
    for row in rows:
        if filled >= limit:
            break
        if row.get("type") != "Insight" or str(row.get("text") or "").strip():
            continue
        try:
            row["text"] = await compose_insight_text(llm, row)
        except Exception:
            logger.debug("insight text skipped for %s", row.get("id"), exc_info=True)
            continue
        if row["text"]:
            filled += 1
    if filled:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    return filled
