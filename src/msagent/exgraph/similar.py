#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""Cross-thread SIMILAR_TO edges. No embeddings, no evolver copy.

Same agent only. Token Jaccard on the user text ``x`` plus set Jaccard on
the case tool path. Thresholds come from the caller (config).
"""

from __future__ import annotations

import re
from typing import Iterable

from msagent.exgraph.schema import CaseRecord, Edge, ExperienceGraph, edge_id

_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9_-]{3,}")


def tokenize(text: str | None) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text or "")}


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a = {str(item) for item in left if item}
    b = {str(item) for item in right if item}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_scores(left: CaseRecord, right: CaseRecord) -> tuple[float, float]:
    tools = jaccard(left.sigma.get("tool_path") or [], right.sigma.get("tool_path") or [])
    tokens = jaccard(tokenize(left.x), tokenize(right.x))
    return tools, tokens


def similar_pairs(
    graphs: Iterable[ExperienceGraph],
    *,
    min_tools: float = 0.5,
    min_tokens: float = 0.25,
) -> list[tuple[CaseRecord, CaseRecord, float, float]]:
    """Cross-thread, same-agent pairs that clear both thresholds."""
    cases = [record for graph in graphs for record in graph.cases.values()]
    pairs: list[tuple[CaseRecord, CaseRecord, float, float]] = []
    for index, left in enumerate(cases):
        left_tools = left.sigma.get("tool_path") or []
        if not left_tools:
            continue
        for right in cases[index + 1 :]:
            if left.thread_id == right.thread_id:
                continue
            if (left.agent or "") != (right.agent or ""):
                continue
            if not (right.sigma.get("tool_path") or []):
                continue
            tools, tokens = pair_scores(left, right)
            if tools >= min_tools and tokens >= min_tokens:
                pairs.append((left, right, tools, tokens))
    return pairs


def similar_edges(
    graphs: Iterable[ExperienceGraph],
    *,
    min_tools: float = 0.5,
    min_tokens: float = 0.25,
) -> list[Edge]:
    edges: list[Edge] = []
    for left, right, tools, tokens in similar_pairs(
        graphs, min_tools=min_tools, min_tokens=min_tokens
    ):
        src, dst = sorted((left.id, right.id))
        edges.append(
            Edge(
                id=edge_id("SIMILAR_TO", src, dst),
                type="SIMILAR_TO",
                src=src,
                dst=dst,
                attrs={
                    "tools": round(tools, 3),
                    "tokens": round(tokens, 3),
                },
            )
        )
    return edges
