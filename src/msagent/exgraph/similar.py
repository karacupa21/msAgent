#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""Cross-thread SIMILAR_TO edges. No embeddings.

Same agent only. Tool path stays set Jaccard. User text ``x`` is BM25
via ``skill_evolver.retrieval`` (CJK bigrams) by default, with Jaccard
as the fallback backend.
"""

from __future__ import annotations

import re
from typing import Iterable

from msagent.exgraph.schema import CaseRecord, Edge, ExperienceGraph, edge_id

_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9_-]{3,}")
TEXT_BACKENDS = ("bm25", "jaccard")


def tokenize(text: str | None) -> set[str]:
    """Legacy ASCII/Cyrillic Jaccard tokens. Prefer retrieval.tokenize for BM25."""
    return {token.lower() for token in _TOKEN.findall(text or "")}


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a = {str(item) for item in left if item}
    b = {str(item) for item in right if item}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _bm25_pair(left_text: str, right_text: str) -> float:
    """Self-normalized BM25: score(left→right) / score(left→left)."""
    if not (left_text or "").strip() or not (right_text or "").strip():
        return 0.0
    try:
        from msagent.skill_evolver.retrieval import BM25Index, SkillDoc
    except Exception:
        return jaccard(tokenize(left_text), tokenize(right_text))
    other = BM25Index([SkillDoc(name="r", description=right_text)])
    hits = other.search(left_text, top_k=1)
    if not hits:
        return 0.0
    self_idx = BM25Index([SkillDoc(name="l", description=left_text)])
    self_hits = self_idx.search(left_text, top_k=1)
    denom = self_hits[0].score if self_hits else 0.0
    if denom <= 0:
        return 0.0
    return min(1.0, hits[0].score / denom)


def pair_scores(
    left: CaseRecord,
    right: CaseRecord,
    *,
    text_backend: str = "bm25",
) -> tuple[float, float]:
    tools = jaccard(left.sigma.get("tool_path") or [], right.sigma.get("tool_path") or [])
    backend = text_backend if text_backend in TEXT_BACKENDS else "bm25"
    if backend == "jaccard":
        tokens = jaccard(tokenize(left.x), tokenize(right.x))
    else:
        tokens = _bm25_pair(left.x or "", right.x or "")
    return tools, tokens


def similar_pairs(
    graphs: Iterable[ExperienceGraph],
    *,
    min_tools: float = 0.5,
    min_tokens: float = 0.25,
    text_backend: str = "bm25",
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
            tools, tokens = pair_scores(left, right, text_backend=text_backend)
            if tools >= min_tools and tokens >= min_tokens:
                pairs.append((left, right, tools, tokens))
    return pairs


def similar_edges(
    graphs: Iterable[ExperienceGraph],
    *,
    min_tools: float = 0.5,
    min_tokens: float = 0.25,
    text_backend: str = "bm25",
) -> list[Edge]:
    edges: list[Edge] = []
    backend = text_backend if text_backend in TEXT_BACKENDS else "bm25"
    for left, right, tools, tokens in similar_pairs(
        graphs, min_tools=min_tools, min_tokens=min_tokens, text_backend=backend
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
                    "backend": backend,
                },
            )
        )
    return edges
