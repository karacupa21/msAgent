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


"""Workspace overlay for Recipes (cross-thread only).

Thread shards stay the source of truth for Cases. This overlay is rebuilt
from ``mine_cross_session`` over a caller-supplied pool — exgraph never
selects the last-N history itself. The pool size is the evolver's
``CROSS_SESSION_LIMIT`` when the CLI passes that list in.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from msagent.exgraph.schema import (
    SCHEMA_VERSION,
    Edge,
    ExperienceGraph,
    Node,
    edge_id,
    recipe_id,
)
from msagent.exgraph.similar import similar_edges
from msagent.exgraph.insight import insight_nodes
from msagent.exgraph.store import WORKSPACE_NAME, _read_jsonl, _write_jsonl
from msagent.skill_evolver.features import FEATURES_VERSION, mine_cross_session
from msagent.trajectory_recorder.model import Trajectory


def workspace_dir(root: Path) -> Path:
    return Path(root) / WORKSPACE_NAME


def _contains_ngram(path: list[str], ngram: list[str]) -> bool:
    size = len(ngram)
    if size == 0 or size > len(path):
        return False
    return any(path[i : i + size] == ngram for i in range(len(path) - size + 1))


def rebuild_overlay(
    pool: list[Trajectory],
    graphs: Iterable[ExperienceGraph],
    root: Path,
) -> Path:
    """Replace the overlay from ``mine_cross_session``.

    Mining is **per agent**. A Profiler n-gram must not become an Accuracy
    recipe even when the tool names coincide (``read_file>grep``).
    The pool itself stays caller-owned; we only partition it.
    """
    directory = workspace_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    by_id = {graph.thread_id: graph for graph in graphs}
    by_agent: dict[str, list[Trajectory]] = {}
    for traj in pool:
        by_agent.setdefault(traj.agent or "", []).append(traj)
    recipes: list = []
    for agent_pool in by_agent.values():
        if len(agent_pool) < 2:
            continue
        recipes.extend(mine_cross_session(agent_pool))
    for episode in recipes:
        ngram = list(episode.facts.get("ngram") or episode.tool_sequence)
        nid = recipe_id(ngram)
        thread_ids = list(episode.facts.get("thread_ids") or [])
        existing = nodes.get(nid)
        if existing is not None:
            merged = list(dict.fromkeys(list(existing.get("thread_ids") or []) + thread_ids))
            existing["thread_ids"] = merged
            existing["support"] = max(int(existing.get("support") or 0), int(episode.facts.get("support") or 0))
            thread_ids = merged
        else:
            nodes[nid] = Node(
                id=nid,
                type="Recipe",
                attrs={
                    "ngram": ngram,
                    "support": episode.facts.get("support"),
                    "thread_ids": thread_ids,
                    "features_version": FEATURES_VERSION,
                },
            ).to_dict()
        allowed_agents = {
            by_id[thread].agent
            for thread in thread_ids
            if thread in by_id
        }
        for thread in thread_ids:
            graph = by_id.get(thread)
            if graph is None:
                continue
            for record in graph.cases.values():
                if allowed_agents and record.agent not in allowed_agents:
                    continue
                path = [str(name) for name in (record.sigma.get("tool_path") or [])]
                if _contains_ngram(path, [str(part) for part in ngram]):
                    ident = edge_id("INSTANTIATES", record.id, nid)
                    edges[ident] = Edge(
                        id=ident,
                        type="INSTANTIATES",
                        src=record.id,
                        dst=nid,
                    ).to_dict()
    graph_list = list(by_id.values())
    try:
        from msagent.exgraph.config import load_exgraph_config

        cfg = load_exgraph_config()
        min_tools = cfg.similar.min_tools
        min_tokens = cfg.similar.min_tokens
        text_backend = cfg.similar.text_backend
        min_support = cfg.insight.min_support
    except Exception:
        min_tools, min_tokens, min_support, text_backend = 0.5, 0.25, 2, "bm25"
    for edge in similar_edges(
        graph_list,
        min_tools=min_tools,
        min_tokens=min_tokens,
        text_backend=text_backend,
    ):
        edges[edge.id] = edge.to_dict()
    extra_nodes, extra_edges = insight_nodes(
        list(nodes.values()), graph_list, min_support=min_support
    )
    for node in extra_nodes:
        nodes[node.id] = node.to_dict()
    for edge in extra_edges:
        edges[edge.id] = edge.to_dict()
    _write_jsonl(directory / "nodes.jsonl", list(nodes.values()))
    _write_jsonl(directory / "edges.jsonl", list(edges.values()))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "workspace_overlay",
        "features_version": FEATURES_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pool_threads": [traj.thread_id for traj in pool],
        "recipes": len(nodes),
        "edges": len(edges),
    }
    import json

    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return directory


def load_overlay(root: Path) -> tuple[list[dict], list[dict]]:
    directory = workspace_dir(root)
    if not directory.is_dir():
        return [], []
    return _read_jsonl(directory / "nodes.jsonl"), _read_jsonl(directory / "edges.jsonl")
