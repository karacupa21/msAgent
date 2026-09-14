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

"""P1 enrichment: persist evolver Episodes and FIXED_BY on an L0 graph.

Detectors stay in ``skill_evolver.features``. This module only materializes
their output as graph nodes and correction edges. It never reimplements
n-grams, Jaccard, or the last-N trajectory pool.
"""

from __future__ import annotations

from pathlib import Path

from msagent.exgraph.schema import (
    SCHEMA_VERSION,
    Edge,
    ExperienceGraph,
    Node,
    case_id,
    edge_id,
    episode_id,
)
from msagent.skill_evolver.features import FEATURES_VERSION, Episode, extract_episodes
from msagent.trajectory_recorder.model import Trajectory


def _case_for_seq(graph: ExperienceGraph, seq: int) -> str | None:
    for record in graph.cases.values():
        if seq in record.evidence:
            return record.id
    best: tuple[int, str] | None = None
    for record in graph.cases.values():
        start = min(record.evidence) if record.evidence else None
        if start is None or start > seq:
            continue
        if best is None or start > best[0]:
            best = (start, record.id)
    return best[1] if best else None


def _previous_case(graph: ExperienceGraph, case: str) -> str | None:
    for edge in graph.edges.values():
        if edge.type == "NEXT_CASE" and edge.dst == case:
            return edge.src
    return None


def _link(graph: ExperienceGraph, etype: str, src: str, dst: str, **attrs) -> None:
    graph.add_edge(
        Edge(id=edge_id(etype, src, dst), type=etype, src=src, dst=dst, attrs=attrs),  # type: ignore[arg-type]
    )


def _tool_steps(graph: ExperienceGraph, cid: str) -> list[Node]:
    steps: list[Node] = []
    for edge in graph.edges.values():
        if edge.type != "HAS_STEP" or edge.src != cid:
            continue
        node = graph.nodes.get(edge.dst)
        if node is not None and node.attrs.get("kind") == "tool":
            steps.append(node)
    steps.sort(key=lambda node: int(node.attrs.get("seq_start") or 0))
    return steps


def attach_episodes(graph: ExperienceGraph, trajectory: Trajectory) -> list[Episode]:
    """Add Episode nodes from ``extract_episodes`` (no skill index in P1)."""
    episodes = extract_episodes(trajectory)
    graph.schema_version = SCHEMA_VERSION
    for episode in episodes:
        first = episode.evidence_seq[0]
        nid = episode_id(episode.thread_id, episode.kind, first)
        graph.add_node(
            Node(
                id=nid,
                type="Episode",
                attrs={
                    "kind": episode.kind,
                    "thread_id": episode.thread_id,
                    "evidence": list(episode.evidence_seq),
                    "tools": list(episode.tool_sequence),
                    "weight": episode.weight,
                    "facts": dict(episode.facts),
                    "features_version": FEATURES_VERSION,
                },
            ),
        )
        owner = _case_for_seq(graph, first)
        if owner is not None:
            _link(graph, "HAS_EPISODE", owner, nid, kind=episode.kind)
    return episodes


def attach_fixed_by(graph: ExperienceGraph, episodes: list[Episode]) -> None:
    """Case- and step-level FIXED_BY. Does not change Case.r."""
    for episode in episodes:
        eid = episode_id(episode.thread_id, episode.kind, episode.evidence_seq[0])
        if episode.kind == "user_correction":
            after = episode.facts.get("run_id_after")
            before = episode.facts.get("run_id_before")
            dst = case_id(str(after)) if after else _case_for_seq(graph, episode.evidence_seq[0])
            src = case_id(str(before)) if before else (_previous_case(graph, dst) if dst else None)
            if src and dst and src in graph.nodes and dst in graph.nodes:
                _link(graph, "FIXED_BY", src, dst, via="user_correction", episode=eid)
            continue
        if episode.kind != "error_recovery":
            continue
        owner = _case_for_seq(graph, episode.evidence_seq[0])
        if owner is None:
            continue
        tool = str(episode.facts.get("tool") or "")
        failed = [
            step for step in _tool_steps(graph, owner)
            if step.attrs.get("name") == tool and step.attrs.get("status") == "error"
        ]
        ok = [
            step for step in _tool_steps(graph, owner)
            if step.attrs.get("name") == tool and step.attrs.get("status") == "ok"
        ]
        if failed and ok:
            _link(graph, "FIXED_BY", failed[0].id, ok[-1].id, via="error_recovery", episode=eid)
        prev = _previous_case(graph, owner)
        if prev and graph.cases.get(prev) and graph.cases[prev].r == "warning":
            _link(graph, "FIXED_BY", prev, owner, via="error_recovery", episode=eid)


def enrich_graph(graph: ExperienceGraph, trajectory: Trajectory) -> list[Episode]:
    """P1 attach on top of an L0 graph. Returns the detector episodes used."""
    episodes = attach_episodes(graph, trajectory)
    attach_fixed_by(graph, episodes)
    return episodes


def remember_thread(
    trajectory: Trajectory,
    *,
    working_dir: Path | None = None,
    state_dir: Path | None = None,
    outcome=None,
) -> ExperienceGraph:
    """Build L0+P1 for one already-loaded trajectory and persist the shard.

    Does not load other trajectories and does not rebuild the recipe overlay.
    """
    from msagent.exgraph.config import is_exgraph_enabled

    if not is_exgraph_enabled():
        raise RuntimeError("experience graph is disabled (MSAGENT_EXGRAPH_DISABLED)")
    from msagent.exgraph.cases import build_graph
    from msagent.exgraph.skills import attach_skill_docs
    from msagent.exgraph.sources import resolve_graph_dir
    from msagent.exgraph.store import save_graph

    graph = build_graph(trajectory, outcome=outcome)
    work = working_dir
    if work is None and trajectory.working_dir:
        work = Path(trajectory.working_dir)
    attach_skill_docs(graph, working_dir=work)
    enrich_graph(graph, trajectory)
    save_graph(graph, resolve_graph_dir(working_dir=work, state_dir=state_dir))
    return graph
