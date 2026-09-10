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


"""Read-only classify appendix: relations Skill Evolver does not already emit.

Does not load trajectories, does not change CROSS_SESSION_LIMIT, and does
not invent evidence seqs. Episode kinds are omitted on purpose — they
already appear as evolver bullets. Missing shards yield an empty string.
"""

from __future__ import annotations

from pathlib import Path

from msagent.exgraph.sources import resolve_graph_dir
from msagent.exgraph.store import find_saved_graph, load_graph
from msagent.exgraph.workspace import load_overlay

APPENDIX_CHAR_CAP = 1600


def _clip(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _first_error(sigma: dict) -> str:
    errors = sigma.get("errors") or []
    if not errors or not isinstance(errors, list):
        return ""
    err = errors[0] if isinstance(errors[0], dict) else {}
    tool = str(err.get("tool") or "")
    detail = _clip(str(err.get("error") or err.get("type") or ""), 80)
    if tool and detail:
        return f" err={tool}: {detail}"
    if detail:
        return f" err={detail}"
    return ""


def _cap_lines(lines: list[str], limit: int = APPENDIX_CHAR_CAP) -> str:
    kept: list[str] = []
    total = 0
    for line in lines:
        extra = len(line) + (1 if kept else 0)
        if total + extra > limit:
            break
        kept.append(line)
        total += extra
    return "\n".join(kept)


def render_thread_context(
    thread_id: str,
    *,
    working_dir: Path | None = None,
    state_dir: Path | None = None,
) -> str:
    """Markdown appendix for the classify bundle. Empty if nothing is stored."""
    try:
        from msagent.exgraph.config import is_exgraph_enabled

        if not is_exgraph_enabled():
            return ""
        root = resolve_graph_dir(working_dir=working_dir, state_dir=state_dir)
        saved = find_saved_graph(root, thread_id)
        if saved is None:
            return ""
        graph = load_graph(saved)
    except Exception:
        return ""

    lines = ["## Experience graph (stored)", ""]
    lines.append(f"Thread `{graph.thread_id}` agent `{graph.agent}`.")

    for record in graph.cases.values():
        tools = " → ".join(str(name) for name in (record.sigma.get("tool_path") or [])[:8])
        extra = f" tools={tools}" if tools else ""
        extra += _first_error(record.sigma)
        lines.append(f"- Case `{record.run_id}` outcome={record.r}{extra}")

    fixes = [edge for edge in graph.edges.values() if edge.type == "FIXED_BY"]
    if fixes:
        lines.append("Corrections:")
        for edge in fixes:
            lines.append(f"- {edge.src} fixed_by {edge.dst} via {edge.attrs.get('via')}")

    skills = [node for node in graph.nodes.values() if node.type == "SkillDoc"]
    if skills:
        lines.append("Skills:")
        for node in skills:
            name = node.attrs.get("name") or node.id
            status = node.attrs.get("status") or ""
            path = node.attrs.get("path") or ""
            suffix = f" `{path}`" if path else ""
            lines.append(f"- SkillDoc {status} {name}{suffix}")

    try:
        recipe_nodes, recipe_edges = load_overlay(root)
    except Exception:
        recipe_nodes, recipe_edges = [], []
    case_ids = set(graph.cases)
    by_id = {node.get("id"): node for node in recipe_nodes}
    similar = [
        edge
        for edge in recipe_edges
        if edge.get("type") == "SIMILAR_TO"
        and (edge.get("src") in case_ids or edge.get("dst") in case_ids)
    ]
    if similar:
        lines.append("Similar cases:")
        seen: set[str] = set()
        for edge in similar:
            other = edge.get("dst") if edge.get("src") in case_ids else edge.get("src")
            key = f"{edge.get('src')}->{edge.get('dst')}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- {edge.get('src')} similar_to {edge.get('dst')} "
                f"tools={edge.get('tools')} tokens={edge.get('tokens')}"
            )
    insights = [node for node in recipe_nodes if node.get("type") == "Insight"]
    if insights:
        lines.append("Insights:")
        for node in insights:
            ngram = ">".join(str(part) for part in (node.get("ngram") or []))
            skills = ",".join(str(name) for name in (node.get("skills") or [])[:3])
            lines.append(
                f"- {ngram or node.get('id')} support={node.get('support')} skills={skills}"
            )
    hits = [edge for edge in recipe_edges if edge.get("src") in case_ids and edge.get("type") == "INSTANTIATES"]
    if hits:
        lines.append("Recipes instantiated:")
        seen: set[str] = set()
        for edge in hits:
            node = by_id.get(edge.get("dst"))
            if not node:
                continue
            ngram = ">".join(str(part) for part in (node.get("ngram") or []))
            if not ngram or ngram in seen:
                continue
            seen.add(ngram)
            others = [
                tid
                for tid in (node.get("thread_ids") or [])
                if tid and tid != graph.thread_id
            ]
            also = f" also={','.join(others)}" if others else ""
            lines.append(f"- {ngram} support={node.get('support')}{also}")

    if len(lines) <= 3:
        return ""
    return _cap_lines(lines)


def render_graph_primary(
    thread_id: str,
    *,
    working_dir: Path | None = None,
    state_dir: Path | None = None,
) -> str:
    """Graph-first classify body: relations first, episode kinds without seqs.

    Used only when ``evidence_mode=graph``. Does not invent ``Evidence:`` or
    ``[evN]`` fragments. Episode kinds come from stored Episode nodes.
    """
    body = render_thread_context(
        thread_id, working_dir=working_dir, state_dir=state_dir
    )
    kinds: list[str] = []
    try:
        from msagent.exgraph.config import is_exgraph_enabled

        if is_exgraph_enabled():
            root = resolve_graph_dir(working_dir=working_dir, state_dir=state_dir)
            saved = find_saved_graph(root, thread_id)
            if saved is not None:
                graph = load_graph(saved)
                seen: set[str] = set()
                for node in graph.nodes.values():
                    if node.type != "Episode":
                        continue
                    kind = str(node.attrs.get("kind") or "")
                    if kind and kind not in seen:
                        seen.add(kind)
                        kinds.append(kind)
    except Exception:
        kinds = []
    header = [
        "## Experience graph (primary)",
        "",
        "Relations below are context, not citable evidence.",
    ]
    if body:
        rest = body.split("\n", 1)[-1] if body.startswith("## ") else body
        header.append(rest.lstrip("\n"))
    if kinds:
        header.append("Episode index (not citable):")
        header.extend(f"- {kind}" for kind in kinds)
    text = "\n".join(header).strip()
    if "Experience graph" not in text and not kinds:
        return ""
    return _cap_lines(text.split("\n"))
