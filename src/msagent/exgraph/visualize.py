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
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Self-contained HTML/JS picture of an experience graph growing over trajectories.

No new Python dependencies. The generated file is one HTML document with
inline CSS and JavaScript — open it in a browser, press Play.

Snapshots are the union of thread shards plus the workspace overlay after
each ingested trajectory. Nodes and edges that were not in the previous
snapshot are marked ``new`` so the picture can highlight growth.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from msagent.exgraph.schema import ExperienceGraph


VALUE_NOTES = {
    "FIXED_BY": "Correction relation Evolver detectors do not emit as an edge",
    "INSTANTIATES": "Case uses a cross-session recipe (shared tool n-gram)",
    "Recipe": "Only appears once two threads share a procedure",
    "SkillDoc": "Pointer at a proposal distilled from this thread",
    "Episode": "Detector kind stored on the case that owns it",
    "HAS_EPISODE": "Case owns this detector episode",
    "DERIVED_SKILL": "Thread produced a skill proposal",
}


def _clip(text: Any, limit: int = 72) -> str:
    if text is None:
        return ""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


def _node_label(node_id: str, ntype: str, attrs: dict[str, Any]) -> str:
    if ntype == "Thread":
        return f"{attrs.get('agent') or 'agent'} / {attrs.get('thread_id') or node_id.split(':', 1)[-1]}"
    if ntype == "TaskAnchor":
        return "task"
    if ntype == "Case":
        run = attrs.get("run_id") or node_id.split(":", 1)[-1]
        return f"{run} · {attrs.get('outcome') or attrs.get('r') or '?'}"
    if ntype == "Step":
        if attrs.get("kind") == "tool":
            status = attrs.get("status") or ""
            name = attrs.get("name") or "tool"
            return f"{name}" + (f" !{status}" if status == "error" else "")
        return "llm"
    if ntype == "Episode":
        return str(attrs.get("kind") or "episode")
    if ntype == "SkillDoc":
        return str(attrs.get("name") or node_id.split(":")[-1])
    if ntype == "Recipe":
        ngram = attrs.get("ngram") or []
        if isinstance(ngram, list) and ngram:
            return ">".join(str(part) for part in ngram)
        return node_id.split(":", 1)[-1]
    if ntype == "SubagentRun":
        return str(attrs.get("namespace") or "subagent")
    return node_id


def node_view(node_id: str, ntype: str, attrs: dict[str, Any] | None = None) -> dict[str, Any]:
    attrs = dict(attrs or {})
    thread = attrs.get("thread_id") or ""
    if not thread and node_id.startswith("thread:"):
        thread = node_id.split(":", 1)[-1]
    if not thread and node_id.startswith("task:thread:"):
        thread = node_id.split("task:thread:", 1)[-1]
    if not thread and node_id.startswith("episode:"):
        parts = node_id.split(":")
        if len(parts) >= 3:
            thread = parts[1]
    return {
        "id": node_id,
        "type": ntype,
        "label": _node_label(node_id, ntype, attrs),
        "thread": thread,
        "agent": attrs.get("agent") or "",
        "outcome": attrs.get("outcome") or attrs.get("r") or "",
        "kind": attrs.get("kind") or "",
        "status": attrs.get("status") or "",
        "detail": _clip(
            attrs.get("x")
            or attrs.get("name")
            or ">".join(str(p) for p in (attrs.get("ngram") or [])[:6])
            or attrs.get("path")
            or "",
            96,
        ),
    }


def edge_view(edge_id: str, etype: str, src: str, dst: str, attrs: dict[str, Any] | None = None) -> dict[str, Any]:
    attrs = dict(attrs or {})
    via = attrs.get("via") or ""
    label = etype if etype in {"FIXED_BY", "INSTANTIATES", "DERIVED_SKILL", "NEXT_CASE"} else ""
    if etype == "FIXED_BY" and via:
        label = f"FIXED_BY · {via}"
    return {"id": edge_id, "type": etype, "src": src, "dst": dst, "label": label, "via": via}


def _views_from_graph(graph: ExperienceGraph) -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    for node in graph.nodes.values():
        attrs = dict(node.attrs)
        attrs.setdefault("thread_id", graph.thread_id)
        attrs.setdefault("agent", graph.agent)
        if node.type == "Case" and node.id in graph.cases:
            record = graph.cases[node.id]
            attrs.setdefault("run_id", record.run_id)
            attrs.setdefault("outcome", record.r)
            attrs.setdefault("x", record.x)
        nodes.append(node_view(node.id, node.type, attrs))
    edges = [edge_view(edge.id, edge.type, edge.src, edge.dst, edge.attrs) for edge in graph.edges.values()]
    return nodes, edges


def _views_from_overlay(overlay_nodes: Iterable[dict], overlay_edges: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    nodes = []
    for row in overlay_nodes:
        nid = row.get("id")
        ntype = row.get("type") or "Recipe"
        if not isinstance(nid, str):
            continue
        attrs = {key: value for key, value in row.items() if key not in {"id", "type"}}
        view = node_view(nid, str(ntype), attrs)
        view["thread"] = view["thread"] or "_workspace"
        nodes.append(view)
    edges = []
    for row in overlay_edges:
        eid = row.get("id")
        etype = row.get("type")
        src = row.get("src")
        dst = row.get("dst")
        if not all(isinstance(value, str) and value for value in (eid, etype, src, dst)):
            continue
        attrs = {key: value for key, value in row.items() if key not in {"id", "type", "src", "dst"}}
        edges.append(edge_view(str(eid), str(etype), str(src), str(dst), attrs))
    return nodes, edges


def union_snapshot(
    graphs: list[ExperienceGraph],
    overlay_nodes: Iterable[dict] | None = None,
    overlay_edges: Iterable[dict] | None = None,
    *,
    step: int,
    title: str,
    subtitle: str = "",
    notes: list[str] | None = None,
    prev_node_ids: set[str] | None = None,
    prev_edge_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Full graph at this step, with ``new_*`` ids relative to the previous snapshot."""
    by_id: dict[str, dict] = {}
    edges_by_id: dict[str, dict] = {}
    for graph in graphs:
        nodes, edges = _views_from_graph(graph)
        for node in nodes:
            by_id[node["id"]] = node
        for edge in edges:
            edges_by_id[edge["id"]] = edge
    onodes, oedges = _views_from_overlay(overlay_nodes or [], overlay_edges or [])
    for node in onodes:
        by_id[node["id"]] = node
    for edge in oedges:
        edges_by_id[edge["id"]] = edge

    prev_n = prev_node_ids or set()
    prev_e = prev_edge_ids or set()
    new_nodes = [nid for nid in by_id if nid not in prev_n]
    new_edges = [eid for eid in edges_by_id if eid not in prev_e]
    new_types = sorted({by_id[nid]["type"] for nid in new_nodes})
    new_etypes = sorted({edges_by_id[eid]["type"] for eid in new_edges})
    auto_notes = list(notes or [])
    for token in new_types + new_etypes:
        hint = VALUE_NOTES.get(token)
        if hint and hint not in auto_notes:
            auto_notes.append(f"{token}: {hint}")
    return {
        "step": step,
        "title": title,
        "subtitle": subtitle,
        "notes": auto_notes,
        "new_node_ids": new_nodes,
        "new_edge_ids": new_edges,
        "new_types": new_types,
        "new_edge_types": new_etypes,
        "counts": {
            "nodes": len(by_id),
            "edges": len(edges_by_id),
            "new_nodes": len(new_nodes),
            "new_edges": len(new_edges),
            "threads": len(graphs),
        },
        "nodes": list(by_id.values()),
        "edges": list(edges_by_id.values()),
    }


def snapshots_after_each_trajectory(
    trajectories,
    *,
    working_dir: Path,
    state_dir: Path,
) -> list[dict[str, Any]]:
    """Ingest one trajectory at a time and snapshot the union graph + overlay."""
    from msagent.exgraph.enrich import remember_thread
    from msagent.exgraph.workspace import load_overlay, rebuild_overlay

    snapshots: list[dict[str, Any]] = []
    graphs: list[ExperienceGraph] = []
    prev_n: set[str] = set()
    prev_e: set[str] = set()
    pool = list(trajectories)
    root = Path(state_dir) / "exgraph"
    for index, trajectory in enumerate(pool, start=1):
        graphs.append(remember_thread(trajectory, working_dir=working_dir, state_dir=state_dir))
        overlay_n: list[dict] = []
        overlay_e: list[dict] = []
        if len(graphs) >= 2:
            rebuild_overlay(pool[:index], graphs, root)
            overlay_n, overlay_e = load_overlay(root)
        latest = graphs[-1]
        notes = [
            f"Ingested {latest.agent} / {latest.thread_id}",
            f"Shard cases: {len(latest.cases)}",
        ]
        if overlay_n:
            notes.append(f"Workspace recipes: {len(overlay_n)}")
        snap = union_snapshot(
            graphs,
            overlay_n,
            overlay_e,
            step=index,
            title=f"Trajectory {index} · {latest.agent} / {latest.thread_id}",
            subtitle=f"{len(latest.cases)} cases in this shard · union {len(graphs)} threads",
            notes=notes,
            prev_node_ids=prev_n,
            prev_edge_ids=prev_e,
        )
        snapshots.append(snap)
        prev_n = {node["id"] for node in snap["nodes"]}
        prev_e = {edge["id"] for edge in snap["edges"]}
    return snapshots


def render_growth_html(snapshots: list[dict[str, Any]], *, title: str = "Experience graph growth") -> str:
    payload = json.dumps(snapshots, ensure_ascii=False)
    safe_title = title.replace("<", "").replace(">", "")
    return _HTML.replace("__TITLE__", safe_title).replace("__SNAPSHOTS__", payload)


def write_growth_html(
    snapshots: list[dict[str, Any]],
    path: Path,
    *,
    title: str = "Experience graph growth",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_growth_html(snapshots, title=title), encoding="utf-8")
    return path


def demo_snapshots() -> list[dict[str, Any]]:
    """Schema-faithful picture of the intensive-test corpus (no recorder import).

    Live tests should prefer ``snapshots_after_each_trajectory``. This fallback
    lets a browser demo exist before the full msAgent tree is on PYTHONPATH.
    """
    from msagent.exgraph.schema import (
        Edge,
        ExperienceGraph,
        Node,
        case_id,
        edge_id,
        episode_id,
        recipe_id,
        skill_proposal_id,
        step_id,
        task_anchor_id,
        thread_id,
    )

    def _graph(thread: str, agent: str) -> ExperienceGraph:
        return ExperienceGraph(schema_version=2, thread_id=thread, agent=agent, source_path="")

    def _add(graph: ExperienceGraph, nid: str, ntype: str, **attrs: Any) -> None:
        graph.add_node(Node(id=nid, type=ntype, attrs=attrs))  # type: ignore[arg-type]

    def _edge(graph: ExperienceGraph, etype: str, src: str, dst: str, **attrs: Any) -> None:
        graph.add_edge(Edge(id=edge_id(etype, src, dst), type=etype, src=src, dst=dst, attrs=attrs))  # type: ignore[arg-type]

    sig = _graph("thread-signals", "Profiler")
    tid = thread_id("thread-signals")
    task = task_anchor_id("thread-signals")
    _add(sig, tid, "Thread", agent="Profiler", thread_id="thread-signals")
    _add(sig, task, "TaskAnchor", thread_id="thread-signals")
    _edge(sig, "HAS_TASK", tid, task)
    prev = None
    cases = [
        ("run-1", "warning", "Profile the training run and find the bottleneck", ["bash", "bash", "bash", "read_file"]),
        ("run-2", "unknown", "No, kernel-level profile, not summary", ["bash", "grep"]),
        ("run-3", "unknown", "continue", ["bash", "ls"]),
    ]
    span = {"run-1": ["s2", "s4", "s6", "s8"], "run-2": ["s11", "s13"], "run-3": ["s16", "s18"]}
    tools = {"s2": ("bash", "error"), "s4": ("bash", "ok"), "s6": ("bash", "ok"), "s8": ("read_file", "ok"),
             "s11": ("bash", "ok"), "s13": ("grep", "ok"), "s16": ("bash", "ok"), "s18": ("ls", "ok")}
    for run, outcome, x, path in cases:
        cid = case_id(run)
        _add(sig, cid, "Case", run_id=run, outcome=outcome, r=outcome, x=x, thread_id="thread-signals", agent="Profiler")
        from msagent.exgraph.schema import CaseRecord

        sig.add_case(CaseRecord(
            id=cid, thread_id="thread-signals", agent="Profiler", run_id=run,
            x=x, y=None, r=outcome, sigma={"tool_path": path}, evidence=[],
        ))
        _edge(sig, "CONTAINS", task, cid)
        if prev:
            _edge(sig, "NEXT_CASE", prev, cid)
        prev = cid
        for sid in span[run]:
            name, status = tools[sid]
            step = step_id(sid)
            _add(sig, step, "Step", kind="tool", name=name, status=status, span_id=sid, thread_id="thread-signals")
            _edge(sig, "HAS_STEP", cid, step, kind="tool")
    for kind, seq, owner in (
        ("error_recovery", 4, "case:run-1"),
        ("user_correction", 10, "case:run-2"),
        ("retry_loop", 4, "case:run-1"),
        ("approval_denied", 26, "case:run-3"),
    ):
        eid = episode_id("thread-signals", kind, seq)
        _add(sig, eid, "Episode", kind=kind, thread_id="thread-signals", weight=0.6)
        _edge(sig, "HAS_EPISODE", owner, eid, kind=kind)
    _edge(sig, "FIXED_BY", step_id("s2"), step_id("s6"), via="error_recovery")
    _edge(sig, "FIXED_BY", case_id("run-1"), case_id("run-2"), via="user_correction")
    skill = skill_proposal_id("thread-signals", "msprof-kernel-profile")
    _add(sig, skill, "SkillDoc", name="msprof-kernel-profile", thread_id="thread-signals")
    _edge(sig, "DERIVED_SKILL", tid, skill)

    reuse = _graph("thread-reuse", "Profiler")
    rid = thread_id("thread-reuse")
    rtask = task_anchor_id("thread-reuse")
    rcid = case_id("run-r1")
    _add(reuse, rid, "Thread", agent="Profiler", thread_id="thread-reuse")
    _add(reuse, rtask, "TaskAnchor", thread_id="thread-reuse")
    _add(reuse, rcid, "Case", run_id="run-r1", outcome="unknown", r="unknown",
         x="Profile the training run and find the kernel-level bottleneck",
         thread_id="thread-reuse", agent="Profiler")
    reuse.add_case(CaseRecord(
        id=rcid, thread_id="thread-reuse", agent="Profiler", run_id="run-r1",
        x="Profile the training run", y=None, r="unknown",
        sigma={"tool_path": ["bash", "read_file", "grep"]}, evidence=[],
    ))
    _edge(reuse, "HAS_TASK", rid, rtask)
    _edge(reuse, "CONTAINS", rtask, rcid)
    for sid, name in (("r2", "bash"), ("r4", "read_file"), ("r6", "grep")):
        step = step_id(sid)
        _add(reuse, step, "Step", kind="tool", name=name, status="ok", span_id=sid, thread_id="thread-reuse")
        _edge(reuse, "HAS_STEP", rcid, step, kind="tool")

    acc = _graph("thread-accuracy", "Accuracy")
    aid = thread_id("thread-accuracy")
    atask = task_anchor_id("thread-accuracy")
    acid = case_id("run-a1")
    _add(acc, aid, "Thread", agent="Accuracy", thread_id="thread-accuracy")
    _add(acc, atask, "TaskAnchor", thread_id="thread-accuracy")
    _add(acc, acid, "Case", run_id="run-a1", outcome="unknown", r="unknown",
         x="Loss became NaN after step 1200", thread_id="thread-accuracy", agent="Accuracy")
    acc.add_case(CaseRecord(
        id=acid, thread_id="thread-accuracy", agent="Accuracy", run_id="run-a1",
        x="Loss became NaN", y=None, r="unknown",
        sigma={"tool_path": ["read_file", "grep"]}, evidence=[],
    ))
    _edge(acc, "HAS_TASK", aid, atask)
    _edge(acc, "CONTAINS", atask, acid)
    for sid, name in (("a2", "read_file"), ("a4", "grep")):
        step = step_id(sid)
        _add(acc, step, "Step", kind="tool", name=name, status="ok", span_id=sid, thread_id="thread-accuracy")
        _edge(acc, "HAS_STEP", acid, step, kind="tool")

    recipe = recipe_id(["bash", "read_file"])
    overlay_nodes = [{
        "id": recipe, "type": "Recipe", "ngram": ["bash", "read_file"],
        "support": 2, "thread_ids": ["thread-signals", "thread-reuse"],
    }]
    overlay_edges = [
        {"id": edge_id("INSTANTIATES", case_id("run-1"), recipe), "type": "INSTANTIATES",
         "src": case_id("run-1"), "dst": recipe},
        {"id": edge_id("INSTANTIATES", rcid, recipe), "type": "INSTANTIATES",
         "src": rcid, "dst": recipe},
    ]

    s1 = union_snapshot(
        [sig], None, None, step=1,
        title="Trajectory 1 · Profiler / thread-signals",
        subtitle="Failing flags, user correction, skill proposal",
        notes=["First shard: cases + FIXED_BY + SkillDoc"],
    )
    s2 = union_snapshot(
        [sig, reuse], overlay_nodes, overlay_edges, step=2,
        title="Trajectory 2 · Profiler / thread-reuse",
        subtitle="Successful first-try session shares bash>read_file",
        notes=["Recipe appears only when two Profiler threads share an n-gram"],
        prev_node_ids={n["id"] for n in s1["nodes"]},
        prev_edge_ids={e["id"] for e in s1["edges"]},
    )
    s3 = union_snapshot(
        [sig, reuse, acc], overlay_nodes, overlay_edges, step=3,
        title="Trajectory 3 · Accuracy / thread-accuracy",
        subtitle="Control thread — recipes must not leak",
        notes=["Accuracy cluster added; INSTANTIATES still only Profiler cases"],
        prev_node_ids={n["id"] for n in s2["nodes"]},
        prev_edge_ids={e["id"] for e in s2["edges"]},
    )
    return [s1, s2, s3]


def default_html_path() -> Path:
    override = os.environ.get("EXGRAPH_GROWTH_HTML")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    # src/msagent/exgraph/visualize.py → repo root
    repo = here.parents[3]
    artifacts = repo / "artifacts"
    if artifacts.is_dir():
        return artifacts / "exgraph_growth_demo.html"
    return Path.cwd() / "exgraph_growth_demo.html"


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m msagent.exgraph.visualize --fixtures DIR -o out.html"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="msagent.exgraph.visualize")
    parser.add_argument("--fixtures", default=None, help="Directory of trajectory JSONL files")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--working-dir", default=None)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use schema-faithful demo_snapshots (also EXGRAPH_VIZ_DEMO=1)",
    )
    args = parser.parse_args(argv)

    demo = args.demo or os.environ.get("EXGRAPH_VIZ_DEMO", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    dest_early = Path(args.output) if args.output else default_html_path()
    if demo:
        snapshots = demo_snapshots()
        write_growth_html(snapshots, dest_early)
        print(f"Wrote {dest_early} ({len(snapshots)} demo steps)")
        return 0

    from msagent.trajectory_recorder.reader import load_trajectory

    fixtures = Path(args.fixtures) if args.fixtures else Path("tests/fixtures/trajectories")
    names = [
        "skill_evolver_signals.jsonl",
        "exgraph_reuse.jsonl",
        "exgraph_accuracy.jsonl",
    ]
    paths = [fixtures / name for name in names if (fixtures / name).is_file()]
    if not paths:
        paths = sorted(fixtures.glob("*.jsonl"))
    if not paths:
        print(f"No trajectory JSONL in {fixtures}", file=sys.stderr)
        return 1
    trajectories = [load_trajectory(path) for path in paths]
    work = Path(args.working_dir) if args.working_dir else Path.cwd() / ".viz-work"
    state = Path(args.state_dir) if args.state_dir else Path.cwd() / ".viz-state"
    work.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    snapshots = snapshots_after_each_trajectory(trajectories, working_dir=work, state_dir=state)
    dest = Path(args.output) if args.output else default_html_path()
    write_growth_html(snapshots, dest)
    print(f"Wrote {dest} ({len(snapshots)} growth steps)")
    return 0


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0b1020;
    --panel: #141b2e;
    --ink: #e7edf7;
    --muted: #8b97ad;
    --line: #243049;
    --new: #ffd166;
    --accent: #5b8def;
    --warn: #e8a54b;
    --ok: #3dcf8e;
    --pink: #e36bae;
    --recipe: #ff7a45;
    --skill: #f0c14a;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
    font: 14px/1.45 "Segoe UI", system-ui, sans-serif; }
  .app { display: grid; grid-template-rows: auto 1fr auto; height: 100%; }
  header { display: flex; align-items: center; gap: 16px; padding: 12px 18px;
    border-bottom: 1px solid var(--line); background: #0e1528; }
  header h1 { font-size: 16px; margin: 0; letter-spacing: .02em; font-weight: 600; }
  header .sub { color: var(--muted); font-size: 12px; }
  .controls { margin-left: auto; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button, label.chk { background: #1c2740; color: var(--ink); border: 1px solid #2c3a5a;
    border-radius: 8px; padding: 6px 12px; cursor: pointer; font: inherit; }
  button.primary { background: #2a4d8f; border-color: #3d6cc0; }
  button:hover { filter: brightness(1.12); }
  .stage { display: grid; grid-template-columns: 1fr 320px; min-height: 0; }
  svg { width: 100%; height: 100%; background:
    radial-gradient(1200px 500px at 20% -10%, #15244a 0%, transparent 55%), var(--bg); }
  .side { border-left: 1px solid var(--line); background: var(--panel); padding: 14px 16px;
    overflow: auto; }
  .side h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em;
    color: var(--muted); margin: 0 0 8px; }
  .kpis { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
  .kpi { background: #0f1628; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  .kpi b { display: block; font-size: 18px; }
  .kpi span { color: var(--muted); font-size: 11px; }
  .note { background: #1a2236; border-left: 3px solid var(--new); padding: 8px 10px;
    margin: 0 0 8px; border-radius: 0 8px 8px 0; font-size: 13px; }
  .note.newish { border-left-color: var(--recipe); }
  .legend { display: flex; flex-wrap: wrap; gap: 8px 12px; font-size: 12px; color: var(--muted); }
  .sw { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }
  footer { padding: 8px 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px;
    display: flex; justify-content: space-between; }
  .steps { display: flex; gap: 6px; }
  .pill { padding: 4px 8px; border-radius: 999px; border: 1px solid var(--line); cursor: pointer; }
  .pill.on { background: #24345c; color: var(--new); border-color: #6b5420; }
  text { font-family: inherit; }
  .hint { fill: var(--muted); font-size: 11px; }
</style>
</head>
<body>
<div class="app">
  <header>
    <div>
      <h1 id="title">Experience graph growth</h1>
      <div class="sub" id="subtitle">Ingest trajectories one by one. Gold ring = appeared on this step.</div>
    </div>
    <div class="controls">
      <button id="reset">Reset</button>
      <button id="prev">Prev</button>
      <button id="play" class="primary">Play</button>
      <button id="next">Next</button>
      <label class="chk"><input type="checkbox" id="hideLlm" checked/> Hide LLM steps</label>
    </div>
  </header>
  <div class="stage">
    <svg id="canvas" xmlns="http://www.w3.org/2000/svg"></svg>
    <aside class="side">
      <h2>This step added</h2>
      <div class="kpis">
        <div class="kpi"><b id="kNodes">0</b><span>new nodes</span></div>
        <div class="kpi"><b id="kEdges">0</b><span>new edges</span></div>
        <div class="kpi"><b id="kAllN">0</b><span>nodes in union</span></div>
        <div class="kpi"><b id="kAllE">0</b><span>edges in union</span></div>
      </div>
      <div id="notes"></div>
      <h2>Why this is value for Evolver</h2>
      <div class="note">Unit tests prove nodes exist. This picture shows the graph <em>growing</em> a relation Evolver cannot see from episode bullets alone.</div>
      <div class="note newish">Step 1 — first Profiler session: cases, tool path, FIXED_BY, optional SkillDoc.</div>
      <div class="note newish">Step 2 — second Profiler session: Recipe + INSTANTIATES (shared n-gram).</div>
      <div class="note newish">Step 3 — Accuracy control: its own cluster. Recipes must not leak onto it.</div>
      <h2>Legend</h2>
      <div class="legend">
        <span><i class="sw" style="background:#5b8def"></i>Thread</span>
        <span><i class="sw" style="background:#7c6af7"></i>Task</span>
        <span><i class="sw" style="background:#e8a54b"></i>Case warning</span>
        <span><i class="sw" style="background:#8aa0b8"></i>Case unknown</span>
        <span><i class="sw" style="background:#4aa3a8"></i>Step</span>
        <span><i class="sw" style="background:#e36bae"></i>Episode</span>
        <span><i class="sw" style="background:#f0c14a"></i>SkillDoc</span>
        <span><i class="sw" style="background:#ff7a45"></i>Recipe</span>
      </div>
    </aside>
  </div>
  <footer>
    <div id="foot">msAgent experience graph · schema v2 · no LLM in this picture</div>
    <div class="steps" id="pills"></div>
  </footer>
</div>
<script>
const SNAPSHOTS = __SNAPSHOTS__;
const COLORS = {
  Thread: "#5b8def",
  TaskAnchor: "#7c6af7",
  Case: "#8aa0b8",
  Step: "#4aa3a8",
  Episode: "#e36bae",
  SkillDoc: "#f0c14a",
  Recipe: "#ff7a45",
  SubagentRun: "#9b8c7a"
};
const EDGE_COLOR = {
  FIXED_BY: "#ffd166",
  INSTANTIATES: "#ff7a45",
  HAS_EPISODE: "#e36bae",
  DERIVED_SKILL: "#f0c14a",
  NEXT_CASE: "#c5d0e6",
  CONTAINS: "#3d4d6e",
  HAS_TASK: "#3d4d6e",
  HAS_STEP: "#2e3d5c",
  PARENT_OF: "#2e3d5c",
  DELEGATES: "#6b7c98",
  IN_SUBAGENT: "#6b7c98"
};

let idx = 0;
let timer = null;
const svg = document.getElementById("canvas");

function caseColor(node) {
  if (node.type !== "Case") return COLORS[node.type] || "#88a";
  if (node.outcome === "warning") return "#e8a54b";
  if (node.outcome === "golden") return "#3dcf8e";
  return "#8aa0b8";
}

function visibleNodes(snap) {
  const hideLlm = document.getElementById("hideLlm").checked;
  return snap.nodes.filter(n => !(hideLlm && n.type === "Step" && n.kind === "llm"));
}
function visibleEdges(snap, nodes) {
  const ids = new Set(nodes.map(n => n.id));
  return snap.edges.filter(e => ids.has(e.src) && ids.has(e.dst));
}

function layout(nodes) {
  const threads = [];
  const seen = new Set();
  for (const n of nodes) {
    const t = n.thread || (n.type === "Recipe" ? "_workspace" : "_");
    if (!seen.has(t) && t !== "_workspace") { seen.add(t); threads.push(t); }
  }
  if (!threads.length) threads.push("_");
  const w = svg.clientWidth || 900;
  const h = svg.clientHeight || 640;
  const colW = Math.max(240, (w - 80) / Math.max(threads.length, 1));
  const pos = {};
  const byThread = {};
  for (const n of nodes) {
    const t = n.type === "Recipe" ? "_workspace" : (n.thread || threads[0]);
    (byThread[t] ||= []).push(n);
  }
  threads.forEach((t, ti) => {
    const x = 70 + ti * colW + colW * 0.38;
    const group = byThread[t] || [];
    const threadN = group.find(n => n.type === "Thread");
    const taskN = group.find(n => n.type === "TaskAnchor");
    const cases = group.filter(n => n.type === "Case");
    const skills = group.filter(n => n.type === "SkillDoc");
    if (threadN) pos[threadN.id] = {x, y: 48};
    if (taskN) pos[taskN.id] = {x, y: 108};
    cases.forEach((c, ci) => {
      const y = 190 + ci * 168;
      pos[c.id] = {x, y};
      const steps = group.filter(n => n.type === "Step" && snapOwns(n, c, nodes));
      // fallback: steps whose id appears on HAS_STEP later; approximate by listing all unmatched steps
    });
  });
  // Second pass: attach steps / episodes to nearest case via edges stored on window.currentEdges
  const edges = window.currentEdges || [];
  const caseOf = {};
  for (const e of edges) {
    if (e.type === "HAS_STEP" || e.type === "HAS_EPISODE") caseOf[e.dst] = e.src;
    if (e.type === "CONTAINS") caseOf[e.dst] = e.dst;
  }
  threads.forEach((t, ti) => {
    const x = 70 + ti * colW + colW * 0.38;
    const group = byThread[t] || [];
    const cases = group.filter(n => n.type === "Case");
    const caseY = {};
    cases.forEach((c, ci) => { caseY[c.id] = 190 + ci * 168; pos[c.id] = {x, y: caseY[c.id]}; });
    const buckets = {};
    for (const n of group) {
      if (n.type === "Step" || n.type === "Episode") {
        const owner = caseOf[n.id];
        (buckets[owner || "_"] ||= []).push(n);
      }
    }
    Object.entries(buckets).forEach(([owner, list]) => {
      const base = pos[owner] || {x, y: 190};
      const steps = list.filter(n => n.type === "Step");
      const eps = list.filter(n => n.type === "Episode");
      steps.forEach((n, i) => { pos[n.id] = {x: base.x + 128, y: base.y - 24 + i * 26}; });
      eps.forEach((n, i) => { pos[n.id] = {x: base.x - 128, y: base.y - 24 + i * 26}; });
    });
    group.filter(n => n.type === "SkillDoc").forEach((n, i) => {
      pos[n.id] = {x, y: 190 + cases.length * 168 + 20 + i * 36};
    });
    group.filter(n => n.type === "SubagentRun").forEach((n, i) => {
      pos[n.id] = {x: x + 200, y: 108 + i * 28};
    });
  });
  const recipes = nodes.filter(n => n.type === "Recipe");
  recipes.forEach((n, i) => {
    const x = (w / 2) - ((recipes.length - 1) * 90) / 2 + i * 180;
    pos[n.id] = {x, y: Math.max(h - 70, 520)};
  });
  for (const n of nodes) if (!pos[n.id]) pos[n.id] = {x: 40, y: 40};
  return pos;
}

function snapOwns() { return true; }

function draw() {
  const snap = SNAPSHOTS[idx];
  if (!snap) return;
  document.getElementById("title").textContent = snap.title;
  document.getElementById("subtitle").textContent = snap.subtitle || "";
  document.getElementById("kNodes").textContent = snap.counts.new_nodes;
  document.getElementById("kEdges").textContent = snap.counts.new_edges;
  document.getElementById("kAllN").textContent = snap.counts.nodes;
  document.getElementById("kAllE").textContent = snap.counts.edges;
  const notes = document.getElementById("notes");
  notes.innerHTML = "";
  (snap.notes || []).forEach(text => {
    const d = document.createElement("div");
    d.className = "note";
    d.textContent = text;
    notes.appendChild(d);
  });
  const newN = new Set(snap.new_node_ids || []);
  const newE = new Set(snap.new_edge_ids || []);
  const nodes = visibleNodes(snap);
  const edges = visibleEdges(snap, nodes);
  window.currentEdges = snap.edges;
  const pos = layout(nodes);
  const w = svg.clientWidth || 900;
  const h = svg.clientHeight || 640;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = "";

  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = `<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9aa8c3"/></marker>`;
  svg.appendChild(defs);

  // thread columns captions
  const threads = [];
  const seenT = new Set();
  nodes.forEach(n => {
    if (n.thread && n.thread !== "_workspace" && !seenT.has(n.thread)) {
      seenT.add(n.thread); threads.push({id: n.thread, agent: n.agent || ""});
    }
  });

  edges.forEach(e => {
    const a = pos[e.src], b = pos[e.dst];
    if (!a || !b) return;
    const isNew = newE.has(e.id);
    const color = EDGE_COLOR[e.type] || "#33405c";
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const midX = (a.x + b.x) / 2;
    const midY = (a.y + b.y) / 2 - (e.type === "FIXED_BY" || e.type === "INSTANTIATES" ? 28 : 0);
    path.setAttribute("d", `M ${a.x} ${a.y} Q ${midX} ${midY} ${b.x} ${b.y}`);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", color);
    path.setAttribute("stroke-width", isNew ? 2.4 : (["FIXED_BY","INSTANTIATES"].includes(e.type) ? 2 : 1));
    path.setAttribute("stroke-dasharray", e.type === "FIXED_BY" ? "6 4" : (isNew ? "4 3" : "0"));
    path.setAttribute("opacity", isNew ? 1 : 0.55);
    path.setAttribute("marker-end", "url(#arrow)");
    svg.appendChild(path);
    if (e.label && ["FIXED_BY","INSTANTIATES","DERIVED_SKILL"].includes(e.type)) {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", midX);
      label.setAttribute("y", midY - 6);
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("fill", color);
      label.setAttribute("font-size", "10");
      label.textContent = e.label;
      svg.appendChild(label);
    }
  });

  nodes.forEach(n => {
    const p = pos[n.id];
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const isNew = newN.has(n.id);
    const fill = caseColor(n);
    if (isNew) {
      const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      ring.setAttribute("cx", p.x); ring.setAttribute("cy", p.y);
      ring.setAttribute("r", n.type === "Case" ? 22 : 16);
      ring.setAttribute("fill", "none");
      ring.setAttribute("stroke", "#ffd166");
      ring.setAttribute("stroke-width", "2");
      ring.setAttribute("opacity", "0.95");
      g.appendChild(ring);
    }
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", p.x); c.setAttribute("cy", p.y);
    c.setAttribute("r", n.type === "Thread" ? 14 : n.type === "Recipe" ? 13 : n.type === "Case" ? 12 : 9);
    c.setAttribute("fill", fill);
    c.setAttribute("stroke", isNew ? "#fff4c4" : "#0b1020");
    c.setAttribute("stroke-width", isNew ? 2 : 1);
    g.appendChild(c);
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", p.x + 16);
    t.setAttribute("y", p.y + 4);
    t.setAttribute("fill", isNew ? "#ffe9a8" : "#d5def0");
    t.setAttribute("font-size", "11");
    t.setAttribute("font-weight", isNew ? "700" : "500");
    t.textContent = (isNew ? "NEW · " : "") + n.label;
    g.appendChild(t);
    svg.appendChild(g);
  });

  const pills = document.getElementById("pills");
  pills.innerHTML = "";
  SNAPSHOTS.forEach((s, i) => {
    const el = document.createElement("div");
    el.className = "pill" + (i === idx ? " on" : "");
    el.textContent = (i + 1) + " · " + (s.title.split("·")[1] || s.title).trim().slice(0, 28);
    el.onclick = () => { idx = i; stop(); draw(); };
    pills.appendChild(el);
  });
}

function stop() {
  if (timer) { clearInterval(timer); timer = null; }
  document.getElementById("play").textContent = "Play";
}
function play() {
  if (timer) { stop(); return; }
  document.getElementById("play").textContent = "Pause";
  timer = setInterval(() => {
    if (idx >= SNAPSHOTS.length - 1) { stop(); return; }
    idx += 1; draw();
  }, 1600);
}

document.getElementById("play").onclick = play;
document.getElementById("next").onclick = () => { stop(); idx = Math.min(SNAPSHOTS.length - 1, idx + 1); draw(); };
document.getElementById("prev").onclick = () => { stop(); idx = Math.max(0, idx - 1); draw(); };
document.getElementById("reset").onclick = () => { stop(); idx = 0; draw(); };
document.getElementById("hideLlm").onchange = draw;
window.addEventListener("resize", draw);
draw();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
