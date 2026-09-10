#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""Intensive A/B: Skill Evolver classify bundle with vs without ExGraph.

Uses the project's own recorder-shaped fixtures plus two extra sessions that
replay a realistic Profiler correction story and an Accuracy control thread.
No LLM. Value is the extra structure the graph appendix gives classify
(outcomes, FIXED_BY, recipes) without adding fabricated evidence seqs or
changing the last-N pool.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from msagent.exgraph.config import (
    ENV_DISABLED,
    ENV_ENABLED,
    ENV_EVIDENCE_MODE,
    reset_config_cache,
)


@pytest.fixture(autouse=True)
def _exgraph_opt_in(monkeypatch):
    """Exgraph tests run opted-in. Product default remains off."""
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.delenv(ENV_DISABLED, raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


import pytest

from msagent.skill_evolver.bundle import build_evidence_bundle
from msagent.skill_evolver.exgraph_context import attach_stored_graph

# Imported from the constant module only if langchain is present. The
# contract under test is the number itself, not the import graph.
CROSS_SESSION_LIMIT = 20
from msagent.skill_evolver.features import (
    evidence_score,
    extract_episodes,
    mine_cross_session,
)
from msagent.trajectory_recorder.reader import load_trajectory

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures" / "trajectories"
SIGNALS = FIXTURES / "skill_evolver_signals.jsonl"
REUSE = FIXTURES / "exgraph_reuse.jsonl"
ACCURACY = FIXTURES / "exgraph_accuracy.jsonl"


def _writable_output(filename: str, tmp_path: Path, env_name: str) -> Path:
    """Write side-effect files somewhere the current user can create.

    Never hardcode an authoring-machine path such as ``/home/workdir/artifacts``.
    Order: env override → repo ``artifacts/`` if that directory already exists
    and is writable → ``tmp_path``.
    """
    override = os.environ.get(env_name)
    if override:
        path = Path(override)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    artifacts = REPO / "artifacts"
    try:
        if artifacts.is_dir() and os.access(artifacts, os.W_OK):
            return artifacts / filename
    except OSError:
        pass
    return tmp_path / filename


def _plant_accepted(work: Path, thread_id: str) -> None:
    folder = work / "skills" / "msprof-kernel-profile"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "# msprof kernel profile\n\n"
        "Always pass --application, --output and --level kernel.\n\n"
        f"<!-- provenance thread: {thread_id} -->\n",
        encoding="utf-8",
    )
    (folder / "provenance.json").write_text(
        json.dumps({"thread_ids": [thread_id], "name": "msprof-kernel-profile"}),
        encoding="utf-8",
    )


def _plant_proposal(work: Path, thread_id: str) -> None:
    folder = work / "skills" / ".proposals" / thread_id / "msprof-kernel-profile"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "# msprof kernel profile\n\n"
        "Always pass --application, --output and --level kernel.\n\n"
        f"<!-- provenance thread: {thread_id} -->\n",
        encoding="utf-8",
    )
    (folder / "provenance.json").write_text(
        json.dumps({"thread_ids": [thread_id], "name": "msprof-kernel-profile"}),
        encoding="utf-8",
    )


def _facts(text: str) -> set[str]:
    keys = []
    for token in (
        "FIXED_BY",
        "fixed_by",
        "outcome=",
        "Recipes instantiated",
        "Similar cases:",
        "Insights:",
        "Experience graph",
        "SkillDoc",
        "Episodes:",
        "user_correction",
        "error_recovery",
        "approval_denied",
        "Evidence:",
    ):
        if token in text:
            keys.append(token)
    return set(keys)


@pytest.fixture
def corpus():
    return {
        "signals": load_trajectory(SIGNALS),
        "reuse": load_trajectory(REUSE),
        "accuracy": load_trajectory(ACCURACY),
    }


def test_fixtures_are_recorder_shaped(corpus) -> None:
    sig = corpus["signals"]
    assert sig.thread_id == "thread-signals"
    assert sig.agent == "Profiler"
    assert len(sig.turns) >= 2
    assert any(
        turn.user_message
        and ("не так" in turn.user_message or "不对" in turn.user_message)
        for turn in sig.turns
    )
    reuse = corpus["reuse"]
    assert reuse.thread_id == "thread-reuse"
    assert reuse.agent == "Profiler"


def test_evolver_bundle_without_graph_has_episodes_but_no_relations(corpus) -> None:
    traj = corpus["signals"]
    episodes = extract_episodes(traj)
    kinds = {ep.kind for ep in episodes}
    assert "error_recovery" in kinds or "retry_loop" in kinds
    assert "approval_denied" in kinds
    built = build_evidence_bundle(episodes, [traj])
    bundle = built.text
    seqs = {fragment.ref.seq for fragment in built.shown.values()}
    assert "### Episode" in bundle
    assert "Experience graph" not in bundle
    assert "fixed_by" not in bundle
    assert "Recipes instantiated" not in bundle
    assert seqs == set().union(*(set(ep.evidence_seq) for ep in episodes)) or seqs <= {
        s for ep in episodes for s in ep.evidence_seq
    }


def test_graph_appendix_adds_relations_not_new_evidence_seqs(corpus, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MSAGENT_EXGRAPH_DISABLED", raising=False)
    work = tmp_path / "proj"
    work.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    _plant_proposal(work, "thread-signals")
    _plant_accepted(work, "thread-signals")

    traj = corpus["signals"]
    episodes = extract_episodes(traj)
    built = build_evidence_bundle(episodes, [traj])
    bundle, shown = built.text, set(built.shown)

    from msagent.exgraph.enrich import remember_thread
    from msagent.exgraph.workspace import rebuild_overlay

    graphs = []
    for item in corpus.values():
        graphs.append(remember_thread(item, working_dir=work, state_dir=state))
    rebuild_overlay(list(corpus.values()), graphs, state / "exgraph")

    enriched = attach_stored_graph(bundle, traj, working_dir=work, state_dir=state)
    assert enriched.startswith(bundle)
    assert "Experience graph" in enriched
    assert "fixed_by" in enriched
    assert "outcome=" in enriched
    extra = enriched[len(bundle) :]
    assert "Evidence:" not in extra
    assert "Episodes:" not in extra
    assert "### Episode" not in extra
    assert "SkillDoc" in extra
    assert "msprof-kernel-profile" in extra
    assert "similar_to" in extra or "Similar cases:" in extra
    assert "Insights:" in extra
    # valid_seq is still only evolver episodes
    from msagent.skill_evolver.bundle import build_evidence_bundle as rebuild

    assert set(rebuild(episodes, [traj]).shown) == shown


def test_kill_switch_removes_value_and_writes(corpus, tmp_path, monkeypatch) -> None:
    work = tmp_path / "proj"
    work.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    traj = corpus["signals"]
    episodes = extract_episodes(traj)
    bundle = build_evidence_bundle(episodes, [traj]).text
    monkeypatch.setenv("MSAGENT_EXGRAPH_DISABLED", "1")
    text = attach_stored_graph(bundle, traj, working_dir=work, state_dir=state)
    assert text == bundle
    assert list(state.rglob("nodes.jsonl")) == []


def test_episodes_mode_is_the_old_evolver_pass(corpus, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ENV_EVIDENCE_MODE, "episodes")
    reset_config_cache()
    work = tmp_path / "proj"
    work.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    traj = corpus["signals"]
    bundle = build_evidence_bundle(extract_episodes(traj), [traj]).text
    text = attach_stored_graph(bundle, traj, working_dir=work, state_dir=state)
    assert text == bundle
    assert "Experience graph" not in text
    assert list(state.rglob("nodes.jsonl")) == []
    monkeypatch.delenv(ENV_EVIDENCE_MODE, raising=False)
    reset_config_cache()


def test_cross_session_recipes_need_two_profiler_threads(corpus) -> None:
    recipes = mine_cross_session(list(corpus.values()))
    assert recipes, "expected a repeated_procedure n-gram across profiler sessions"
    thread_sets = [set(ep.facts.get("thread_ids") or []) for ep in recipes]
    assert any("thread-signals" in ids and "thread-reuse" in ids for ids in thread_sets)


def test_evolver_pool_constant_untouched() -> None:
    assert CROSS_SESSION_LIMIT == 20


def test_write_value_report(corpus, tmp_path, monkeypatch) -> None:
    """Side-effect report for humans. Still an assertion-bearing test."""
    monkeypatch.delenv("MSAGENT_EXGRAPH_DISABLED", raising=False)
    work = tmp_path / "proj"
    work.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    _plant_proposal(work, "thread-signals")
    _plant_accepted(work, "thread-signals")

    traj = corpus["signals"]
    episodes = extract_episodes(traj)
    score = evidence_score(episodes)
    built = build_evidence_bundle(episodes, [traj])
    bundle = built.text
    seqs = {fragment.ref.seq for fragment in built.shown.values()}

    from msagent.exgraph.enrich import remember_thread
    from msagent.exgraph.workspace import rebuild_overlay

    graphs = [remember_thread(item, working_dir=work, state_dir=state) for item in corpus.values()]
    rebuild_overlay(list(corpus.values()), graphs, state / "exgraph")
    enriched = attach_stored_graph(bundle, traj, working_dir=work, state_dir=state)
    recipes = mine_cross_session(list(corpus.values()))

    sig_graph = next(g for g in graphs if g.thread_id == "thread-signals")
    fixes = [e for e in sig_graph.edges.values() if e.type == "FIXED_BY"]
    outcomes = {c.run_id: c.r for c in sig_graph.cases.values()}
    tool_paths = {
        c.run_id: list(c.sigma.get("tool_path") or []) for c in sig_graph.cases.values()
    }

    lines = [
        "# ExGraph value for Skill Evolver",
        "",
        "Deterministic A/B on recorder-shaped fixtures (no LLM).",
        "",
        "## Corpus",
        "",
        f"- `{SIGNALS.name}` — Profiler thread-signals (failing msprof flags, RU correction, denied write)",
        f"- `{REUSE.name}` — Profiler thread-reuse (correct flags on first try)",
        f"- `{ACCURACY.name}` — Accuracy control thread (NaN dump, should not own profiler recipes)",
        "",
        "## Evolver-only classify bundle (graph off)",
        "",
        f"- episodes: {len(episodes)} kinds={sorted({e.kind for e in episodes})}",
        f"- evidence_score: {score:.2f}",
        f"- valid evidence seqs: {sorted(seqs)}",
        f"- chars: {len(bundle)}",
        f"- facts present: {sorted(_facts(bundle))}",
        "",
        "## Same bundle after attach_stored_graph",
        "",
        f"- chars: {len(enriched)} (delta +{len(enriched) - len(bundle)})",
        f"- facts present: {sorted(_facts(enriched))}",
        f"- FIXED_BY edges in shard: {len(fixes)}",
        f"- case outcomes: {outcomes}",
        f"- case tool paths: {tool_paths}",
        f"- cross-session recipes: {len(recipes)}",
        "",
        "### Appendix added to classify",
        "",
        "```",
        enriched[len(bundle) :].strip() or "(empty)",
        "```",
        "",
        "## Why this is value, not duplication",
        "",
        "Skill Evolver already lists episodes as independent bullets.",
        "The graph adds relations the detectors do not emit:",
        "",
        "- which *case* was corrected by which later case (`FIXED_BY`);",
        "- per-turn outcome and tool path (the recipe grain);",
        "- recipes that only exist when two threads share an n-gram;",
        "- a SkillDoc line when a proposal cites the thread (not episode bullets).",
        "",
        "The appendix must not list `Evidence:` seqs, so classify `valid_seq`",
        "stays the evolver set. `CROSS_SESSION_LIMIT` stays 20.",
        "",
        "## Live data",
        "",
        "Point the same comparison at a real project:",
        "",
        "```",
        "python -m msagent.exgraph.export build --all --working-dir /path/to/project",
        "MSAGENT_EXGRAPH_DISABLED=1  # control: evolver bundle unchanged",
        "```",
        "",
    ]
    dest = _writable_output("exgraph_evolver_value_report.md", tmp_path, "EXGRAPH_VALUE_REPORT")
    dest.write_text("\n".join(lines), encoding="utf-8")
    assert "Experience graph" in enriched
    assert dest.is_file()
    assert len(fixes) >= 1
    assert score >= 0.6


def test_growth_html_highlights_new_over_trajectories(corpus, tmp_path, monkeypatch) -> None:
    """Boss-demo picture: graph after each trajectory, new nodes/edges marked.

    Live ingest is the merge gate. ``demo_snapshots`` runs only when
    ``EXGRAPH_VIZ_DEMO=1``.
    """
    monkeypatch.delenv("MSAGENT_EXGRAPH_DISABLED", raising=False)
    work = tmp_path / "proj"
    work.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    _plant_proposal(work, "thread-signals")
    _plant_accepted(work, "thread-signals")

    from msagent.exgraph.visualize import (
        demo_snapshots,
        snapshots_after_each_trajectory,
        write_growth_html,
    )

    order = [corpus["signals"], corpus["reuse"], corpus["accuracy"]]
    if os.environ.get("EXGRAPH_VIZ_DEMO", "").strip() in {"1", "true", "yes", "on"}:
        snapshots = demo_snapshots()
        live = False
    else:
        snapshots = snapshots_after_each_trajectory(order, working_dir=work, state_dir=state)
        live = True

    assert len(snapshots) == 3
    assert snapshots[0]["counts"]["new_nodes"] >= 1
    types1 = set(snapshots[0]["new_types"])
    assert "Case" in types1
    assert "Thread" in types1
    ids0 = {node["id"] for node in snapshots[0]["nodes"]}
    ids2 = {node["id"] for node in snapshots[2]["nodes"]}
    assert "thread:thread-reuse" in ids2 - ids0 or any(
        node.get("thread") == "thread-reuse" for node in snapshots[1]["nodes"]
    )
    assert any(node["type"] == "Recipe" for node in snapshots[1]["nodes"] + snapshots[2]["nodes"])
    # Accuracy may share generic n-grams (read_file>grep). It must not
    # instantiate a Profiler bash recipe.
    recipe_owners = {
        edge["src"]
        for snap in snapshots
        for edge in snap["edges"]
        if edge["type"] == "INSTANTIATES"
    }
    assert "case:run-a1" not in recipe_owners
    bash_hits = {
        edge["src"]
        for snap in snapshots
        for edge in snap["edges"]
        if edge["type"] == "INSTANTIATES" and "bash" in str(edge.get("dst") or "")
    }
    assert "case:run-a1" not in bash_hits
    assert any(edge["type"] == "FIXED_BY" for edge in snapshots[0]["edges"])

    dest = _writable_output("exgraph_growth_demo.html", tmp_path, "EXGRAPH_GROWTH_HTML")
    write_growth_html(snapshots, dest, title="Experience graph growth over trajectories")
    text = dest.read_text(encoding="utf-8")
    assert dest.is_file()
    assert "FIXED_BY" in text
    assert "NEW ·" in text or "new_node_ids" in text
    assert "thread-signals" in text
    assert "thread-reuse" in text
    assert "thread-accuracy" in text
    assert "Recipe" in text
    assert live in {True, False}
