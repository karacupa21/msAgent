#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""P2.2 classify-input A/B. No new graph types, no LLM.

Runs the same trajectory through attach_stored_graph under
``episodes`` / ``hybrid`` / ``graph`` and records what classify would see.
Full /skill-mine A/B (verdict + proposal) is the same env flip plus the
evolver command; this module measures the input half deterministically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from msagent.exgraph.config import ENV_DISABLED, ENV_ENABLED, ENV_EVIDENCE_MODE, reset_config_cache
from msagent.trajectory_recorder.model import Trajectory

MODES = ("episodes", "hybrid", "graph")


@dataclass
class ModeSnapshot:
    mode: str
    text: str
    chars: int
    appendix_chars: int
    has_graph: bool
    graph_first: bool
    invented_evidence: bool
    shards: int


@dataclass
class AbReport:
    thread_id: str
    agent: str
    source: str
    episode_kinds: list[str]
    evidence_score: float
    modes: list[ModeSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "agent": self.agent,
            "source": self.source,
            "episode_kinds": self.episode_kinds,
            "evidence_score": self.evidence_score,
            "modes": [
                {
                    "mode": snap.mode,
                    "chars": snap.chars,
                    "appendix_chars": snap.appendix_chars,
                    "has_graph": snap.has_graph,
                    "graph_first": snap.graph_first,
                    "invented_evidence": snap.invented_evidence,
                    "shards": snap.shards,
                }
                for snap in self.modes
            ],
        }


def _facts(text: str) -> dict[str, bool]:
    extra_ok = "Experience graph" in text
    return {
        "has_graph": extra_ok,
        "graph_first": text.lstrip().startswith("## Experience graph"),
    }


def compare_trajectory(
    trajectory: Trajectory,
    *,
    working_dir: Path,
    state_dir: Path,
    modes: tuple[str, ...] = ("episodes", "hybrid"),
) -> AbReport:
    """Build the evolver bundle once, then attach under each mode."""
    from msagent.skill_evolver.bundle import build_evidence_bundle
    from msagent.skill_evolver.exgraph_context import attach_stored_graph
    from msagent.skill_evolver.features import evidence_score, extract_episodes

    episodes = extract_episodes(trajectory)
    bundle = build_evidence_bundle(episodes, [trajectory]).text
    report = AbReport(
        thread_id=trajectory.thread_id,
        agent=trajectory.agent,
        source=str(getattr(trajectory, "path", "") or trajectory.source),
        episode_kinds=sorted({ep.kind for ep in episodes}),
        evidence_score=float(evidence_score(episodes)),
    )
    os.environ[ENV_ENABLED] = "1"
    os.environ.pop(ENV_DISABLED, None)
    for mode in modes:
        if mode not in MODES:
            continue
        os.environ[ENV_EVIDENCE_MODE] = mode
        reset_config_cache()
        text = attach_stored_graph(
            bundle,
            trajectory,
            working_dir=working_dir,
            state_dir=state_dir,
        )
        extra = text[len(bundle) :] if text.startswith(bundle) else text
        if mode == "graph" and "## Episode bundle" in text:
            extra = text.split("## Episode bundle", 1)[0]
        facts = _facts(text)
        invented = "Evidence:" in extra or "[ev" in extra and mode != "episodes"
        if mode == "episodes":
            invented = False
        shards = list(state_dir.rglob("nodes.jsonl")) if mode == "episodes" else list(state_dir.rglob("nodes.jsonl"))
        report.modes.append(
            ModeSnapshot(
                mode=mode,
                text=text,
                chars=len(text),
                appendix_chars=max(0, len(text) - len(bundle)) if mode != "graph" else max(0, len(extra)),
                has_graph=facts["has_graph"] and mode != "episodes",
                graph_first=facts["graph_first"] and mode == "graph",
                invented_evidence=bool(invented) and mode != "episodes",
                shards=len(list(state_dir.rglob("nodes.jsonl"))),
            )
        )
    os.environ.pop(ENV_EVIDENCE_MODE, None)
    reset_config_cache()
    return report


def render_markdown(report: AbReport) -> str:
    lines = [
        "# ExGraph classify A/B",
        "",
        f"- thread: `{report.thread_id}` agent `{report.agent}`",
        f"- source: `{report.source}`",
        f"- episodes: {report.episode_kinds}",
        f"- evidence_score: {report.evidence_score:.2f}",
        "",
        "| mode | chars | appendix | graph | graph-first | invented Evidence | shards |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for snap in report.modes:
        lines.append(
            f"| {snap.mode} | {snap.chars} | {snap.appendix_chars} | "
            f"{snap.has_graph} | {snap.graph_first} | {snap.invented_evidence} | {snap.shards} |"
        )
    lines += [
        "",
        "episodes = Skill Evolver first pass (no shard from this hook).",
        "hybrid = bundle + relation appendix. graph = relations first.",
        "This report is the classify *input*. To A/B proposals, run "
        "`/skill-mine` twice with `MSAGENT_EXGRAPH_EVIDENCE_MODE`.",
        "",
    ]
    return "\n".join(lines)
