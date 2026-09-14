#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""P2.2/P2.4 classify-input A/B. No new graph types, no LLM."""

from __future__ import annotations

import json
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


def _graph_portion(mode: str, bundle: str, text: str) -> str:
    if mode == "episodes":
        return ""
    if mode == "hybrid":
        if text.startswith(bundle):
            return text[len(bundle) :]
        return text if "Experience graph" in text else ""
    if "## Episode bundle" in text:
        return text.split("## Episode bundle", 1)[0]
    return text if text.lstrip().startswith("## Experience graph") else ""


def _invented_evidence(portion: str) -> bool:
    if not portion:
        return False
    return "Evidence:" in portion or "[ev" in portion


def compare_trajectory(
    trajectory: Trajectory,
    *,
    working_dir: Path,
    state_dir: Path,
    modes: tuple[str, ...] = ("episodes", "hybrid"),
    force_enable: bool = True,
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
    saved_enabled = os.environ.get(ENV_ENABLED)
    saved_disabled = os.environ.get(ENV_DISABLED)
    saved_mode = os.environ.get(ENV_EVIDENCE_MODE)
    if force_enable:
        os.environ[ENV_ENABLED] = "1"
        os.environ.pop(ENV_DISABLED, None)
    try:
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
            portion = _graph_portion(mode, bundle, text)
            report.modes.append(
                ModeSnapshot(
                    mode=mode,
                    text=text,
                    chars=len(text),
                    appendix_chars=len(portion),
                    has_graph="Experience graph" in text and mode != "episodes",
                    graph_first=text.lstrip().startswith("## Experience graph") and mode == "graph",
                    invented_evidence=_invented_evidence(portion),
                    shards=len(list(state_dir.rglob("nodes.jsonl"))),
                )
            )
    finally:
        for key, value in (
            (ENV_ENABLED, saved_enabled),
            (ENV_DISABLED, saved_disabled),
            (ENV_EVIDENCE_MODE, saved_mode),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_config_cache()
    return report


def compare_decision_reports(path_a: Path, path_b: Path) -> dict[str, Any]:
    """Diff two Skill Evolver decision JSON files from /skill-mine."""

    def pick(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        clf = data.get("classifier") or {}
        proposals = list(data.get("proposals") or [])
        return {
            "path": str(path),
            "thread_id": data.get("thread_id"),
            "verdict": clf.get("verdict"),
            "candidates": clf.get("candidates"),
            "proposals": proposals,
            "proposal_count": len(proposals),
        }

    left, right = pick(path_a), pick(path_b)
    return {
        "a": left,
        "b": right,
        "same_thread": left["thread_id"] == right["thread_id"],
        "same_verdict": left["verdict"] == right["verdict"],
        "same_proposal_count": left["proposal_count"] == right["proposal_count"],
    }


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
