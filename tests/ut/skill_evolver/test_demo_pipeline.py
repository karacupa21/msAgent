#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""End-to-end demo-mode tests: the synthetic fixture through both commands on a scripted LLM."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from langchain_core.messages import HumanMessage
from rich.console import Console

# The handlers package must be initialized before the evolver modules are
# imported directly (a pre-existing import cycle the CLI never triggers).
import msagent.cli.handlers  # noqa: F401
from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import theme
from msagent.exgraph.config import ENV_DISABLED, ENV_ENABLED
from msagent.exgraph.config import reset_config_cache as reset_exgraph_cache
from msagent.skill_evolver import direct_skill_generation as dsg_module
from msagent.skill_evolver import mining as mining_module
from msagent.skill_evolver.bundle import EvidenceBundle, build_evidence_bundle
from msagent.skill_evolver.config import DirectSkillGenerationConfig, load_skill_evolver_config
from msagent.skill_evolver.direct_skill_generation import DirectSkillGenerationHandler
from msagent.skill_evolver.features import extract_episodes
from msagent.skill_evolver.policy import (
    DEMO_WORKFLOW_BLOCK,
    STRICT_KNOWLEDGE_BLOCK,
    generation_policy_block,
    review_policy_block,
)
from msagent.skills.factory import Skill
from msagent.trajectory_recorder.config import reset_config_cache
from msagent.trajectory_recorder.reader import load_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "trajectories"
DEMO = FIXTURES / "skill_evolver_demo_success.jsonl"
SIGNALS = FIXTURES / "skill_evolver_signals.jsonl"
DEMO_THREAD_ID = "thread-demo-synthetic"
DEMO_AGENT = "SyntheticDemo"
SIGNALS_THREAD_ID = "thread-signals"
SIGNALS_AGENT = "Profiler"
NO_EPISODES = f"Nothing to save: no episodes detected in thread {DEMO_THREAD_ID}"

CLASSIFY_TEMPLATE = "Library:\n{skill_library}\n\nPolicy:\n{selection_policy}\n\nBundle:\n{evidence_bundle}\n"
GENERATION_TEMPLATE = "Policy:\n{generation_policy}\n\nCandidates:\n{candidates}\n\nExisting:\n{existing_skill}\n"
REVIEW_TEMPLATE = (
    "Policy:\n{review_policy}\n\nSkill:\n{skill_md}\n\nCandidates:\n{candidates}\n\n"
    "Evidence:\n{evidence}\n\nExisting:\n{existing_skill}\n"
)
REVIEW_PASS = json.dumps({"verdict": "pass", "issues": []})
DEMO_SKILL_NAME = "demo-csv-column-sum"
SUM_COMMAND = (
    'python3 -c "import csv,sys; print(sum(float(r[sys.argv[2]]) for r in csv.DictReader(open(sys.argv[1]))))"'
)
DEMO_SKILL = "\n".join(
    [
        "---",
        f"name: {DEMO_SKILL_NAME}",
        "description: Use when a CSV column must be summed and the total checked against an expected value.",
        "---",
        "",
        "# CSV column sum",
        "",
        "## Inputs",
        "",
        "- `<csv-path>`: a CSV file with a header row.",
        "- `<column>`: the numeric column to sum.",
        "- `<expected-total>`: the number the total is compared with.",
        "",
        "## Workflow",
        "",
        f"1. Run `{SUM_COMMAND} <csv-path> <column>` and compare the printed number with `<expected-total>`.",
        "",
        "## Outputs",
        "",
        "The column total, compared with the expected value.",
        "",
    ]
)
# Built at runtime so no secret-looking literal is committed.
TOKEN = "Bearer " + "x" * 24
LEAKY_SKILL = DEMO_SKILL.replace(
    "## Outputs",
    f"## Constraints\n\n- Send the header `Authorization: {TOKEN}` with every request.\n\n## Outputs",
)


def _recorder() -> Console:
    return Console(record=True, width=200, no_color=True, theme=theme.rich_theme)


class _NullStatus:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _ConsoleSpy:
    """Records the themed console calls; muted lines land in ``plain`` with their markup."""

    def __init__(self) -> None:
        self.info: list[str] = []
        self.success: list[str] = []
        self.warning: list[str] = []
        self.error: list[str] = []
        self.plain: list[str] = []
        self.renderables: list[object] = []
        self.console = SimpleNamespace(status=lambda *_a, **_k: _NullStatus(), print=self.renderables.append)

    def print(self, *args, **_kwargs) -> None:
        self.plain.extend(str(arg) for arg in args)

    def print_info(self, content: str) -> None:
        self.info.append(content)

    def print_success(self, content: str) -> None:
        self.success.append(content)

    def print_warning(self, content: str) -> None:
        self.warning.append(content)

    def print_error(self, content: str) -> None:
        self.error.append(content)

    def text(self) -> str:
        return "\n".join(self.plain)

    def rendered_text(self) -> str:
        recorder = _recorder()
        for renderable in self.renderables:
            recorder.print(renderable)
        return recorder.export_text()


def _boom(*_args, **_kwargs):
    raise AssertionError("the LLM must not be created")


@pytest.fixture(autouse=True)
def _exgraph_default(monkeypatch: pytest.MonkeyPatch):
    """Neither env switch set: the packaged default (off) applies unless a test opts in."""
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    monkeypatch.delenv(ENV_DISABLED, raising=False)
    reset_exgraph_cache()
    yield
    reset_exgraph_cache()


# ------------------------------------------------------------ fixture variants


def _events(path: Path = DEMO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _lines(events: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(event, ensure_ascii=False) for event in events]


def _orphans(events: list[dict[str, Any]]) -> list[str]:
    """Every tool.start without its result: three orphan calls."""
    return _lines([event for event in events if event["event"] != "tool.result"])


def _ai_claim(events: list[dict[str, Any]]) -> list[str]:
    """No tool events at all; the assistant's closing message claims the total is right."""
    kept = [
        event
        for event in events
        if event["event"] in ("recorder.attach", "turn.start", "turn.end")
        or (event["event"] == "message.ai" and not event["message"]["data"]["tool_calls"])
    ]
    assert [event["event"] for event in kept] == ["recorder.attach", "turn.start", "message.ai", "turn.end"]
    return _lines(kept)


def _error_output(events: list[dict[str, Any]]) -> list[str]:
    """The check call keeps status ok but its output is a traceback.

    It is the last call on purpose: since v5 a failure inside an ok result that a later ok
    call of the same tool follows is an error_recovery (a real finding, tested in
    test_features), which would admit the thread under the demo override.
    """
    for event in events:
        if event["event"] == "tool.result" and event["seq"] == 11:
            event["output"] = (
                "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nKeyError: 'amount'"
            )
    return _lines(events)


def _with_secret(events: list[dict[str, Any]]) -> list[str]:
    """The sum command carries a bearer token in its environment prefix."""
    for event in events:
        if event["event"] == "tool.start" and event["seq"] == 7:
            event["input"]["cmd"] = f'AUTH="{TOKEN}" ' + event["input"]["cmd"]
        if event["event"] == "message.ai" and event["seq"] == 6:
            call = event["message"]["data"]["tool_calls"][0]
            call["args"]["cmd"] = f'AUTH="{TOKEN}" ' + call["args"]["cmd"]
    lines = _lines(events)
    assert sum(TOKEN in line for line in lines) == 2
    return lines


def _signals_with_chain() -> list[str]:
    """The signals fixture (retry loop and more) plus a clean two-call chain in a new dispatch turn."""

    def event(seq: int, kind: str, **fields: Any) -> str:
        base = {
            "v": 1,
            "event": kind,
            "ts": f"2026-09-02T10:01:{seq - 30:02d}.000+00:00",
            "seq": seq,
            "rec": "rec-a",
            "thread_id": SIGNALS_THREAD_ID,
            "agent": SIGNALS_AGENT,
            "run_id": "run-4",
        }
        base.update(fields)
        return json.dumps(base, ensure_ascii=False)

    graph = {"checkpoint_ns": "tools:z1", "langgraph_node": "tools"}
    lines = SIGNALS.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 35
    return [
        *lines,
        event(
            36,
            "turn.start",
            source="dispatch",
            model="default",
            approval_mode="active",
            user_message="Sum the amount column of input/sales.csv and check the total is 60",
        ),
        event(
            37,
            "tool.start",
            span_id="x1",
            parent_span_id="c1",
            name="read_file",
            input={"path": "input/sales.csv"},
            graph=graph,
        ),
        event(
            38,
            "tool.result",
            span_id="x1",
            parent_span_id="c1",
            name="read_file",
            status="ok",
            duration_ms=1,
            graph=graph,
            output="id,amount\n1,10\n2,20\n3,30\n",
        ),
        event(
            39,
            "tool.start",
            span_id="x2",
            parent_span_id="c2",
            name="bash",
            input={"cmd": f"{SUM_COMMAND} input/sales.csv amount"},
            graph=graph,
        ),
        event(
            40,
            "tool.result",
            span_id="x2",
            parent_span_id="c2",
            name="bash",
            status="ok",
            duration_ms=1,
            graph=graph,
            output="60.0",
        ),
        event(41, "turn.end", status="completed", duration_ms=10),
    ]


# ------------------------------------------------------------------ harness


class _Pipeline:
    """/direct-skill-generation under test: its console spy, trajectory dir and knobs."""

    def __init__(self, handler: DirectSkillGenerationHandler, spy: _ConsoleSpy, trajectories_dir: Path, fake_llm_cls):
        self.handler = handler
        self.spy = spy
        self.trajectories_dir = trajectories_dir
        self.fake_llm_cls = fake_llm_cls
        self.config = DirectSkillGenerationConfig(demo_mode=True, policy="reusable_workflow")
        self.skills: list[Skill] = []
        self.llm = fake_llm_cls()
        self.thread_id = DEMO_THREAD_ID
        self.source = trajectories_dir / f"{DEMO_AGENT}_{DEMO_THREAD_ID}.jsonl"

    def script(self, *replies: str) -> None:
        self.llm = self.fake_llm_cls(*replies)

    def install(
        self, lines: list[str] | None = None, *, thread_id: str = DEMO_THREAD_ID, agent: str = DEMO_AGENT
    ) -> Path:
        """Make ``lines`` (default: the demo fixture) the only trajectory of the project."""
        for stale in self.trajectories_dir.glob("*.jsonl"):
            stale.unlink()
        self.thread_id = thread_id
        self.source = self.trajectories_dir / f"{agent}_{thread_id}.jsonl"
        if lines is None:
            shutil.copy(DEMO, self.source)
        else:
            self.source.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.source

    def bundle(self) -> EvidenceBundle:
        """The demo bundle the pipeline builds for the installed trajectory (same budgets)."""
        trajectory = load_trajectory(self.source)
        episodes = extract_episodes(trajectory, demo=True)
        return build_evidence_bundle(
            episodes,
            [trajectory],
            max_chars=self.config.bundle_max_chars,
            excerpt_chars=self.config.excerpt_max_chars,
            demo=True,
        )

    def observed(self, run_id: str = "run-d1") -> tuple[str, list[str]]:
        """Episode id and fragment ids of the observed_procedure chain of ``run_id``."""
        bundle = self.bundle()
        episode_id, episode = next(
            (eid, ep)
            for eid, ep in bundle.episode_ids.items()
            if ep.kind == "observed_procedure" and ep.anchors[0].startswith(f"{run_id}#")
        )
        refs = {item.ref for item in episode.evidence}
        return episode_id, [fid for fid, fragment in bundle.shown.items() if fragment.ref in refs]

    def instruction(self, call: int) -> str:
        """The first human message of the ``call``-th LLM payload."""
        return self.llm.payloads[call][0][1]

    def proposal(self, name: str = DEMO_SKILL_NAME) -> Path:
        return self.root / "skills" / ".proposals" / self.thread_id / name / "SKILL.md"

    @property
    def root(self) -> Path:
        return Path(self.handler.session.context.working_dir)

    def reports(self) -> list[dict[str, Any]]:
        folder = initializer.get_project_paths(self.root).root / "skill-evolver" / "decisions"
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]


def _session(working_dir: Path, agent: str) -> SimpleNamespace:
    context = SimpleNamespace(agent=agent, thread_id=DEMO_THREAD_ID, working_dir=working_dir, model="default")
    return SimpleNamespace(context=context, graph=None)


async def _fake_stage_prompt(_self, _root, _cfg, stage):
    templates = {"classify": CLASSIFY_TEMPLATE, "generate": GENERATION_TEMPLATE, "review": REVIEW_TEMPLATE}
    return templates[stage], f"packaged/{stage}/prompt_v2.md"


@pytest.fixture
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_llm_cls):
    spy = _ConsoleSpy()
    monkeypatch.setattr(dsg_module, "console", spy)
    monkeypatch.delenv("MSAGENT_TRAJECTORY_CONFIG", raising=False)
    reset_config_cache()

    trajectories_dir = initializer.get_project_paths(tmp_path).root / "trajectories"
    trajectories_dir.mkdir(parents=True)
    state = _Pipeline(DirectSkillGenerationHandler(_session(tmp_path, DEMO_AGENT)), spy, trajectories_dir, fake_llm_cls)
    state.install()

    async def fake_load_history(_session, _target):
        return state.thread_id, [HumanMessage(content="sum it")]

    async def fake_load_skills(_self):
        return list(state.skills)

    async def fake_load_llm_config(_model, _working_dir):
        return SimpleNamespace(model="fake-model", context_window=128_000)

    monkeypatch.setattr(dsg_module, "load_history", fake_load_history)
    monkeypatch.setattr(DirectSkillGenerationHandler, "_load_config", staticmethod(lambda: state.config))
    monkeypatch.setattr(DirectSkillGenerationHandler, "_load_stage_prompt", _fake_stage_prompt)
    monkeypatch.setattr(DirectSkillGenerationHandler, "_load_skills", fake_load_skills)
    monkeypatch.setattr(initializer, "load_llm_config", fake_load_llm_config)
    monkeypatch.setattr(initializer.llm_factory, "create", lambda _config: state.llm)
    yield state
    reset_config_cache()


def _decision(episode_id: str, decision: str = "accept", reason_code: str = "accepted") -> dict[str, Any]:
    return {
        "episode_ids": [episode_id],
        "decision": decision,
        "reason_code": reason_code,
        "explanation": "scripted",
        "evidence_refs": [],
    }


def _candidate(refs: list[str], **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "title": "CSV column sum",
        "rule": "Sum a CSV column with csv.DictReader and compare the total with the expected value.",
        "evidence_refs": list(refs),
        "future_applicability": "high",
        "target": {"action": "create", "existing_skill": None},
    }
    data.update(overrides)
    return data


def _classify_reply(episode_id: str, *candidates: dict[str, Any]) -> str:
    """A contract-2 reply accepting ``episode_id`` (or rejecting it when no candidate is given)."""
    decision = _decision(episode_id) if candidates else _decision(episode_id, "reject", "routine_activity")
    return json.dumps(
        {
            "contract_version": 2,
            "verdict": "save" if candidates else "nothing",
            "candidates": list(candidates),
            "decisions": [decision],
        }
    )


def _demo_replies(pipeline: _Pipeline, skill: str = DEMO_SKILL, **overrides: Any) -> tuple[str, str, str]:
    """classify (citing the real fragment ids of the fixture's chain), generate, review."""
    episode_id, refs = pipeline.observed()
    return _classify_reply(episode_id, _candidate(refs, **overrides)), skill, REVIEW_PASS


def _seq_at(path: Path, line: int) -> int:
    """The ``seq`` written on physical ``line`` of ``path`` (read like the reader does)."""
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            if number == line:
                return json.loads(raw)["seq"]
    raise AssertionError(f"{path} has no line {line}")


def _library_skill(root: Path, name: str, category: str = "default") -> Skill:
    skill_dir = root / "skills" / category / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: Use when a CSV column has to be summed.\n---\n\n# Sum\n\nold body\n",
        encoding="utf-8",
    )
    return Skill(name=name, description="Use when a CSV column has to be summed.", category=category, path=path)


def _provenance(proposal: Path) -> dict[str, Any]:
    return json.loads((proposal.parent / "provenance.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ scenarios


@pytest.mark.asyncio
async def test_demo_fixture_reaches_a_valid_demo_proposal(pipeline: _Pipeline, tmp_path: Path) -> None:
    pipeline.script(*_demo_replies(pipeline))

    await pipeline.handler.handle([])

    assert pipeline.spy.error == [] and pipeline.spy.warning == []
    assert len(pipeline.llm.payloads) == 3 and pipeline.llm.replies == []
    proposal = pipeline.proposal()
    assert proposal.is_file() and proposal.read_text(encoding="utf-8") == DEMO_SKILL
    assert proposal.parent.parent.parent.name == ".proposals" and proposal.parent.parent.name == DEMO_THREAD_ID
    # The library itself is untouched: only the inactive proposal exists.
    assert not (tmp_path / "skills" / "default").exists()
    assert sorted(p.name for p in (tmp_path / "skills").iterdir()) == [".proposals"]

    provenance = _provenance(proposal)
    assert provenance["provenance_version"] == 5 and provenance["demo"] is True
    assert provenance["policy"] == {"requested": "reusable_workflow", "selection": "demo_workflow", "source": "config"}
    assert provenance["target"] == {"action": "create", "existing_skill": None, "existing_path": None}
    assert provenance["verification"] == {"level": "evidence_supported", "note": "not executed by generator"}
    assert provenance["quality_review"]["verdict"] == "pass" and provenance["quality_review"]["corrected"] is False
    assert set(provenance["prompt_hashes"]) == {"classify", "generate", "review"}
    assert all(
        value.startswith("sha256:") and len(value) == len("sha256:") + 64
        for value in provenance["prompt_hashes"].values()
    )
    assert provenance["prompt_variants"]["review"] == "packaged/review/prompt_v2.md"
    # Every observed_procedure event resolves to the physical line that holds it.
    source_rows = provenance["observed_procedure_source"]
    assert [row["role"] for row in source_rows] == ["task", "step", "result", "step", "result", "step", "result"]
    for row in source_rows:
        assert row["source"] == pipeline.source.name
        assert _seq_at(pipeline.source, row["line"]) == row["seq"]
    for entry in provenance["evidence_shown"].values():
        assert _seq_at(pipeline.source, entry["line"]) == entry["seq"]

    plain = pipeline.spy.text()
    assert "Requested policy: reusable_workflow (config.skill.evolver.yml)" in plain
    assert "Demo mode: true (config.skill.evolver.yml)" in plain
    assert "Effective selection: demo_workflow" in plain
    assert "Evidence score: 0.00 (min_evidence_score 1.00; 1 episodes, 1 incidents)" in plain
    assert "Gate: pass (demo_override; observed_procedure has required evidence)" in plain
    assert "Candidate 'CSV column sum': accepted (trivial procedure allowed)" in plain
    assert "Quality review: passed" in plain
    assert "Verification: evidence_supported; not executed by generator" in plain
    assert "Proposal: saved, inactive, DEMO" in plain
    assert pipeline.spy.success == [f"Skill proposal saved to {proposal}"]

    (report,) = pipeline.reports()
    assert report["command"] == "direct-skill-generation" and report["thread_id"] == DEMO_THREAD_ID
    assert report["synthetic"] is True
    assert report["gate"]["reason"] == "demo_override" and report["gate"]["passes"] is True
    assert report["policy"] == {
        "requested": "reusable_workflow",
        "selection": "demo_workflow",
        "demo_mode": True,
        "overrides": [],
    }
    assert report["proposals"] == [str(proposal)] and report["llm"]["calls_used"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["strict_knowledge", "reusable_workflow"])
async def test_demo_fixture_without_demo_is_refused_before_llm(
    pipeline: _Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    pipeline.config = DirectSkillGenerationConfig(demo_mode=False, policy=policy)
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    monkeypatch.setattr(initializer, "load_llm_config", _boom)

    await pipeline.handler.handle([])

    assert pipeline.spy.error == []
    assert pipeline.spy.info == [dsg_module._DEPRECATION_HINT, NO_EPISODES]
    plain = pipeline.spy.text()
    assert f"Requested policy: {policy} (config.skill.evolver.yml)" in plain
    assert "Demo mode: false (config.skill.evolver.yml)" in plain
    assert "Gate: skip (no episodes)" in plain and "demo_override" not in plain
    assert not (tmp_path / "skills").exists()
    (report,) = pipeline.reports()
    assert report["gate"] == {
        "score": 0.0,
        "min_evidence_score": 1.0,
        "passes": False,
        "reason": "no episodes",
        "detail": "",
        "incidents": 0,
    }
    assert report["episodes"]["total"] == 0 and report["stop_message"] == NO_EPISODES
    assert report["llm"]["calls_used"] == 0 and report["proposals"] == []


@pytest.mark.asyncio
async def test_demo_with_strict_knowledge_has_no_contradictory_instructions(pipeline: _Pipeline) -> None:
    pipeline.config = DirectSkillGenerationConfig(demo_mode=True, policy="strict_knowledge")
    pipeline.script(*_demo_replies(pipeline))

    await pipeline.handler.handle([])

    assert pipeline.spy.error == [], pipeline.spy.error
    classify = pipeline.instruction(0)
    assert "# Selection policy: demo_workflow" in classify and DEMO_WORKFLOW_BLOCK in classify
    assert "# Selection policy: strict_knowledge" not in classify and STRICT_KNOWLEDGE_BLOCK not in classify
    assert "non-obvious" not in classify and "{selection_policy}" not in classify
    generation = pipeline.instruction(1)
    assert generation_policy_block(True) in generation and generation_policy_block(False) not in generation
    review = pipeline.instruction(2)
    assert review_policy_block(True) in review and review_policy_block(False) not in review
    provenance = _provenance(pipeline.proposal())
    assert provenance["policy"] == {"requested": "strict_knowledge", "selection": "demo_workflow", "source": "config"}
    assert provenance["demo"] is True
    assert "Effective selection: demo_workflow" in pipeline.spy.text()


@pytest.mark.asyncio
async def test_demo_considers_observed_procedure_next_to_a_retry_episode(pipeline: _Pipeline) -> None:
    pipeline.install(_signals_with_chain(), thread_id=SIGNALS_THREAD_ID, agent=SIGNALS_AGENT)
    episode_id, refs = pipeline.observed("run-4")
    assert len(refs) == 5
    pipeline.script(_classify_reply(episode_id, _candidate(refs)), DEMO_SKILL, REVIEW_PASS)

    await pipeline.handler.handle([])

    assert pipeline.spy.error == [], pipeline.spy.error
    classify = pipeline.instruction(0)
    # The retry loop of the profiling turn and the clean chain of the new turn are both shown.
    assert "— retry_loop (weight 0.70" in classify
    assert "— observed_procedure (weight 0.00" in classify
    assert f"### Episode {episode_id} — observed_procedure" in classify
    assert "Sum the amount column of input/sales.csv" in classify
    assert "msprof --collect train.py" in classify
    # The ordinary rules admit the thread; demo adds the chain without changing the score.
    plain = pipeline.spy.text()
    assert "Evidence score: 2.60 (min_evidence_score 1.00; 8 episodes, 5 incidents)" in plain
    assert "Gate: pass (score >= min_evidence_score)" in plain
    assert "Proposal: saved, inactive, DEMO" in plain
    proposal = pipeline.proposal()
    assert proposal.is_file()
    provenance = _provenance(proposal)
    assert [row["seq"] for row in provenance["observed_procedure_source"]] == [36, 37, 38, 39, 40]
    kinds = {row["kind"] for row in provenance["episodes"]}
    assert {"retry_loop", "observed_procedure", "error_recovery", "user_correction", "approval_denied"} <= kinds
    (report,) = pipeline.reports()
    assert report["episodes"]["counts"]["retry_loop"] == 1 and report["episodes"]["counts"]["observed_procedure"] == 3
    assert report["gate"]["reason"] == "score >= min_evidence_score"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("variant", "detail"),
    [
        (_orphans, "3 domain calls: 0 ok, 0 error, 3 orphan; no completed chain"),
        (_ai_claim, "0 domain calls: 0 ok, 0 error, 0 orphan; no completed chain"),
    ],
)
async def test_demo_orphan_or_ai_claim_only_yields_no_skill(
    pipeline: _Pipeline,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: Callable[[list[dict[str, Any]]], list[str]],
    detail: str,
) -> None:
    pipeline.install(variant(_events()))
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)

    await pipeline.handler.handle([])

    assert pipeline.spy.error == []
    assert pipeline.spy.info == [dsg_module._DEPRECATION_HINT, NO_EPISODES]
    assert "Gate: skip (no episodes)" in pipeline.spy.text()
    assert not (tmp_path / "skills").exists()
    (report,) = pipeline.reports()
    assert report["gate"]["passes"] is False and report["llm"]["calls_used"] == 0
    assert report["episodes"]["detector_notes"] == [
        {"detector": "observed_procedure", "reason": "no_completed_calls", "detail": detail, "seqs": []},
    ]


@pytest.mark.asyncio
async def test_demo_ok_status_with_error_output_is_not_success(
    pipeline: _Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline.install(_error_output(_events()))
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)

    await pipeline.handler.handle([])

    assert pipeline.spy.error == []
    assert pipeline.spy.info == [dsg_module._DEPRECATION_HINT, NO_EPISODES]
    assert not (tmp_path / "skills").exists()
    (report,) = pipeline.reports()
    assert report["episodes"]["counts"] == {} and report["llm"]["calls_used"] == 0
    (note,) = report["episodes"]["detector_notes"]
    assert (note["detector"], note["reason"], note["seqs"]) == ("observed_procedure", "error_in_output", [4, 7, 10])
    assert note["detail"] == "bash at seq 11 returned ok but its output contains 'Traceback (most recent call last)'"


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupted", [False, True])
async def test_empty_or_corrupted_trajectory_makes_no_llm_call(
    pipeline: _Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupted: bool
) -> None:
    header = DEMO.read_text(encoding="utf-8").splitlines()[0]
    pipeline.install([header, "{not json", "[1, 2", "garbage"] if corrupted else [])
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)

    await pipeline.handler.handle([])

    assert pipeline.spy.error == []
    decisions = initializer.get_project_paths(tmp_path).root / "skill-evolver" / "decisions"
    if corrupted:
        # A readable header and only broken lines: no turns, no episodes, an explicit refusal.
        assert load_trajectory(pipeline.source).malformed_lines == 3
        assert pipeline.spy.info == [dsg_module._DEPRECATION_HINT, NO_EPISODES]
        (report,) = pipeline.reports()
        assert report["gate"]["reason"] == "no episodes" and report["llm"]["calls_used"] == 0
    else:
        (warning,) = pipeline.spy.warning
        assert warning.startswith("Trajectory unreadable (") and "no events found" in warning
        assert warning.endswith("; the LLM was not called")
        assert pipeline.spy.info == [dsg_module._DEPRECATION_HINT]
        assert not decisions.exists() or list(decisions.glob("*.json")) == []
    assert not (tmp_path / "skills").exists()


@pytest.mark.asyncio
async def test_demo_with_on_nothing_stop_still_works(pipeline: _Pipeline) -> None:
    pipeline.config = DirectSkillGenerationConfig(demo_mode=True, policy="reusable_workflow", on_nothing="stop")
    pipeline.script(*_demo_replies(pipeline))

    await pipeline.handler.handle([])

    assert pipeline.spy.error == [] and len(pipeline.llm.payloads) == 3
    assert pipeline.proposal().is_file()
    assert "Proposal: saved, inactive, DEMO" in pipeline.spy.text()
    (report,) = pipeline.reports()
    assert report["config"]["effective"]["on_nothing"] == "stop" and report["plans"]["proposals"] == 1


@pytest.mark.asyncio
async def test_demo_duplicate_of_active_skill_creates_separate_demo_proposal(
    pipeline: _Pipeline, tmp_path: Path
) -> None:
    library = _library_skill(tmp_path, "csv-column-sum")
    before = library.path.read_bytes()
    pipeline.skills = [library]
    pipeline.script(*_demo_replies(pipeline, covered_by="csv-column-sum"))

    await pipeline.handler.handle([])

    assert pipeline.spy.error == [] and pipeline.spy.warning == []
    assert "- csv-column-sum: Use when a CSV column has to be summed." in pipeline.instruction(0)
    generation = pipeline.instruction(1)
    assert (
        "Covered by library skill: csv-column-sum (write a separate teaching skill; do not copy the library text)"
        in generation
    )
    assert "old body" not in generation
    proposal = pipeline.proposal(DEMO_SKILL_NAME)
    assert proposal.is_file()
    provenance = _provenance(proposal)
    assert provenance["target"]["action"] == "create" and provenance["demo"] is True
    assert provenance["covering_skill"] == {
        "display_name": "csv-column-sum",
        "path": str(library.path),
        "sha256": hashlib.sha256(before).hexdigest(),
    }
    assert provenance["candidates"][0]["covered_by"] == "csv-column-sum"
    assert library.path.read_bytes() == before
    assert sorted(p.name for p in (tmp_path / "skills" / "default").iterdir()) == ["csv-column-sum"]


@pytest.mark.asyncio
async def test_demo_classify_reply_with_update_is_a_contract_error_not_a_swap(
    pipeline: _Pipeline, tmp_path: Path
) -> None:
    library = _library_skill(tmp_path, "csv-column-sum")
    before = library.path.read_bytes()
    pipeline.skills = [library]
    episode_id, refs = pipeline.observed()
    update = _candidate(refs, target={"action": "update", "existing_skill": "csv-column-sum"})
    pipeline.script(_classify_reply(episode_id, update), _classify_reply(episode_id, update), DEMO_SKILL, REVIEW_PASS)

    await pipeline.handler.handle([])

    # One corrective retry, then the diagnosed contract error; the generation stage is never reached.
    assert len(pipeline.llm.payloads) == 2 and len(pipeline.llm.replies) == 2
    correction = pipeline.llm.payloads[1][-1][1]
    assert "target.action 'update' is not allowed in demo mode" in correction
    (error,) = pipeline.spy.error
    assert error.startswith("Error generating skill: classify: reply violates the contract after one retry: ")
    assert "not allowed in demo mode" in error and "Nothing to save" not in error
    assert not any("Nothing to save" in line for line in pipeline.spy.info)
    assert not (tmp_path / "skills" / ".proposals").exists()
    assert library.path.read_bytes() == before
    (report,) = pipeline.reports()
    assert report["failed"].startswith("ClassifyContractError: ")
    assert [row["code"] for row in report["code_rejections"]] == ["contract_error"]
    assert report["stages"][-1]["stage"] == "classify" and report["stages"][-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_dry_run_demo_prints_effective_config_and_writes_nothing(
    pipeline: _Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    user_file = config_dir / "config.skill.evolver.yml"
    text = "schema_version: 2\nclassification:\n  policy: strict_knowledge\ndemo_mode: false\n"
    user_file.write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        DirectSkillGenerationHandler, "_load_config", staticmethod(lambda: load_skill_evolver_config(config_dir))
    )
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    monkeypatch.setattr(initializer, "load_llm_config", _boom)

    await pipeline.handler.handle(["--dry-run", "--demo", "--policy", "reusable_workflow"])

    assert pipeline.spy.error == [] and pipeline.spy.warning == []
    assert pipeline.spy.info == [dsg_module._DEPRECATION_HINT, dsg_module._DRY_RUN_DONE]
    plain = pipeline.spy.text()
    assert "Requested policy: reusable_workflow (--policy)" in plain
    assert "Demo mode: true (--demo)" in plain
    assert "Effective selection: demo_workflow" in plain
    assert "Evidence score: 0.00 (min_evidence_score 1.00; 1 episodes, 1 incidents)" in plain
    assert "Gate: pass (demo_override; observed_procedure has required evidence)" in plain
    assert "observed_procedure: read_file → bash → bash (7 events)" in plain
    assert "observed_procedure candidates: 1 selected; dropped: none" in plain
    assert user_file.read_text(encoding="utf-8") == text
    assert not (tmp_path / "skills").exists()
    assert not (initializer.get_project_paths(tmp_path).root / "skill-evolver").exists()


@pytest.mark.asyncio
async def test_llm_budget_exhausted_writes_nothing(pipeline: _Pipeline, tmp_path: Path) -> None:
    pipeline.config = DirectSkillGenerationConfig(demo_mode=True, policy="reusable_workflow", max_llm_calls=2)
    pipeline.script(*_demo_replies(pipeline))

    await pipeline.handler.handle([])

    # classify (1) and generate (2); the quality review is refused, so nothing is written.
    assert len(pipeline.llm.payloads) == 2 and pipeline.llm.replies == [REVIEW_PASS]
    (error,) = [line for line in pipeline.spy.error if line.startswith("plan ")]
    assert "LLM call budget exhausted (2/2 calls per thread); SKILL.md not written" in error
    assert pipeline.spy.success == []
    assert not (tmp_path / "skills").exists()
    (report,) = pipeline.reports()
    assert report["llm"] == {
        "model": "fake-model",
        "context_window": 128_000,
        "calls_used": 2,
        "limit": 2,
        "bound": 2,
        "budget_exhausted": True,
        "note": "transport retries not counted",
    }
    assert [row["code"] for row in report["code_rejections"]] == ["budget_exhausted"]
    assert report["plans"] == {
        "generated": 1,
        "proposals": 0,
        "generation_errors": 1,
        "rejected_targets": 0,
        "deferred": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("leaky", [True, False])
async def test_secret_in_trajectory_never_reaches_the_package(pipeline: _Pipeline, tmp_path: Path, leaky: bool) -> None:
    pipeline.install(_with_secret(_events()))
    episode_id, refs = pipeline.observed()
    classify = _classify_reply(episode_id, _candidate(refs))
    # A leaky SKILL.md reply echoes the token twice (first try and the validator's corrective turn).
    pipeline.script(*((classify, LEAKY_SKILL, LEAKY_SKILL) if leaky else (classify, DEMO_SKILL, REVIEW_PASS)))

    await pipeline.handler.handle([])

    # The token was shown to the models as evidence; it must stop there.
    assert TOKEN in pipeline.instruction(0) and TOKEN in pipeline.instruction(1)
    proposal = pipeline.proposal()
    if leaky:
        # The validator refuses the echoed token twice; the plan is rejected and nothing written.
        assert len(pipeline.llm.payloads) == 3
        assert any("possible secret (bearer token)" in line for line in pipeline.spy.error)
        assert any(
            line.endswith("SKILL.md rejected after one correction; nothing written:") for line in pipeline.spy.error
        )
        assert not (tmp_path / "skills").exists()
    else:
        assert pipeline.spy.error == [], pipeline.spy.error
        assert proposal.is_file() and TOKEN not in proposal.read_text(encoding="utf-8")
        provenance_text = (proposal.parent / "provenance.json").read_text(encoding="utf-8")
        assert TOKEN not in provenance_text and "[REDACTED:bearer token]" in provenance_text
        provenance = json.loads(provenance_text)
        step = next(entry for entry in provenance["evidence_shown"].values() if entry["seq"] == 7)
        assert 'AUTH=\\"[REDACTED:bearer token]\\"' in step["text"]
    for report_path in (initializer.get_project_paths(tmp_path).root / "skill-evolver" / "decisions").glob("*"):
        assert TOKEN not in report_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", [ENV_ENABLED, ENV_DISABLED])
async def test_demo_pipeline_with_exgraph_on_and_off(
    pipeline: _Pipeline, monkeypatch: pytest.MonkeyPatch, switch: str
) -> None:
    monkeypatch.setenv(switch, "1")
    reset_exgraph_cache()
    pipeline.script(*_demo_replies(pipeline))

    await pipeline.handler.handle([])

    assert pipeline.spy.error == [], pipeline.spy.error
    assert len(pipeline.llm.payloads) == 3
    # The stored-graph appendix is the only difference the classifier sees.
    assert ("## Experience graph (stored)" in pipeline.instruction(0)) is (switch == ENV_ENABLED)
    proposal = pipeline.proposal(DEMO_SKILL_NAME)
    assert proposal.is_file() and proposal.read_text(encoding="utf-8") == DEMO_SKILL
    provenance = _provenance(proposal)
    assert (provenance["demo"], provenance["policy"]["selection"]) == (True, "demo_workflow")
    assert "Proposal: saved, inactive, DEMO" in pipeline.spy.text()
    (report,) = pipeline.reports()
    assert report["proposals"] == [str(proposal)] and report["gate"]["reason"] == "demo_override"


# ------------------------------------------------------------ /skill-mine mirror


@pytest.fixture
def mine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_llm_cls):
    """A mining handler over a tmp project holding only the demo fixture; the LLM explodes unless scripted."""
    spy = _ConsoleSpy()
    monkeypatch.setattr(mining_module, "console", spy)
    monkeypatch.delenv("MSAGENT_TRAJECTORY_CONFIG", raising=False)
    reset_config_cache()

    trajectories = initializer.get_project_paths(tmp_path).root / "trajectories"
    trajectories.mkdir(parents=True)
    # Only the demo fixture: other fixtures would add repeated_procedure n-grams and pass the gate on score.
    shutil.copy(DEMO, trajectories / f"{DEMO_AGENT}_{DEMO_THREAD_ID}.jsonl")

    state = SimpleNamespace(
        handler=mining_module.SkillMiningHandler(_session(tmp_path, DEMO_AGENT)),
        spy=spy,
        root=tmp_path,
        config=DirectSkillGenerationConfig(output_dir=tmp_path / "skills"),
        llm=None,
    )
    monkeypatch.setattr(DirectSkillGenerationHandler, "_load_config", staticmethod(lambda: state.config))
    monkeypatch.setattr(DirectSkillGenerationHandler, "_load_stage_prompt", _fake_stage_prompt)

    async def fake_refresh(*, agent: str, working_dir: Path) -> list:
        return []

    monkeypatch.setattr(initializer, "refresh_cached_skills", fake_refresh)
    monkeypatch.setattr(initializer, "load_llm_config", _boom)
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)

    def script(*replies: str):
        async def fake_load_llm_config(_model, _working_dir):
            return SimpleNamespace(model="fake-model", context_window=128_000)

        state.llm = fake_llm_cls(*replies)
        monkeypatch.setattr(initializer, "load_llm_config", fake_load_llm_config)
        monkeypatch.setattr(initializer.llm_factory, "create", lambda _config: state.llm)
        return state.llm

    state.script = script
    yield state
    reset_config_cache()


@pytest.mark.asyncio
async def test_mine_demo_thread_writes_demo_proposal(mine) -> None:
    trajectory = load_trajectory(DEMO)
    bundle = build_evidence_bundle(extract_episodes(trajectory, demo=True), [trajectory], excerpt_chars=1000, demo=True)
    (episode_id,) = bundle.episode_ids
    mine.script(_classify_reply(episode_id, _candidate(sorted(bundle.shown))), DEMO_SKILL, REVIEW_PASS)

    await mine.handler.handle(["--thread", DEMO_THREAD_ID, "--demo"])

    assert mine.spy.error == [], mine.spy.error
    assert len(mine.llm.payloads) == 3 and mine.llm.replies == []
    proposal = mine.root / "skills" / ".proposals" / DEMO_THREAD_ID / DEMO_SKILL_NAME / "SKILL.md"
    assert proposal.is_file() and proposal.read_text(encoding="utf-8") == DEMO_SKILL
    provenance = _provenance(proposal)
    assert provenance["demo"] is True
    assert provenance["policy"] == {"requested": "strict_knowledge", "selection": "demo_workflow", "source": "config"}
    plain = mine.spy.text()
    assert "Demo mode: true (--demo)" in plain and "Effective selection: demo_workflow" in plain
    assert "Gate: pass (demo_override; observed_procedure has required evidence)" in plain
    assert "Proposal: saved, inactive, DEMO" in plain
    assert "Threads (min_evidence_score 1.00, demo override)" in mine.spy.rendered_text()
    assert any("demo_override; observed_procedure has required evidence" in line for line in mine.spy.info)
    assert any("Mined 1 threads: 1 proposals, 0 skipped by the gate" in line for line in mine.spy.success)
    decisions = initializer.get_project_paths(mine.root).root / "skill-evolver" / "decisions"
    (report_path,) = sorted(decisions.glob("*.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["command"] == "skill-mine" and report["synthetic"] is True
    assert report["policy"]["overrides"] == ["--demo"] and report["gate"]["reason"] == "demo_override"


@pytest.mark.asyncio
async def test_mine_demo_thread_without_demo_flag_skips_gate(mine) -> None:
    await mine.handler.handle(["--thread", DEMO_THREAD_ID])

    assert mine.spy.error == []
    assert any(
        line == "Nothing to mine: 1 threads skipped by the gate (min_evidence_score 1.00)" for line in mine.spy.info
    )
    plain = mine.spy.text()
    assert "Demo mode: false (config.skill.evolver.yml)" in plain and "demo_override" not in plain
    text = mine.spy.rendered_text()
    assert "Threads (min_evidence_score 1.00)" in text and "no episodes" in text
    assert not (mine.root / "skills").exists()
    # The refused thread still leaves its gate-only decision report.
    decisions = initializer.get_project_paths(mine.root).root / "skill-evolver" / "decisions"
    (report_path,) = sorted(decisions.glob("*.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["command"] == "skill-mine" and report["gate"]["passes"] is False
    assert report["stop_message"] == NO_EPISODES
    assert report["llm"]["calls_used"] == 0 and report["proposals"] == []
