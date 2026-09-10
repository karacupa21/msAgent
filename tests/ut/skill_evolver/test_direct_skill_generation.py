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

"""Tests for /direct-skill-generation: the evidence pipeline on a scripted LLM."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage
from rich.markup import escape

# The handlers package must be initialized before the module is imported directly:
# handlers/__init__ re-exports the handler while the handler imports session_history
# from that same package (a pre-existing import cycle that the CLI never triggers).
import msagent.cli.handlers  # noqa: F401
from msagent.core.constants import CONFIG_SKILL_EVOLVER_FILE_NAME
from msagent.skill_evolver import direct_skill_generation as module
from msagent.skill_evolver.budget import ContextBudget
from msagent.skill_evolver.bundle import build_evidence_bundle
from msagent.skill_evolver.classify import Candidate
from msagent.skill_evolver.config import SkillEvolverConfig, SkillEvolverConfigError, effective_rules
from msagent.skill_evolver.direct_skill_generation import (
    DirectSkillGenerationConfig,
    DirectSkillGenerationHandler,
    PlanContext,
)
from msagent.skill_evolver.features import extract_episodes
from msagent.skill_evolver.prompts import PromptText, StagePrompts, prompt_sha256
from msagent.skill_evolver.render import NO_EXISTING_SKILL, RenderPlan, RenderPlans
from msagent.skill_evolver.retrieval import BM25Index
from msagent.skills.factory import Skill, SkillFactory
from msagent.trajectory_recorder.config import reset_config_cache
from msagent.trajectory_recorder.model import ToolCall, Trajectory, Turn
from msagent.trajectory_recorder.reader import load_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "trajectories" / "skill_evolver_signals.jsonl"
THREAD_ID = "thread-signals"
AGENT = "Profiler"

CLASSIFY_TEMPLATE = "Library:\n{skill_library}\n\nPolicy:\n{selection_policy}\n\nBundle:\n{evidence_bundle}\n"
RENDER_TEMPLATE = "Policy:\n{render_policy}\n\nCandidates:\n{candidates}\n\nExisting:\n{existing_skill}\n"
REVIEW_TEMPLATE = (
    "Policy:\n{review_policy}\n\nSkill:\n{skill_md}\n\nCandidates:\n{candidates}\n\n"
    "Evidence:\n{evidence}\n\nExisting:\n{existing_skill}\n"
)
REVIEW_PASS = json.dumps({"verdict": "pass", "issues": []})
SKILL_NAME = "generated-source-debugging"
VALID_SKILL = "\n".join(
    [
        "---",
        f"name: {SKILL_NAME}",
        "description: Use when diagnosing failures involving generated source artifacts.",
        "---",
        "",
        "# Generated Source Debugging",
        "",
        "## Inputs",
        "",
        "- The failing output.",
        "",
        "## Workflow",
        "",
        "1. Reproduce the failure.",
        "2. Regenerate the sources before type checking.",
        "",
        "## Outputs",
        "",
        "A verified diagnosis.",
        "",
    ]
)
INVALID_SKILL = "---\nname: fix-it\ndescription: Instructions for debugging\n---\n"
SECOND_NAME = "kernel-profile-first"


def _revised(name: str) -> str:
    """VALID_SKILL under another frontmatter name."""
    return VALID_SKILL.replace(f"name: {SKILL_NAME}", f"name: {name}")


SECOND_SKILL = _revised(SECOND_NAME)


class _NullStatus:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _ConsoleSpy:
    def __init__(self) -> None:
        self.info: list[str] = []
        self.success: list[str] = []
        self.warning: list[str] = []
        self.error: list[str] = []
        # Muted lines (markup included), e.g. "[muted]Gate: pass (...)[/muted]".
        self.plain: list[str] = []
        self.console = SimpleNamespace(status=lambda *_args, **_kwargs: _NullStatus())

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


class _Pipeline:
    """The handler under test, its console spy and the knobs a test may turn."""

    def __init__(self, handler: DirectSkillGenerationHandler, spy: _ConsoleSpy, trajectories_dir: Path, fake_llm_cls):
        self.handler = handler
        self.spy = spy
        self.trajectories_dir = trajectories_dir
        self.fake_llm_cls = fake_llm_cls
        self.config = DirectSkillGenerationConfig()
        self.skills: list[Skill] = []
        self.llm = fake_llm_cls()

    def script(self, *replies: str) -> None:
        self.llm = self.fake_llm_cls(*replies)


def _session(working_dir: Path) -> SimpleNamespace:
    context = SimpleNamespace(agent=AGENT, thread_id=THREAD_ID, working_dir=working_dir, model="default")
    return SimpleNamespace(context=context, graph=None)


@pytest.fixture
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_llm_cls):
    spy = _ConsoleSpy()
    monkeypatch.setattr(module, "console", spy)
    monkeypatch.delenv("MSAGENT_TRAJECTORY_CONFIG", raising=False)
    reset_config_cache()

    # Where resolve_trajectories_dir() looks for this working dir (isolated home).
    trajectories_dir = module.initializer.get_project_paths(tmp_path).root / "trajectories"
    trajectories_dir.mkdir(parents=True)
    shutil.copy(FIXTURE, trajectories_dir / f"{AGENT}_{THREAD_ID}.jsonl")

    state = _Pipeline(DirectSkillGenerationHandler(_session(tmp_path)), spy, trajectories_dir, fake_llm_cls)

    async def fake_load_history(_session, _target):
        return THREAD_ID, [HumanMessage(content="分析性能")]

    async def fake_stage_prompt(self, _root, _cfg, stage):
        templates = {"classify": CLASSIFY_TEMPLATE, "render": RENDER_TEMPLATE, "review": REVIEW_TEMPLATE}
        return templates[stage], f"packaged/{stage}/prompt_v1.md"

    async def fake_load_skills(self):
        return list(state.skills)

    async def fake_load_llm_config(_model, _working_dir):
        return SimpleNamespace(model="fake-model", context_window=128_000)

    monkeypatch.setattr(module, "load_history", fake_load_history)
    monkeypatch.setattr(DirectSkillGenerationHandler, "_load_config", staticmethod(lambda: state.config))
    monkeypatch.setattr(DirectSkillGenerationHandler, "_load_stage_prompt", fake_stage_prompt)
    monkeypatch.setattr(DirectSkillGenerationHandler, "_load_skills", fake_load_skills)
    monkeypatch.setattr(module.initializer, "load_llm_config", fake_load_llm_config)
    monkeypatch.setattr(module.initializer.llm_factory, "create", lambda _config: state.llm)
    yield state
    reset_config_cache()


# ------------------------------------------------------------------ builders


def _classify_reply(*candidates: dict[str, Any], verdict: str = "save") -> str:
    """A contract-2 reply; the signals fixture always renders E1, so its default decision is valid."""
    if candidates:
        decision = {"decision": "accept", "reason_code": "accepted"}
    else:
        decision = {"decision": "reject", "reason_code": "routine_activity"}
    decisions = [{"episode_ids": ["E1"], **decision, "explanation": "scripted", "evidence_refs": []}]
    return json.dumps(
        {"contract_version": 2, "verdict": verdict, "candidates": list(candidates), "decisions": decisions}
    )


def _candidate(refs: list[str], **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "title": "Generated source debugging",
        "rule": "Regenerate sources before type checking.",
        "evidence_refs": list(refs),
        "future_applicability": "high",
        "target": {"action": "create", "existing_skill": None},
    }
    data.update(overrides)
    return data


def _update(refs: list[str], existing: str, **overrides: Any) -> dict[str, Any]:
    target = {"action": "update", "existing_skill": existing}
    return _candidate(refs, target=target, **overrides)


def _valid_refs(path: Path = FIXTURE) -> list[str]:
    """Two fragment ids the classify stage keeps: they are in the trajectory's evidence bundle."""
    trajectory = load_trajectory(path)
    return sorted(build_evidence_bundle(extract_episodes(trajectory), [trajectory]).shown)[:2]


def _seq_at(path: Path, line: int) -> int:
    """The ``seq`` written on physical ``line`` of ``path`` (read like the reader does)."""
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            if number == line:
                return json.loads(raw)["seq"]
    raise AssertionError(f"{path} has no line {line}")


def _library_skill(tmp_path: Path, name: str, category: str = "default") -> Skill:
    skill_dir = tmp_path / "skills" / category / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: Use when testing.\n---\nold body\n", encoding="utf-8")
    return Skill(name=name, description="Use when testing.", category=category, path=path)


def _proposal(tmp_path: Path, name: str) -> Path:
    return tmp_path / "skills" / ".proposals" / THREAD_ID / name / "SKILL.md"


def _provenance_of(proposal: Path) -> dict[str, Any]:
    return json.loads((proposal.parent / "provenance.json").read_text(encoding="utf-8"))


def _instruction(pipeline: _Pipeline, call: int) -> str:
    """The first human message of the ``call``-th LLM payload."""
    return pipeline.llm.payloads[call][0][1]


def _call(name: str, seq: int) -> ToolCall:
    return ToolCall(
        span_id=f"s{seq}",
        parent_span_id=None,
        name=name,
        args={},
        status="ok",
        output_text="",
        error_type=None,
        error=None,
        duration_ms=1,
        seq_start=seq,
        seq_end=seq + 1,
        line_start=seq,
        line_end=seq + 1,
        subagent=None,
    )


def _trajectory(thread_id: str, names: list[str]) -> Trajectory:
    calls = [_call(name, seq=2 * index + 2) for index, name in enumerate(names)]
    turn = Turn(
        run_id="run-1",
        seq_start=1,
        line_start=1,
        user_message="do it",
        source="dispatch",
        tool_calls=calls,
        status="completed",
    )
    return Trajectory(
        path=Path(f"{thread_id}.jsonl"),
        thread_id=thread_id,
        agent=AGENT,
        model=None,
        working_dir="/w",
        started_at="2026-09-01T10:00:00.000+00:00",
        turns=[turn],
    )


# ------------------------------------------------------------ handle: writes


@pytest.mark.asyncio
async def test_handle_writes_proposal_not_library(pipeline: _Pipeline, tmp_path: Path) -> None:
    refs = _valid_refs()
    stated = _candidate(refs, applies_when="the generated sources are older than the schema")
    pipeline.script(_classify_reply(stated), VALID_SKILL, REVIEW_PASS)

    await pipeline.handler.handle([])

    proposal = tmp_path / "skills" / ".proposals" / THREAD_ID / SKILL_NAME / "SKILL.md"
    assert proposal.is_file(), (pipeline.spy.error, pipeline.spy.warning, pipeline.spy.info)
    assert proposal.read_text(encoding="utf-8") == VALID_SKILL
    assert not (tmp_path / "skills" / "default").exists()
    assert pipeline.spy.success == [f"Skill proposal saved to {proposal}"]
    assert pipeline.spy.error == [] and pipeline.spy.warning == []
    # Three calls: classify, render, quality review.
    assert len(pipeline.llm.payloads) == 3
    classify_instruction = pipeline.llm.payloads[0][0][1]
    render_instruction = pipeline.llm.payloads[1][0][1]
    assert "[ev1]" in classify_instruction and "Evidence: seq" not in classify_instruction
    assert "The skill library is currently empty." in classify_instruction
    assert "# Selection policy: strict_knowledge" in classify_instruction
    assert "{selection_policy}" not in classify_instruction
    assert "Regenerate sources before type checking." in render_instruction
    assert "   When: the generated sources are older than the schema" in render_instruction

    provenance = json.loads((proposal.parent / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["provenance_version"] == 4
    assert provenance["thread_ids"] == [THREAD_ID]
    assert provenance["model"] == "fake-model"
    assert provenance["prompt_variants"] == {
        "classify": "packaged/classify/prompt_v1.md",
        "render": "packaged/render/prompt_v1.md",
        "review": "packaged/review/prompt_v1.md",
    }
    assert provenance["prompt_hashes"] == {
        "classify": prompt_sha256(CLASSIFY_TEMPLATE),
        "render": prompt_sha256(RENDER_TEMPLATE),
        "review": prompt_sha256(REVIEW_TEMPLATE),
    }
    assert provenance["features_version"] == 4
    assert provenance["category"] == "default"
    assert provenance["policy"] == {
        "requested": "strict_knowledge",
        "selection": "strict_knowledge",
        "source": "config",
    }
    assert provenance["demo"] is False
    assert provenance["quality_review"]["verdict"] == "pass" and provenance["quality_review"]["corrected"] is False
    assert provenance["verification"] == {"level": "evidence_supported", "note": "not executed by generator"}
    assert provenance["target"] == {"action": "create", "existing_skill": None, "existing_path": None}
    source = pipeline.trajectories_dir / f"{AGENT}_{THREAD_ID}.jsonl"
    assert provenance["sources"] == {source.name: str(source)}
    # The fragments the rendered candidate cites (a subset of what the
    # classify model saw) are in the classify payload and point at the
    # physical line holding that very event.
    shown = provenance["evidence_shown"]
    assert set(shown) == set(refs)
    for fragment_id, entry in shown.items():
        assert f"- [{fragment_id}] {entry['text']}" in classify_instruction
        assert entry["source"] == source.name
        assert _seq_at(source, entry["line"]) == entry["seq"]
    assert provenance["episodes"]
    for episode in provenance["episodes"]:
        assert episode["thread_id"] == THREAD_ID
        assert episode["bundle_status"] == "shown"
        assert all(item["id"] is None or item["id"] in shown for item in episode["evidence"])
    assert not any(line.startswith("Plans:") for line in pipeline.spy.info)
    (candidate,) = provenance["candidates"]
    assert (candidate["candidate_id"], candidate["evidence_refs"]) == ("c1", refs)
    assert candidate["applies_when"] == "the generated sources are older than the schema"
    assert provenance["candidates_rejected"] == []
    # What the renderer was quoted is recorded and was really in its payload.
    quoted = provenance["render_evidence"]["c1"]
    assert quoted and set(quoted) <= set(refs)
    for fragment_id in quoted:
        assert f"   - {shown[fragment_id]['text']}" in render_instruction
        assert fragment_id not in render_instruction


@pytest.mark.asyncio
async def test_handle_long_correction_phrase_is_in_provenance(pipeline: _Pipeline, tmp_path: Path) -> None:
    # The correcting phrase sits after the first 600 characters of the user
    # message (line 17 of the fixture, the run-2 turn.start).
    source = pipeline.trajectories_dir / f"{AGENT}_{THREAD_ID}.jsonl"
    lines = source.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[16])
    assert event["event"] == "turn.start" and event["run_id"] == "run-2"
    phrase = "不对：首先应该看 kernel-level 的性能剖析"
    event["user_message"] = "任务背景和前情提要。" * 64 + phrase + "，而不是 summary。"
    assert event["user_message"].index(phrase) > 600
    lines[16] = json.dumps(event, ensure_ascii=False)
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    trajectory = load_trajectory(source)
    bundle = build_evidence_bundle(extract_episodes(trajectory), [trajectory])
    (correction,) = [f for f in bundle.shown.values() if f.role == "correction"]
    assert phrase in correction.text
    pipeline.script(_classify_reply(_candidate([correction.id])), VALID_SKILL, REVIEW_PASS)

    await pipeline.handler.handle([])

    proposal = tmp_path / "skills" / ".proposals" / THREAD_ID / SKILL_NAME / "SKILL.md"
    assert proposal.is_file(), (pipeline.spy.error, pipeline.spy.warning, pipeline.spy.info)
    provenance = json.loads((proposal.parent / "provenance.json").read_text(encoding="utf-8"))
    entry = provenance["evidence_shown"][correction.id]
    assert phrase in entry["text"]
    assert _seq_at(source, entry["line"]) == entry["seq"] == 17
    assert phrase in pipeline.llm.payloads[0][0][1]
    assert phrase in pipeline.llm.payloads[1][0][1]
    assert provenance["render_evidence"]["c1"] == [correction.id]


@pytest.mark.asyncio
async def test_handle_bundle_exclusion_creates_no_llm(
    pipeline: _Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A budget too small for any episode's required evidence excludes them
    # all; their weight must not carry the thread to the LLM.
    pipeline.config = DirectSkillGenerationConfig(bundle_max_chars=1)

    def boom(_config):
        raise AssertionError("the LLM must not be created")

    monkeypatch.setattr(module.initializer.llm_factory, "create", boom)
    count = len(extract_episodes(load_trajectory(FIXTURE)))

    await pipeline.handler.handle([])

    assert pipeline.spy.error == []
    assert len(pipeline.spy.warning) == count
    assert all(line.startswith("Excluded ") and "insufficient context" in line for line in pipeline.spy.warning)
    assert pipeline.spy.info[-1] == (
        f"Nothing to save: {count} episodes excluded from the bundle; "
        "remaining evidence score 0.00 < min_evidence_score 1.00"
    )
    assert not (tmp_path / "skills").exists()


@pytest.mark.asyncio
async def test_proposal_invisible_to_skill_scanners(pipeline: _Pipeline, tmp_path: Path) -> None:
    from msagent.agents.factory import AgentFactory

    pipeline.script(_classify_reply(_candidate(_valid_refs())), VALID_SKILL, REVIEW_PASS)
    await pipeline.handler.handle([])
    skills_root = tmp_path / "skills"
    assert (skills_root / ".proposals").is_dir()

    assert await SkillFactory().load_skills(skills_root) == {}
    assert AgentFactory._resolve_existing_paths([skills_root]) == []


@pytest.mark.asyncio
async def test_handle_update_passes_existing_text(pipeline: _Pipeline, tmp_path: Path) -> None:
    skill = _library_skill(tmp_path, "real")
    original = skill.path.read_text(encoding="utf-8")
    pipeline.skills = [skill]
    target = {"action": "update", "existing_skill": "real"}
    revised = VALID_SKILL.replace(f"name: {SKILL_NAME}", "name: real")
    pipeline.script(_classify_reply(_candidate(_valid_refs(), target=target)), revised, REVIEW_PASS)

    await pipeline.handler.handle([])

    proposal = tmp_path / "skills" / ".proposals" / THREAD_ID / "real" / "SKILL.md"
    assert proposal.is_file(), (pipeline.spy.error, pipeline.spy.warning)
    assert original in pipeline.llm.payloads[1][0][1]
    # The reviewer sees the existing skill too.
    assert original in pipeline.llm.payloads[2][0][1]
    assert "- real: Use when testing." in pipeline.llm.payloads[0][0][1]
    assert skill.path.read_text(encoding="utf-8") == original
    provenance = json.loads((proposal.parent / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["provenance_version"] == 4
    assert provenance["target"] == {
        "action": "update",
        "existing_skill": "real",
        "existing_path": str(skill.path),
        "base_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
    }
    assert [c["candidate_id"] for c in provenance["candidates"]] == ["c1"]
    assert pipeline.spy.success == [f"Skill proposal saved to {proposal}"]
    assert pipeline.spy.error == []


# ----------------------------------------------------------- handle: refusals


@pytest.mark.asyncio
async def test_handle_refuses_without_trajectory(pipeline: _Pipeline, tmp_path: Path) -> None:
    for file in pipeline.trajectories_dir.iterdir():
        file.unlink()

    await pipeline.handler.handle([])

    assert pipeline.llm.payloads == []
    (error,) = pipeline.spy.error
    assert error.startswith(f"No recorded trajectory for thread {THREAD_ID} in ")
    assert error.endswith("; the LLM was not called")
    assert not (tmp_path / "skills").exists()


@pytest.mark.asyncio
async def test_handle_stops_below_evidence_threshold(pipeline: _Pipeline, tmp_path: Path) -> None:
    pipeline.config = DirectSkillGenerationConfig(min_evidence_score=100.0)

    await pipeline.handler.handle([])

    assert pipeline.llm.payloads == []
    hint, info = pipeline.spy.info
    assert hint == module._DEPRECATION_HINT
    assert info.startswith("Nothing to save: evidence score ")
    assert "< min_evidence_score 100.00" in info
    assert info.endswith("incidents)")
    assert not (tmp_path / "skills").exists()


@pytest.mark.asyncio
async def test_handle_nothing_verdict(pipeline: _Pipeline, tmp_path: Path) -> None:
    pipeline.script(_classify_reply(verdict="nothing"))

    await pipeline.handler.handle([])

    assert pipeline.spy.info == [
        module._DEPRECATION_HINT,
        f"Nothing to save: no durable learning found in thread {THREAD_ID}",
    ]
    assert len(pipeline.llm.payloads) == 1
    assert not (tmp_path / "skills").exists()


@pytest.mark.asyncio
async def test_handle_fabricated_refs_dropped(pipeline: _Pipeline, tmp_path: Path) -> None:
    pipeline.script(_classify_reply(_candidate(["ev9999"])))

    await pipeline.handler.handle([])

    assert pipeline.spy.warning == [
        "Rejected 'Generated source debugging': evidence not shown in the bundle: ['ev9999']",
    ]
    assert pipeline.spy.info == [
        module._DEPRECATION_HINT,
        f"Nothing to save: no candidate with valid evidence refs in thread {THREAD_ID} (invalid_evidence)",
    ]
    assert len(pipeline.llm.payloads) == 1
    assert not (tmp_path / "skills").exists()


@pytest.mark.asyncio
async def test_handle_rejects_invalid_skill_twice(pipeline: _Pipeline, tmp_path: Path) -> None:
    pipeline.script(_classify_reply(_candidate(_valid_refs())), INVALID_SKILL, INVALID_SKILL)

    await pipeline.handler.handle([])

    assert len(pipeline.llm.payloads) == 3
    assert pipeline.spy.error[0] == f"plan create: Generated source debugging: {module._REJECTED}"
    assert pipeline.spy.error[-1] == "Plans: 0 proposals, 1 render errors, 0 rejected targets, 0 deferred"
    assert any("task identifier" in error for error in pipeline.spy.error)
    assert any("description: must start with" in error for error in pipeline.spy.error)
    assert any("missing section '## Inputs'" in error for error in pipeline.spy.error)
    assert pipeline.spy.success == []
    assert not (tmp_path / "skills").exists()


@pytest.mark.asyncio
async def test_handle_reference_only(pipeline: _Pipeline, tmp_path: Path) -> None:
    pipeline.skills = [_library_skill(tmp_path, "real")]
    target = {"action": "reference", "existing_skill": "real"}
    pipeline.script(_classify_reply(_candidate(_valid_refs(), target=target)))

    await pipeline.handler.handle([])

    assert len(pipeline.llm.payloads) == 1
    assert pipeline.spy.info == [
        module._DEPRECATION_HINT,
        "reference real: Generated source debugging (classifier's claim; skill text read, coverage not verified)",
        "Nothing to save: no candidate left to render",
    ]
    assert not (tmp_path / "skills" / ".proposals").exists()


@pytest.mark.asyncio
async def test_handle_unknown_update_target_dropped(pipeline: _Pipeline, tmp_path: Path) -> None:
    target = {"action": "update", "existing_skill": "ghost"}
    pipeline.script(_classify_reply(_candidate(_valid_refs(), target=target)))

    await pipeline.handler.handle([])

    assert len(pipeline.llm.payloads) == 1
    assert pipeline.spy.warning == [
        "Rejected target 'Generated source debugging': invalid_target — "
        "existing_skill 'ghost' is not in the skill library"
    ]
    assert pipeline.spy.info == [
        module._DEPRECATION_HINT,
        "Nothing to save: no candidate left to render",
    ]
    assert not (tmp_path / "skills").exists()


@pytest.mark.asyncio
async def test_handle_reference_to_missing_skill_is_invalid_target(pipeline: _Pipeline, tmp_path: Path) -> None:
    target = {"action": "reference", "existing_skill": "ghost"}
    pipeline.script(_classify_reply(_candidate(_valid_refs(), target=target)))

    await pipeline.handler.handle([])

    assert len(pipeline.llm.payloads) == 1
    (warning,) = pipeline.spy.warning
    assert warning.startswith("Rejected target 'Generated source debugging': invalid_target — ")
    assert "'ghost'" in warning
    assert pipeline.spy.info == [module._DEPRECATION_HINT, "Nothing to save: no candidate left to render"]
    assert not (tmp_path / "skills" / ".proposals").exists()


@pytest.mark.asyncio
async def test_handle_ambiguous_bare_name(pipeline: _Pipeline, tmp_path: Path) -> None:
    pipeline.skills = [
        _library_skill(tmp_path, "real", category="profiler"),
        _library_skill(tmp_path, "real", category="modeling"),
    ]
    refs = _valid_refs()
    reference = {"action": "reference", "existing_skill": "real"}
    pipeline.script(
        _classify_reply(
            _update(refs, "real", title="Update rule"),
            _candidate(refs, title="Reference rule", target=reference),
        ),
    )

    await pipeline.handler.handle([])

    assert len(pipeline.llm.payloads) == 1
    assert [w.split(": ")[0] for w in pipeline.spy.warning] == [
        "Rejected target 'Update rule'",
        "Rejected target 'Reference rule'",
    ]
    assert all("ambiguous_target" in w for w in pipeline.spy.warning)
    assert pipeline.spy.info == [module._DEPRECATION_HINT, "Nothing to save: no candidate left to render"]
    assert not (tmp_path / "skills" / ".proposals").exists()


# ------------------------------------------------------- handle: many plans


@pytest.mark.asyncio
async def test_handle_update_a_and_update_b_two_proposals(pipeline: _Pipeline, tmp_path: Path) -> None:
    alpha = _library_skill(tmp_path, "alpha")
    beta = _library_skill(tmp_path, "beta")
    pipeline.skills = [alpha, beta]
    refs = _valid_refs()
    pipeline.script(
        _classify_reply(_update(refs[:1], "alpha", title="Alpha rule"), _update(refs[1:], "beta", title="Beta rule")),
        _revised("alpha"),
        REVIEW_PASS,
        _revised("beta"),
        REVIEW_PASS,
    )

    await pipeline.handler.handle([])

    assert len(pipeline.llm.payloads) == 5 and pipeline.llm.replies == []
    first, second = _instruction(pipeline, 1), _instruction(pipeline, 3)
    assert "name: alpha" in first and "Alpha rule" in first
    assert "name: beta" not in first and "Beta rule" not in first
    assert "name: beta" in second and "Beta rule" in second
    assert "name: alpha" not in second and "Alpha rule" not in second
    proposals = [_proposal(tmp_path, "alpha"), _proposal(tmp_path, "beta")]
    assert all(p.is_file() for p in proposals), (pipeline.spy.error, pipeline.spy.warning)
    assert pipeline.spy.success == [f"Skill proposal saved to {p}" for p in proposals]
    assert pipeline.spy.error == [] and pipeline.spy.warning == []
    for_alpha, for_beta = _provenance_of(proposals[0]), _provenance_of(proposals[1])
    assert for_alpha["target"]["existing_skill"] == "alpha"
    assert [c["candidate_id"] for c in for_alpha["candidates"]] == ["c1"]
    assert set(for_alpha["evidence_shown"]) == set(refs[:1])
    assert for_alpha["render_evidence"] == {"c1": refs[:1]}
    assert "Beta rule" not in json.dumps(for_alpha)
    assert for_beta["target"]["existing_skill"] == "beta"
    assert [c["candidate_id"] for c in for_beta["candidates"]] == ["c2"]
    assert set(for_beta["evidence_shown"]) == set(refs[1:])
    assert pipeline.spy.info[-1] == "Plans: 2 proposals, 0 render errors, 0 rejected targets, 0 deferred"


@pytest.mark.asyncio
async def test_handle_update_and_create_are_separate_plans(pipeline: _Pipeline, tmp_path: Path) -> None:
    pipeline.skills = [_library_skill(tmp_path, "real")]
    refs = _valid_refs()
    pipeline.script(
        _classify_reply(_update(refs, "real", title="Real rule"), _candidate(refs)),
        _revised("real"),
        REVIEW_PASS,
        VALID_SKILL,
        REVIEW_PASS,
    )

    await pipeline.handler.handle([])

    assert len(pipeline.llm.payloads) == 5 and pipeline.llm.replies == []
    update, create = _instruction(pipeline, 1), _instruction(pipeline, 3)
    assert "old body" in update and "Real rule" in update
    assert "Generated source debugging" not in update
    assert NO_EXISTING_SKILL in create and "Generated source debugging" in create
    assert "Real rule" not in create
    assert _proposal(tmp_path, "real").is_file() and _proposal(tmp_path, SKILL_NAME).is_file()
    assert pipeline.spy.warning == [] and pipeline.spy.error == []


@pytest.mark.asyncio
async def test_handle_two_creates_two_proposals(pipeline: _Pipeline, tmp_path: Path) -> None:
    refs = _valid_refs()
    second = _candidate(refs, title="Profile before summary", rule="Collect a kernel profile before summarising.")
    pipeline.script(_classify_reply(_candidate(refs), second), VALID_SKILL, REVIEW_PASS, SECOND_SKILL, REVIEW_PASS)

    await pipeline.handler.handle([])

    assert len(pipeline.llm.payloads) == 5 and pipeline.llm.replies == []
    assert _proposal(tmp_path, SKILL_NAME).is_file()
    assert _proposal(tmp_path, SECOND_NAME).is_file()
    assert len(pipeline.spy.success) == 2
    assert [c["candidate_id"] for c in _provenance_of(_proposal(tmp_path, SECOND_NAME))["candidates"]] == ["c2"]


@pytest.mark.asyncio
async def test_handle_second_create_cannot_reuse_first_name(pipeline: _Pipeline, tmp_path: Path) -> None:
    refs = _valid_refs()
    second = _candidate(refs, title="Profile before summary")
    pipeline.script(
        _classify_reply(_candidate(refs), second), VALID_SKILL, REVIEW_PASS, VALID_SKILL, SECOND_SKILL, REVIEW_PASS
    )

    await pipeline.handler.handle([])

    assert len(pipeline.llm.payloads) == 6 and pipeline.llm.replies == []
    correction = pipeline.llm.payloads[4][-1][1]
    assert f"'{SKILL_NAME}' already exists in the skill library" in correction
    assert _proposal(tmp_path, SKILL_NAME).is_file() and _proposal(tmp_path, SECOND_NAME).is_file()
    assert pipeline.spy.error == []


@pytest.mark.asyncio
async def test_handle_first_plan_fails_second_written(pipeline: _Pipeline, tmp_path: Path) -> None:
    refs = _valid_refs()
    second = _candidate(refs, title="Profile before summary")
    pipeline.script(_classify_reply(_candidate(refs), second), INVALID_SKILL, INVALID_SKILL, SECOND_SKILL, REVIEW_PASS)

    await pipeline.handler.handle([])

    assert len(pipeline.llm.payloads) == 5 and pipeline.llm.replies == []
    assert pipeline.spy.error[0] == f"plan create: Generated source debugging: {module._REJECTED}"
    assert not _proposal(tmp_path, SKILL_NAME).exists()
    assert _proposal(tmp_path, SECOND_NAME).is_file()
    assert pipeline.spy.success == [f"Skill proposal saved to {_proposal(tmp_path, SECOND_NAME)}"]
    assert pipeline.spy.error[-1] == "Plans: 1 proposals, 1 render errors, 0 rejected targets, 0 deferred"


@pytest.mark.asyncio
async def test_handle_plans_over_limit_deferred(pipeline: _Pipeline, tmp_path: Path) -> None:
    pipeline.config = DirectSkillGenerationConfig(max_plans=1)
    refs = _valid_refs()
    second = _candidate(refs, title="Profile before summary")
    pipeline.script(_classify_reply(_candidate(refs), second), VALID_SKILL, REVIEW_PASS)

    await pipeline.handler.handle([])

    assert len(pipeline.llm.payloads) == 3 and pipeline.llm.replies == []
    assert "Deferred plan: create: Profile before summary — max_plans 1 reached" in pipeline.spy.info
    assert _proposal(tmp_path, SKILL_NAME).is_file()
    assert not _proposal(tmp_path, SECOND_NAME).exists()
    assert pipeline.spy.info[-1] == "Plans: 1 proposals, 0 render errors, 0 rejected targets, 1 deferred"


# ------------------------------------------------------------ plan helpers


def _prompts() -> StagePrompts:
    """The fake templates as the pipeline's prompt record (honest hashes of the fake text)."""
    texts = {"classify": CLASSIFY_TEMPLATE, "render": RENDER_TEMPLATE, "review": REVIEW_TEMPLATE}
    return StagePrompts(
        **{
            stage: PromptText(stage, text, f"packaged/{stage}/prompt_v1.md", prompt_sha256(text), 2)
            for stage, text in texts.items()
        }
    )


def _plan_context(tmp_path: Path) -> tuple[PlanContext, list[str]]:
    trajectory = load_trajectory(FIXTURE)
    bundle = build_evidence_bundle(extract_episodes(trajectory), [trajectory])
    prompts = _prompts()
    context = PlanContext(
        thread_id=THREAD_ID,
        thread_ids=[THREAD_ID],
        bundle=bundle,
        rejected=[],
        sources={trajectory.source: str(trajectory.path)},
        model="fake-model",
        prompt_variants=prompts.variants(),
        category="default",
        output_root=tmp_path / "skills",
        prompt_hashes=prompts.hashes(),
        requested=DirectSkillGenerationConfig().as_record(),
    )
    return context, sorted(bundle.shown)[:2]


def _render_kwargs(tmp_path: Path, llm: Any, context: PlanContext, taken: set[str]) -> dict[str, Any]:
    """The keyword arguments of the per-plan function with an unlimited context budget and default rules."""
    return {
        "llm": llm,
        "prompts": _prompts(),
        "context": context,
        "taken": taken,
        "budget": ContextBudget.for_window(None),
        "rules": effective_rules(DirectSkillGenerationConfig(), working_dir=tmp_path),
    }


def _plan(refs: list[str], existing: Skill | None = None, **overrides: Any) -> RenderPlan:
    data = _candidate(refs, **overrides)
    if existing is not None:
        data["target"] = {"action": "update", "existing_skill": existing.display_name}
    candidate = Candidate.model_validate(data).model_copy(update={"candidate_id": "c1"})
    return RenderPlan(candidates=[candidate], existing=existing)


@pytest.mark.asyncio
async def test_render_plan_create_adds_name_to_taken(tmp_path: Path, fake_llm_cls) -> None:
    context, refs = _plan_context(tmp_path)
    llm = fake_llm_cls(VALID_SKILL, REVIEW_PASS)
    taken = {"other"}

    outcome = await DirectSkillGenerationHandler._render_plan(
        _plan(refs), **_render_kwargs(tmp_path, llm, context, taken)
    )

    assert outcome.written and outcome.name == SKILL_NAME
    # Render plus one quality review.
    assert outcome.calls == 2 and outcome.errors == [] and outcome.rejection is None
    assert outcome.review["verdict"] == "pass" and outcome.review["corrected"] is False
    assert outcome.skill_path == tmp_path / "skills" / ".proposals" / THREAD_ID / SKILL_NAME / "SKILL.md"
    assert taken == {"other", SKILL_NAME}
    assert _provenance_of(outcome.skill_path)["target"]["action"] == "create"


@pytest.mark.asyncio
async def test_render_plan_update_reads_existing_and_leaves_taken(tmp_path: Path, fake_llm_cls) -> None:
    context, refs = _plan_context(tmp_path)
    skill = _library_skill(tmp_path, "real")
    llm = fake_llm_cls(_revised("real"), REVIEW_PASS)
    taken: set[str] = set()

    outcome = await DirectSkillGenerationHandler._render_plan(
        _plan(refs, existing=skill), **_render_kwargs(tmp_path, llm, context, taken)
    )

    assert outcome.written and outcome.name == "real" and outcome.calls == 2
    assert "old body" in llm.payloads[0][0][1]
    assert taken == set()
    target = _provenance_of(outcome.skill_path)["target"]
    assert target == {
        "action": "update",
        "existing_skill": "real",
        "existing_path": str(skill.path),
        "base_sha256": hashlib.sha256(skill.path.read_bytes()).hexdigest(),
    }


@pytest.mark.asyncio
async def test_render_plan_invalid_twice_returns_errors_without_writing(tmp_path: Path, fake_llm_cls) -> None:
    context, refs = _plan_context(tmp_path)
    llm = fake_llm_cls(INVALID_SKILL, INVALID_SKILL)
    taken = {"other"}

    outcome = await DirectSkillGenerationHandler._render_plan(
        _plan(refs), **_render_kwargs(tmp_path, llm, context, taken)
    )

    # Two render calls, no review: the validator refused both replies.
    assert not outcome.written and outcome.calls == 2 and outcome.rejection is None and outcome.review is None
    assert any("missing section '## Inputs'" in error for error in outcome.errors)
    assert taken == {"other"}
    assert not (tmp_path / "skills").exists()


def test_report_plans_lines(tmp_path: Path) -> None:
    from msagent.skill_evolver.render import PlanRejection

    skill = Skill(name="real", description="Use when testing.", category="profiler", path=tmp_path / "SKILL.md")
    refs = ["ev1"]
    kept = Candidate.model_validate(_candidate(refs, title="Kept"))
    plans = RenderPlans(
        plans=[],
        deferred=[(RenderPlan(candidates=[kept], existing=None), "max_plans 1 reached")],
        references=[(Candidate.model_validate(_candidate(refs, title="Seen")), skill)],
        rejected=[PlanRejection(Candidate.model_validate(_candidate(refs, title="Lost")), "invalid_target", "why")],
    )

    warnings, notes = DirectSkillGenerationHandler._report_plans(plans, ["unverified"])

    assert warnings == ["Rejected target 'Lost': invalid_target — why"]
    assert notes == [
        "reference profiler/real: Seen (classifier's claim; skill text unreadable, coverage not verified)",
        "Deferred plan: create: Kept — max_plans 1 reached",
    ]


@pytest.mark.asyncio
async def test_handle_reports_pipeline_errors(pipeline: _Pipeline, tmp_path: Path) -> None:
    pipeline.script("not json", "still not json")

    await pipeline.handler.handle([])

    (error,) = pipeline.spy.error
    assert error.startswith("Error generating skill: classify: reply is not valid JSON")
    assert not (tmp_path / "skills").exists()


# ------------------------------------------------------------ evidence


def test_collect_episodes_cross_session_cites_current_thread() -> None:
    current = _trajectory("thread-a", ["bash", "read_file", "grep"])
    other = _trajectory("thread-b", ["bash", "read_file", "grep"])

    episodes = module._collect_episodes(current, [other], skill_index=BM25Index([]))

    (episode,) = episodes
    assert episode.kind == "repeated_procedure"
    assert episode.thread_id == "thread-a"
    assert episode.tool_sequence == ["bash", "read_file", "grep"]
    assert episode.evidence_seq == [1, 2, 3, 4, 5, 6, 7]
    assert episode.facts["thread_ids"] == ["thread-a", "thread-b"]
    # Owner steps interleaved with their optional results, then the
    # supporting session's steps (required evidence, so the bundle must
    # index that trajectory too), then the owner's task context.
    assert [(i.ref.source, i.role, i.ref.seq, i.required) for i in episode.evidence[:6]] == [
        ("thread-a.jsonl", "step", 2, True),
        ("thread-a.jsonl", "result", 3, False),
        ("thread-a.jsonl", "step", 4, True),
        ("thread-a.jsonl", "result", 5, False),
        ("thread-a.jsonl", "step", 6, True),
        ("thread-a.jsonl", "result", 7, False),
    ]
    assert [(i.ref.source, i.ref.seq, i.required) for i in episode.evidence[6:9]] == [
        ("thread-b.jsonl", 2, True),
        ("thread-b.jsonl", 4, True),
        ("thread-b.jsonl", 6, True),
    ]
    task = episode.evidence[9]
    assert (task.ref.source, task.ref.seq, task.role, task.required) == ("thread-a.jsonl", 1, "task", False)
    assert module._supporting([other], episodes) == [other]
    assert module._collect_episodes(current, [], skill_index=BM25Index([])) == []
    assert module._supporting([other], []) == []


def test_cited_threads_lists_supporting_threads_after_current() -> None:
    current = _trajectory("thread-a", ["bash", "read_file", "grep"])
    other = _trajectory("thread-b", ["bash", "read_file", "grep"])
    episodes = module._collect_episodes(current, [other], skill_index=BM25Index([]))

    cited = DirectSkillGenerationHandler._cited_threads(current, episodes)

    assert cited == ["thread-a", "thread-b"]


# ------------------------------------------------------------------ prompts


@pytest.mark.asyncio
async def test_load_stage_prompt_prefers_user_root_then_packaged(tmp_path: Path) -> None:
    handler = DirectSkillGenerationHandler(_session(tmp_path))
    root = tmp_path / "prompts"
    (root / "render").mkdir(parents=True)
    (root / "render" / "prompt_v2.md").write_text(
        "## description: x\n## contract_version: 2\nuser render {candidates} {existing_skill} {render_policy}\n",
        encoding="utf-8",
    )
    cfg = DirectSkillGenerationConfig()

    text, source = await handler._load_stage_prompt(root, cfg, "render")
    assert "user render" in text
    assert source == str(root / "render" / "prompt_v2.md")

    text, source = await handler._load_stage_prompt(root, cfg, "classify")
    assert "{evidence_bundle}" in text and "{skill_library}" in text and "{selection_policy}" in text
    assert Path(source).parts[-2:] == ("classify", "prompt_v2.md")

    with pytest.raises(ValueError, match="Unknown prompt stage"):
        await handler._load_stage_prompt(root, cfg, "default")


@pytest.mark.asyncio
async def test_load_stage_prompt_missing_file_is_an_error(tmp_path: Path, monkeypatch) -> None:
    spy = _ConsoleSpy()
    monkeypatch.setattr(module, "console", spy)
    handler = DirectSkillGenerationHandler(_session(tmp_path))
    root = tmp_path / "prompts"
    (root / "classify").mkdir(parents=True)
    (root / "classify" / "a.md").write_text("A", encoding="utf-8")
    (root / "classify" / "b.md").write_text("B", encoding="utf-8")
    cfg = DirectSkillGenerationConfig(classify_prompt="missing.md")

    with pytest.raises(ValueError, match="prompt file not found"):
        await handler._load_stage_prompt(root, cfg, "classify")

    assert spy.warning == []


def test_packaged_render_prompt_is_resolved_by_default_config() -> None:
    cfg = DirectSkillGenerationHandler._load_config()
    packaged = REPO_ROOT / "resources" / "configs" / "default" / "skill-evolver" / "prompts"

    assert module.STAGES == ("classify", "render", "review")
    for stage in module.STAGES:
        assert (packaged / stage / cfg.prompt_for(stage)).is_file()


# ----------------------------------------------------------- legacy replay


@pytest.fixture
def legacy_handler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spy = _ConsoleSpy()
    monkeypatch.setattr(module, "console", spy)
    return DirectSkillGenerationHandler(_session(tmp_path)), spy


@pytest.mark.asyncio
async def test_generate_skill_md_reports_middle_omissions(legacy_handler, monkeypatch) -> None:
    instance, _spy = legacy_handler
    payloads: list[list] = []

    class _FakeLLM:
        async def ainvoke(self, payload):
            payloads.append(list(payload))
            return AIMessage(content="Nothing to save.")

    async def fake_load_llm_config(_model, _working_dir):
        return SimpleNamespace(context_window=1000)

    async def fake_snapshot(self):
        return "- demo-skill: no description"

    monkeypatch.setattr(module.initializer, "load_llm_config", fake_load_llm_config)
    monkeypatch.setattr(module.initializer.llm_factory, "create", lambda _config: _FakeLLM())
    monkeypatch.setattr(module, "trim_history", lambda messages, _llm, _window, **_kw: (list(messages), 3))
    monkeypatch.setattr(DirectSkillGenerationHandler, "_build_skill_library_snapshot", fake_snapshot)

    result = await instance._generate_skill_md(
        [HumanMessage(content="你好")], "Library:\n{skill_library}", THREAD_ID
    )

    assert result == "Nothing to save."
    instruction = payloads[0][-1].content
    assert instruction.startswith(
        "[Note: 3 messages from the middle of the session were omitted due to context limits.]"
    )
    assert "- demo-skill: no description" in instruction


# ------------------------------------------------------------------- config


def test_load_config_reads_valid_packaged_default() -> None:
    cfg = DirectSkillGenerationHandler._load_config()

    assert cfg.schema_version == 2
    assert cfg.prompt_for("classify") == "prompt_v2.md"
    assert cfg.legacy_format is False
    assert cfg.source == "packaged"


def test_packaged_config_active_passes_variant_validation() -> None:
    config_path = REPO_ROOT / "resources" / "configs" / "default" / "config.skill.evolver.yml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert data["schema_version"] == 2
    SkillEvolverConfig.model_validate(data)


def test_packaged_prompt_matches_single_call_pipeline() -> None:
    prompt_path = (
        REPO_ROOT / "resources" / "configs" / "default" / "skill-evolver" / "prompts" / "default" / "prompt_v1.md"
    )
    text = prompt_path.read_text(encoding="utf-8")

    assert "# Output contract" in text
    assert "{skill_library}" in text
    for agentic_marker in (
        "skills_list",
        "skill_view",
        "skill_manage",
        "`mkdir`",
        "# Standard filesystem and shell access",
        "# Execution rules",
        "# Final response",
    ):
        assert agentic_marker not in text


def test_load_config_default_min_evidence_score() -> None:
    assert DirectSkillGenerationHandler._load_config().min_evidence_score == 1.0


def test_load_config_default_max_plans() -> None:
    assert DirectSkillGenerationHandler._load_config().max_plans == 3


def _write_user_config(text: str) -> None:
    config_dir = module.initializer.app_paths.config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / CONFIG_SKILL_EVOLVER_FILE_NAME.name).write_text(text, encoding="utf-8")


def _config_problems() -> list[tuple[str, str]]:
    with pytest.raises(SkillEvolverConfigError) as info:
        DirectSkillGenerationHandler._load_config()
    return info.value.problems


@pytest.mark.parametrize(("raw", "expected"), [("2.5", 2.5), ("0", 0.0)])
def test_load_config_parses_min_evidence_score(raw: str, expected: float) -> None:
    _write_user_config(f"active: default\nmin_evidence_score: {raw}\n")

    assert DirectSkillGenerationHandler._load_config().min_evidence_score == expected


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("abc", "must be a finite non-negative number, got 'abc'"),
        ("-1", "must be a finite non-negative number, got -1"),
    ],
)
def test_load_config_rejects_invalid_min_evidence_score(raw: str, message: str) -> None:
    _write_user_config(f"active: default\nmin_evidence_score: {raw}\n")

    assert _config_problems() == [("min_evidence_score", message)]


@pytest.mark.parametrize(("raw", "expected"), [("5", 5), ("1", 1)])
def test_load_config_parses_max_plans(raw: str, expected: int) -> None:
    _write_user_config(f"active: default\nmax_plans: {raw}\n")

    assert DirectSkillGenerationHandler._load_config().max_plans == expected


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("0", "must be a positive whole number, got 0"),
        ("abc", "must be a whole number, got 'abc'"),
        ("-2", "must be a positive whole number, got -2"),
    ],
)
def test_load_config_rejects_invalid_max_plans(raw: str, message: str) -> None:
    _write_user_config(f"active: default\nmax_plans: {raw}\n")

    assert _config_problems() == [("max_plans", message)]


def test_load_config_rejects_unsafe_prompt_file() -> None:
    _write_user_config("prompt_file: ../../etc/passwd\n")

    assert _config_problems() == [
        ("prompt_file", "must be a file name without path separators or '..', got '../../etc/passwd'"),
    ]


def test_load_config_v1_user_file_migrates_and_v2_errors_stop() -> None:
    _write_user_config("active: default\nmin_evidence_score: 2.5\n")

    cfg = DirectSkillGenerationHandler._load_config()

    assert cfg.min_evidence_score == 2.5 and cfg.legacy_format is True

    _write_user_config("schema_version: 2\ngate: {min_evidence_score: abc}\n")

    assert _config_problems() == [("gate.min_evidence_score", "must be a finite non-negative number, got 'abc'")]


# --------------------------------------------------------------- arguments


def test_parse_direct_args() -> None:
    assert module.parse_direct_args([]) == module.DirectOptions()
    options = module.parse_direct_args(["last", "--dry-run", "--policy", "reusable_workflow", "--demo"])
    assert options == module.DirectOptions(target="last", policy="reusable_workflow", demo=True, dry_run=True)
    assert module.parse_direct_args(["--no-demo", "abc"]) == module.DirectOptions(target="abc", demo=False)
    for args, message in (
        (["a", "b"], "unexpected argument 'b'"),
        (["--bogus"], "unknown argument '--bogus'"),
        (["--policy"], "--policy requires a value"),
        (["--policy", "x"], "--policy: expected one of strict_knowledge, reusable_workflow, got 'x'"),
        (["--demo", "--no-demo"], "--demo and --no-demo cannot be combined"),
    ):
        with pytest.raises(module.CliArgsError, match=re.escape(message)):
            module.parse_direct_args(args)


@pytest.mark.asyncio
async def test_bad_direct_argument_reports_usage_and_stops(pipeline: _Pipeline) -> None:
    await pipeline.handler.handle(["--bogus"])

    assert pipeline.spy.error == ["unknown argument '--bogus'"]
    # The usage line is markup-escaped ("\\[last|...").
    assert any(escape(module.DIRECT_USAGE) in line for line in pipeline.spy.plain)
    assert pipeline.llm.payloads == []


# ------------------------------------------------------- dry run and report


@pytest.mark.asyncio
async def test_direct_dry_run_creates_no_llm_and_no_report(
    pipeline: _Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_config):
        raise AssertionError("the LLM must not be created")

    monkeypatch.setattr(module.initializer.llm_factory, "create", boom)

    await pipeline.handler.handle(["--dry-run", "--demo"])

    assert pipeline.spy.error == []
    assert pipeline.spy.info == [module._DEPRECATION_HINT, module._DRY_RUN_DONE]
    plain = "\n".join(pipeline.spy.plain)
    assert "Requested policy: strict_knowledge (config.skill.evolver.yml)" in plain
    assert "Demo mode: true (--demo)" in plain
    assert "Effective selection: demo_workflow" in plain
    assert "Evidence score: " in plain and "Gate: pass (score >= min_evidence_score)" in plain
    assert "bundle: " in plain and "observed_procedure candidates: " in plain
    state = module.initializer.get_project_paths(tmp_path).root
    assert not (state / "skill-evolver").exists()
    assert not (tmp_path / "skills").exists()


@pytest.mark.asyncio
async def test_direct_run_writes_decision_report_and_prompts_line_lists_review(
    pipeline: _Pipeline, tmp_path: Path
) -> None:
    pipeline.script(_classify_reply(_candidate(_valid_refs())), VALID_SKILL, REVIEW_PASS)

    await pipeline.handler.handle([])

    assert pipeline.spy.error == []
    decisions = module.initializer.get_project_paths(tmp_path).root / "skill-evolver" / "decisions"
    (report_path,) = sorted(decisions.glob("*.json"))
    assert report_path.name.startswith(f"{THREAD_ID}-")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report_version"] == 1 and report["command"] == "direct-skill-generation"
    assert report["thread_id"] == THREAD_ID and report["synthetic"] is False
    assert report["plans"] == {"rendered": 1, "proposals": 1, "render_errors": 0, "rejected_targets": 0, "deferred": 0}
    assert report["llm"]["calls_used"] == 3 and report["llm"]["limit"] == 16 and report["llm"]["bound"] == 16
    assert report["prompts"]["variants"]["review"] == "packaged/review/prompt_v1.md"
    assert report["gate"]["passes"] is True and report["stop_message"] is None and report["failed"] is None
    assert report["evidence_text_file"] is None
    assert not sorted(decisions.glob("*.evidence.md"))
    plain = pipeline.spy.plain
    prompts_line = (
        "Prompts: classify=packaged/classify/prompt_v1.md, render=packaged/render/prompt_v1.md, "
        "review=packaged/review/prompt_v1.md"
    )
    assert any(prompts_line in line for line in plain)
    assert any("Quality review: passed" in line for line in plain)
    assert any("Verification: evidence_supported; not executed by generator" in line for line in plain)
    assert any(line.endswith("Proposal: saved, inactive[/muted]") for line in plain)
