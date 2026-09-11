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

"""Flow-control tests of the shared per-thread pipeline (run_thread) on a scripted LLM."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import msagent.cli.handlers  # noqa: F401
from msagent.exgraph.config import ENV_DISABLED, reset_config_cache
from msagent.skill_evolver import pipeline as module
from msagent.skill_evolver.budget import CHARS_PER_TOKEN, REPLY_RESERVE_TOKENS, USABLE_RATIO, ContextBudget
from msagent.skill_evolver.bundle import build_evidence_bundle
from msagent.skill_evolver.classify import BUNDLE_PLACEHOLDER, ClassifyParseError
from msagent.skill_evolver.config import DirectSkillGenerationConfig, apply_overrides, effective_rules
from msagent.skill_evolver.features import DetectorNote, extract_episodes
from msagent.skill_evolver.pipeline import (
    REJECTED,
    CodeRejection,
    LazyLlm,
    RunContext,
    ThreadInput,
    ThreadResult,
    covering_skill_record,
    read_existing_skill,
    reference_coverage,
    run_thread,
)
from msagent.skill_evolver.policy import selection_policy_block
from msagent.skill_evolver.prompts import PromptText, StagePrompts, prompt_sha256
from msagent.skill_evolver.render import resolve_library_skill
from msagent.skill_evolver.report import decisions_dir
from msagent.skills.factory import Skill
from msagent.trajectory_recorder.reader import load_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "trajectories"
SIGNALS = FIXTURES / "skill_evolver_signals.jsonl"
DEMO = FIXTURES / "skill_evolver_demo_success.jsonl"
THREAD_ID = "thread-signals"
DEMO_THREAD_ID = "thread-demo-synthetic"

CLASSIFY_TEMPLATE = "Library:\n{skill_library}\n\nPolicy:\n{selection_policy}\n\nBundle:\n{evidence_bundle}\n"
RENDER_TEMPLATE = "Policy:\n{render_policy}\n\nCandidates:\n{candidates}\n\nExisting:\n{existing_skill}\n"
REVIEW_TEMPLATE = (
    "Policy:\n{review_policy}\n\nSkill:\n{skill_md}\n\nCandidates:\n{candidates}\n\n"
    "Evidence:\n{evidence}\n\nExisting:\n{existing_skill}\n"
)
REVIEW_PASS = json.dumps({"verdict": "pass", "issues": []})
REVIEW_FAIL = json.dumps(
    {
        "verdict": "fail",
        "issues": [
            {"code": "unsupported_addition", "detail": "the evidence shows no --jobs flag", "evidence_refs": []}
        ],
    }
)
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
DEMO_SKILL_NAME = "demo-csv-column-sum"
DEMO_SKILL = "\n".join(
    [
        "---",
        f"name: {DEMO_SKILL_NAME}",
        "description: Use when summing one numeric column of a CSV file and checking the total.",
        "---",
        "",
        "# CSV column sum",
        "",
        "## Inputs",
        "",
        "- `<csv-path>`, `<column>` and `<expected-total>`.",
        "",
        "## Workflow",
        "",
        '1. Run `python3 -c "import csv,sys; print(sum(float(r[sys.argv[2]]) for r in csv.DictReader(open(sys.argv[1]))))"'
        " <csv-path> <column>`.",
        "2. Compare the printed total with `<expected-total>`.",
        "",
        "## Outputs",
        "",
        "The column total, checked.",
        "",
    ]
)
ONE_STEP_SKILL = "\n".join(
    [
        "---",
        "name: profile-without-force",
        "description: Use when re-running the profiler without discarding the saved profile.",
        "---",
        "",
        "# Profile without force",
        "",
        "## Inputs",
        "",
        "- The profiler command.",
        "",
        "## Workflow",
        "",
        "1. Run the profiler without `--force`.",
        "",
        "## Constraints",
        "",
        "- Never use `--force`; the evidence shows it deleted the profile.",
        "",
        "## Outputs",
        "",
        "The saved profile is kept.",
        "",
    ]
)
REPORT_KEYS = {
    "report_version",
    "command",
    "generated_at",
    "thread_id",
    "source",
    "synthetic",
    "policy",
    "config",
    "prompts",
    "episodes",
    "gate",
    "second_gate",
    "bundles",
    "stages",
    "classifier",
    "code_rejections",
    "plans",
    "quality_review",
    "coverage",
    "proposals",
    "llm",
    "stop_message",
    "failed",
    "evidence_text_file",
    "rejected_draft_files",
}


@pytest.fixture(autouse=True)
def _exgraph_off(monkeypatch: pytest.MonkeyPatch):
    """The stored-graph appendix would change prompt sizes; keep the context arithmetic exact."""
    monkeypatch.setenv(ENV_DISABLED, "1")
    reset_config_cache()
    yield
    reset_config_cache()


class _NullStatus:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Sink:
    """Records the themed console calls; muted lines land in ``plain`` with their markup."""

    def __init__(self) -> None:
        self.info: list[str] = []
        self.success: list[str] = []
        self.warning: list[str] = []
        self.error: list[str] = []
        self.plain: list[str] = []
        self.console = SimpleNamespace(status=lambda *_a, **_k: _NullStatus())

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


# ---------------------------------------------------------------- builders


def _prompts(classify: str = CLASSIFY_TEMPLATE, *, contract_version: int = 2) -> StagePrompts:
    texts = {"classify": classify, "render": RENDER_TEMPLATE, "review": REVIEW_TEMPLATE}
    return StagePrompts(
        **{
            stage: PromptText(stage, text, f"fake/{stage}", prompt_sha256(text), contract_version)
            for stage, text in texts.items()
        }
    )


def _thread(path: Path = SIGNALS, *, demo: bool = False) -> ThreadInput:
    trajectory = load_trajectory(path)
    notes: list[DetectorNote] = []
    episodes = extract_episodes(trajectory, demo=demo, notes=notes)
    return ThreadInput(trajectory, [], episodes, notes)


def _refs(thread: ThreadInput, *, demo: bool = False, count: int = 2) -> list[str]:
    bundle = build_evidence_bundle(thread.episodes, [thread.trajectory], demo=demo)
    return sorted(bundle.shown)[:count]


def _decision(episode_ids: list[str], decision: str, reason_code: str) -> dict[str, Any]:
    return {
        "episode_ids": episode_ids,
        "decision": decision,
        "reason_code": reason_code,
        "explanation": "scripted",
        "evidence_refs": [],
    }


def _classify_reply(
    *candidates: dict[str, Any], verdict: str = "save", decisions: list[dict[str, Any]] | None = None
) -> str:
    if decisions is None:
        decisions = [
            _decision(["E1"], "accept", "accepted") if candidates else _decision(["E1"], "reject", "routine_activity")
        ]
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


def _library_skill(root: Path, name: str, category: str = "default") -> Skill:
    skill_dir = root / "library" / category / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: Use when testing.\n---\nold body\n", encoding="utf-8")
    return Skill(name=name, description="Use when testing.", category=category, path=path)


class _Harness:
    """One RunContext over a scripted LLM; the state dir receives the decision reports."""

    def __init__(
        self,
        tmp_path: Path,
        fake_llm_cls,
        *replies: str,
        cfg: DirectSkillGenerationConfig | None = None,
        skills: list[Skill] | None = None,
        context_window: int | None = 128_000,
        report: bool = True,
        prompts: StagePrompts | None = None,
    ) -> None:
        cfg = cfg or DirectSkillGenerationConfig()
        skills = skills or []
        self.root = tmp_path
        self.state = tmp_path / "state"
        self.sink = _Sink()
        self.llm = fake_llm_cls(*replies)
        self.run = RunContext(
            command="direct-skill-generation",
            working_dir=tmp_path,
            state_dir=self.state,
            requested=cfg,
            rules=effective_rules(cfg, working_dir=tmp_path),
            prompts=prompts or _prompts(),
            skills=skills,
            llm_slot=LazyLlm(session=None, llm=self.llm, model="fake-model", context_window=context_window),
            taken={s.name for s in skills} | {s.display_name for s in skills},
            sink=self.sink,
            report_dir=decisions_dir(self.state) if report else None,
        )

    async def thread(self, thread: ThreadInput) -> ThreadResult:
        return await run_thread(self.run, thread)

    def payload(self, index: int) -> str:
        """The first human message of the ``index``-th LLM call."""
        return self.llm.payloads[index][0][1]

    def reports(self) -> list[dict[str, Any]]:
        folder = self.state / "skill-evolver" / "decisions"
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]

    def proposal(self, name: str, thread_id: str = THREAD_ID) -> Path:
        return self.root / "skills" / ".proposals" / thread_id / name / "SKILL.md"


def _provenance(proposal: Path) -> dict[str, Any]:
    return json.loads((proposal.parent / "provenance.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ budget


@pytest.mark.asyncio
async def test_budget_exhausted_mid_plan_writes_nothing_and_defers_the_rest(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    refs = _refs(thread)
    second = _candidate(refs, title="Profile before summary")
    # classify (1), invalid render (2), corrected render (3): the review would be call 4.
    harness = _Harness(
        tmp_path,
        fake_llm_cls,
        _classify_reply(_candidate(refs), second),
        INVALID_SKILL,
        VALID_SKILL,
        cfg=DirectSkillGenerationConfig(max_llm_calls=3),
    )

    result = await harness.thread(thread)

    assert harness.llm.replies == [] and result.llm_calls == 3 and result.llm_limit == 3
    assert result.budget_stop is True
    tally = result.tally
    assert (tally.plans, tally.proposals, tally.render_errors, tally.deferred) == (1, 0, 1, 1)
    (error,) = harness.sink.error
    assert error.startswith("plan create: Generated source debugging: LLM call budget exhausted (3/3")
    assert error.endswith("; SKILL.md not written")
    assert "Deferred plan: create: Profile before summary — llm call budget exhausted (3/3)" in harness.sink.info
    assert [rejection.code for rejection in result.code_rejections] == ["budget_exhausted"]
    assert not (tmp_path / "skills").exists()
    (report,) = harness.reports()
    assert report["llm"]["budget_exhausted"] is True and report["llm"]["calls_used"] == 3
    assert report["plans"] == {"rendered": 1, "proposals": 0, "render_errors": 1, "rejected_targets": 0, "deferred": 1}


@pytest.mark.asyncio
async def test_budget_exhausted_before_review_does_not_write_unverified_skill(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(
        tmp_path,
        fake_llm_cls,
        _classify_reply(_candidate(_refs(thread))),
        VALID_SKILL,
        cfg=DirectSkillGenerationConfig(max_llm_calls=2),
    )

    result = await harness.thread(thread)

    assert len(harness.llm.payloads) == 2 and result.budget_stop is True
    assert "LLM call budget exhausted (2/2" in harness.sink.error[0]
    assert result.quality_reviews == [] and result.proposals == [] and harness.sink.success == []
    assert not (tmp_path / "skills").exists()


@pytest.mark.asyncio
async def test_budget_exhausted_during_classify_ends_the_thread_with_nothing_written(
    tmp_path: Path, fake_llm_cls
) -> None:
    # One call allowed: the unparsable first reply asks for the corrective retry, which the budget refuses.
    thread = _thread()
    harness = _Harness(tmp_path, fake_llm_cls, "not json", cfg=DirectSkillGenerationConfig(max_llm_calls=1))

    result = await harness.thread(thread)

    assert len(harness.llm.payloads) == 1 and result.llm_calls == 1 and result.budget_stop is True
    assert result.failed is None and result.tally.plans == 0
    message = "LLM call budget exhausted (1/1 calls per thread) before classification finished; nothing written"
    assert harness.sink.error == [message] and harness.sink.info == [] and harness.sink.success == []
    assert result.stop_message == message
    assert [(r.code, r.subject) for r in result.code_rejections] == [("budget_exhausted", "classify")]
    assert ("classify", "budget_exhausted") in [(s.stage, s.status) for s in result.stages]
    (report,) = harness.reports()
    assert report["failed"] is None and report["stop_message"] == message
    assert report["llm"]["budget_exhausted"] is True and report["llm"]["calls_used"] == 1
    assert not (tmp_path / "skills").exists()


# ----------------------------------------------------------- context budget


def _window_for(available_chars: int) -> int:
    """A context window whose ContextBudget offers at least ``available_chars``."""
    return math.ceil((available_chars + REPLY_RESERVE_TOKENS * CHARS_PER_TOKEN) / (CHARS_PER_TOKEN * USABLE_RATIO))


@pytest.mark.asyncio
async def test_context_budget_cuts_bundle_once_then_classifies(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    full = build_evidence_bundle(thread.episodes, [thread.trajectory])
    template = CLASSIFY_TEMPLATE.replace("{skill_library}", module.skill_library_snapshot([])).replace(
        "{selection_policy}", selection_policy_block("strict_knowledge")
    )
    overhead = len(template) - len(BUNDLE_PLACEHOLDER)
    window = _window_for(len(full.text) // 2 + overhead)
    room = ContextBudget.for_window(window).room_for(overhead)
    assert 500 <= room < len(full.text)
    harness = _Harness(tmp_path, fake_llm_cls, _classify_reply(verdict="nothing"), context_window=window)

    result = await harness.thread(thread)

    assert [stats["max_chars"] for stats in result.bundle_stats] == [60000, room]
    assert result.bundle_stats[1]["chars"] <= room and result.bundle_stats[1]["excluded"] > 0
    # Two error_recovery exclusions after the cut are one code rejection.
    assert result.bundle_stats[1]["excluded_kinds"] == ["retry_loop", "error_recovery", "error_recovery"]
    assert [(r.code, r.subject) for r in result.code_rejections] == [
        ("insufficient_context_budget", "retry_loop"),
        ("insufficient_context_budget", "error_recovery"),
    ]
    assert (
        harness.sink.warning[0]
        == f"Bundle cut to {room} chars to fit the model context (context window {window} tokens -> {room + overhead} chars for the prompt)"
    )
    assert result.second_gate is not None and result.second_gate["passes"] is True
    assert ("context_fit", "cut") in [(stage.stage, stage.status) for stage in result.stages]
    assert len(harness.llm.payloads) == 1 and result.classifier_verdict == "nothing"
    assert len(harness.payload(0)) <= room + overhead


@pytest.mark.asyncio
async def test_context_budget_too_small_stops_before_the_llm(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(tmp_path, fake_llm_cls, context_window=1000)

    result = await harness.thread(thread)

    assert harness.llm.payloads == []
    assert result.stop_message.startswith(
        "Nothing to save: the classify prompt does not fit the model context "
        "(context window 1000 tokens -> 0 chars for the prompt); bundle needs "
    )
    assert result.stop_message.endswith(" chars, 0 available")
    assert harness.sink.info == [result.stop_message]
    assert [(r.code, r.subject) for r in result.code_rejections] == [("insufficient_context_budget", "thread")]
    (report,) = harness.reports()
    assert report["stop_message"] == result.stop_message and report["llm"]["calls_used"] == 0


# ------------------------------------------------------------------ expand


def _insufficient(*episode_ids: str) -> str:
    return _classify_reply(
        verdict="nothing", decisions=[_decision(list(episode_ids), "reject", "insufficient_evidence")]
    )


@pytest.mark.asyncio
async def test_expand_context_once_runs_one_round_only_on_new_events(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(tmp_path, fake_llm_cls, _insufficient("E1"), _insufficient("E1"))

    result = await harness.thread(thread)

    assert len(harness.llm.payloads) == 2 and harness.llm.replies == []
    assert harness.payload(1).count("[ev") > harness.payload(0).count("[ev")
    assert len(result.bundle_stats) == 2
    assert result.bundle_stats[1]["fragments"] > result.bundle_stats[0]["fragments"]
    (note, final) = harness.sink.info
    assert re.fullmatch(
        r"Insufficient evidence: one more classification round with \d+ added events \(2 per side\)", note
    )
    assert final == f"Nothing to save: insufficient evidence in thread {THREAD_ID} (classifier)"
    assert [(s.stage, s.status) for s in result.stages if s.stage == "expand"] == [("expand", "retried")]
    assert len(result.decisions) == 2 and result.llm_calls == 2


@pytest.mark.asyncio
async def test_expand_is_not_run_with_on_nothing_stop(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(tmp_path, fake_llm_cls, _insufficient("E1"), cfg=DirectSkillGenerationConfig(on_nothing="stop"))

    result = await harness.thread(thread)

    assert len(harness.llm.payloads) == 1
    assert harness.sink.info == [f"Nothing to save: insufficient evidence in thread {THREAD_ID} (classifier)"]
    assert not any(stage.stage == "expand" for stage in result.stages)


@pytest.mark.asyncio
async def test_expand_is_not_run_without_insufficient_evidence(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(tmp_path, fake_llm_cls, _classify_reply(verdict="nothing"))

    result = await harness.thread(thread)

    assert len(harness.llm.payloads) == 1
    assert harness.sink.info == [f"Nothing to save: no durable learning found in thread {THREAD_ID}"]
    assert not any(stage.stage == "expand" for stage in result.stages)


@pytest.mark.asyncio
async def test_expand_is_skipped_when_no_new_context_can_be_shown(tmp_path: Path, fake_llm_cls) -> None:
    # With no surrounding events an error_recovery episode (results and task
    # already cited) gains nothing, so the round is not spent.
    thread = _thread()
    bundle = build_evidence_bundle(thread.episodes, [thread.trajectory])
    flagged = next(eid for eid, episode in bundle.episode_ids.items() if episode.kind == "error_recovery")
    harness = _Harness(
        tmp_path, fake_llm_cls, _insufficient(flagged), cfg=DirectSkillGenerationConfig(surrounding_events=0)
    )

    result = await harness.thread(thread)

    assert len(harness.llm.payloads) == 1
    assert harness.sink.info == [
        module.EXPAND_NOT_RETRIED,
        f"Nothing to save: insufficient evidence in thread {THREAD_ID} (classifier)",
    ]
    assert [(s.stage, s.status) for s in result.stages if s.stage == "expand"] == [("expand", "skipped")]
    assert len(result.bundle_stats) == 1


@pytest.mark.asyncio
async def test_expand_is_not_retried_when_the_cut_bundle_shows_no_new_context(tmp_path: Path, fake_llm_cls) -> None:
    # Round 1 fits the window exactly. Expanding the lightest episode makes the
    # probe a strict superset, but cut back to the room it shows the same
    # fragments as round 1: the retry is decided on that bundle, not the probe.
    thread = _thread()
    excerpt_chars = effective_rules(DirectSkillGenerationConfig(), working_dir=tmp_path).excerpt_max_chars
    full = build_evidence_bundle(thread.episodes, [thread.trajectory], excerpt_chars=excerpt_chars)
    flagged = list(full.episode_ids)[-1]
    assert full.episode_ids[flagged].kind == "error_recovery"
    template = CLASSIFY_TEMPLATE.replace("{skill_library}", module.skill_library_snapshot([])).replace(
        "{selection_policy}", selection_policy_block("strict_knowledge")
    )
    overhead = len(template) - len(BUNDLE_PLACEHOLDER)
    window = _window_for(len(full.text) + overhead)
    room = ContextBudget.for_window(window).room_for(overhead)
    assert len(full.text) <= room
    harness = _Harness(tmp_path, fake_llm_cls, _insufficient(flagged), context_window=window)

    result = await harness.thread(thread)

    assert len(harness.llm.payloads) == 1
    assert harness.sink.info == [
        module.EXPAND_NOT_RETRIED,
        f"Nothing to save: insufficient evidence in thread {THREAD_ID} (classifier)",
    ]
    assert [(s.stage, s.status, s.detail) for s in result.stages if s.stage == "expand"] == [
        ("expand", "skipped", "no new context fits the model context")
    ]
    assert [stats["max_chars"] for stats in result.bundle_stats] == [60000, 60000, room]
    assert result.bundle_stats[1]["fragments"] > result.bundle_stats[0]["fragments"]
    assert result.bundle_stats[2]["fragments"] == result.bundle_stats[0]["fragments"]
    (warning,) = harness.sink.warning
    assert warning.startswith(f"Bundle cut to {room} chars to fit the model context")


# ------------------------------------------------------------- nothing causes


@pytest.mark.asyncio
async def test_nothing_messages_distinguish_causes(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    stop = DirectSkillGenerationConfig(on_nothing="stop")

    classifier = _Harness(tmp_path / "a", fake_llm_cls, _insufficient("E1"), cfg=stop)
    await classifier.thread(thread)
    assert classifier.sink.info == [f"Nothing to save: insufficient evidence in thread {THREAD_ID} (classifier)"]

    invalid = _Harness(tmp_path / "b", fake_llm_cls, _classify_reply(_candidate(["ev9999"])), cfg=stop)
    result = await invalid.thread(thread)
    assert invalid.sink.warning == [
        "Rejected 'Generated source debugging': evidence not shown in the bundle: ['ev9999']"
    ]
    assert invalid.sink.info == [
        f"Nothing to save: no candidate with valid evidence refs in thread {THREAD_ID} (invalid_evidence)"
    ]
    assert [(r.code, r.subject) for r in result.code_rejections] == [("invalid_evidence", "Generated source debugging")]

    routine = _Harness(tmp_path / "c", fake_llm_cls, _classify_reply(verdict="nothing"), cfg=stop)
    await routine.thread(thread)
    assert routine.sink.info == [f"Nothing to save: no durable learning found in thread {THREAD_ID}"]
    assert "Decision E1: reject (routine_activity)" in routine.sink.text()


# -------------------------------------------------------------------- demo


@pytest.mark.asyncio
async def test_demo_overlay_passes_both_gates_and_marks_the_proposal(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread(DEMO, demo=True)
    refs = _refs(thread, demo=True, count=7)
    candidate = _candidate(
        refs, title="CSV column sum", rule="Sum a CSV column with csv.DictReader and check the total."
    )
    harness = _Harness(
        tmp_path,
        fake_llm_cls,
        _classify_reply(candidate),
        DEMO_SKILL,
        REVIEW_PASS,
        cfg=DirectSkillGenerationConfig(demo_mode=True),
    )

    result = await harness.thread(thread)

    assert harness.sink.error == [] and harness.sink.warning == []
    assert result.gate["passes"] is True and result.gate["reason"] == "demo_override" and result.gate["score"] == 0.0
    text = harness.sink.text()
    assert "Evidence score: 0.00 (min_evidence_score 1.00; 1 episodes, 1 incidents)" in text
    assert "Gate: pass (demo_override; observed_procedure has required evidence)" in text
    assert "Decision E1: accept (accepted)" in text
    assert "Candidate 'CSV column sum': accepted (trivial procedure allowed)" in text
    assert "Quality review: passed" in text and "Proposal: saved, inactive, DEMO" in text
    assert "# Selection policy: demo_workflow" in harness.payload(0)
    assert "Selection policy: demo." in harness.payload(1) and "Review policy: demo." in harness.payload(2)
    proposal = harness.proposal(DEMO_SKILL_NAME, DEMO_THREAD_ID)
    assert proposal.is_file() and proposal.read_text(encoding="utf-8") == DEMO_SKILL
    provenance = _provenance(proposal)
    assert provenance["demo"] is True
    assert provenance["policy"] == {"requested": "strict_knowledge", "selection": "demo_workflow", "source": "config"}
    assert provenance["target"]["action"] == "create" and provenance["covering_skill"] is None
    assert provenance["observed_procedure_source"] and provenance["config"]["effective"]["demo"] is True
    (report,) = harness.reports()
    assert set(report) == REPORT_KEYS
    assert report["synthetic"] is True and report["policy"]["demo_mode"] is True
    assert report["gate"]["reason"] == "demo_override" and report["second_gate"] is None
    assert report["episodes"] == {"total": 1, "counts": {"observed_procedure": 1}, "detector_notes": []}
    assert report["proposals"] == [str(proposal)] and report["quality_review"][0]["verdict"] == "pass"


@pytest.mark.asyncio
async def test_demo_fixture_without_demo_makes_no_llm_call(tmp_path: Path, fake_llm_cls) -> None:
    plain = _Harness(tmp_path / "a", fake_llm_cls)
    result = await plain.thread(_thread(DEMO))
    assert plain.llm.payloads == [] and result.gate["passes"] is False
    assert plain.sink.info == [f"Nothing to save: no episodes detected in thread {DEMO_THREAD_ID}"]

    # Even with the observed procedure extracted, the ordinary rules refuse it.
    gated = _Harness(tmp_path / "b", fake_llm_cls)
    result = await gated.thread(_thread(DEMO, demo=True))
    assert gated.llm.payloads == [] and result.gate["reason"] == "score < min_evidence_score"
    assert gated.sink.info == [
        "Nothing to save: evidence score 0.00 < min_evidence_score 1.00 (1 episodes, 1 incidents)"
    ]
    assert "Gate: skip (score < min_evidence_score)" in gated.sink.text()
    (report,) = gated.reports()
    assert report["stop_message"] == gated.sink.info[0] and report["llm"]["calls_used"] == 0


# ---------------------------------------------------------------- review


@pytest.mark.asyncio
async def test_quality_review_allows_one_correction(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(
        tmp_path,
        fake_llm_cls,
        _classify_reply(_candidate(_refs(thread))),
        VALID_SKILL,
        REVIEW_FAIL,
        VALID_SKILL,
        REVIEW_PASS,
    )

    result = await harness.thread(thread)

    assert len(harness.llm.payloads) == 5 and harness.sink.error == []
    assert result.tally.proposals == 1 and harness.proposal(SKILL_NAME).is_file()
    assert "Quality review: passed after one correction" in harness.sink.text()
    assert "the evidence shows no --jobs flag" in harness.llm.payloads[3][-1][1]
    provenance = _provenance(harness.proposal(SKILL_NAME))
    assert provenance["quality_review"]["corrected"] is True
    assert provenance["quality_review"]["initial_issues"] == ["unsupported_addition: the evidence shows no --jobs flag"]
    assert result.quality_reviews == [{"plan": f"create: {_candidate([])['title']}", **provenance["quality_review"]}]


@pytest.mark.asyncio
async def test_quality_review_blocks_after_one_correction(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(
        tmp_path,
        fake_llm_cls,
        _classify_reply(_candidate(_refs(thread))),
        VALID_SKILL,
        REVIEW_FAIL,
        VALID_SKILL,
        REVIEW_FAIL,
    )

    result = await harness.thread(thread)

    assert len(harness.llm.payloads) == 5 and result.tally.render_errors == 1
    assert harness.sink.error == [
        "plan create: Generated source debugging: quality review failed after one correction; nothing written:",
        "  - unsupported_addition: the evidence shows no --jobs flag",
    ]
    assert [rejection.code for rejection in result.code_rejections] == ["quality_review_failed"]
    assert not (tmp_path / "skills").exists()
    (report,) = harness.reports()
    assert report["quality_review"][0]["verdict"] == "fail" and report["plans"]["render_errors"] == 1
    # The first review's findings survive the failed correction, and the corrected draft is
    # kept next to the report — in the private state, never under skills/.
    assert report["quality_review"][0]["initial_issues"] == ["unsupported_addition: the evidence shows no --jobs flag"]
    assert report["quality_review"][0]["corrected"] is False
    (draft_name,) = report["rejected_draft_files"]
    draft = result.report_path.with_name(draft_name)
    assert result.rejected_draft_paths == [draft]
    assert draft.name.endswith(".draft-1-create-generated-source-debugging.rejected.md")
    text = draft.read_text(encoding="utf-8")
    assert text.startswith(VALID_SKILL.strip()) and text.rstrip().endswith("never a proposal -->")
    assert "quality_review_failed" in text


@pytest.mark.asyncio
async def test_rejected_draft_is_not_kept_when_disabled(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(
        tmp_path,
        fake_llm_cls,
        _classify_reply(_candidate(_refs(thread))),
        VALID_SKILL,
        REVIEW_FAIL,
        VALID_SKILL,
        REVIEW_FAIL,
        cfg=DirectSkillGenerationConfig(save_rejected_drafts=False),
    )

    result = await harness.thread(thread)

    assert result.tally.render_errors == 1 and result.rejected_draft_paths == []
    (report,) = harness.reports()
    assert report["rejected_draft_files"] == []
    assert list(result.report_path.parent.glob("*.rejected.md")) == []


@pytest.mark.asyncio
async def test_quality_reviewer_invalid_twice_writes_nothing(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(
        tmp_path, fake_llm_cls, _classify_reply(_candidate(_refs(thread))), VALID_SKILL, "not json", "still not json"
    )

    result = await harness.thread(thread)

    assert len(harness.llm.payloads) == 4 and result.tally.render_errors == 1
    (error,) = harness.sink.error
    assert error.startswith(
        "plan create: Generated source debugging: quality reviewer reply invalid; nothing written: "
    )
    assert "after 2 attempt(s)" in error
    assert not (tmp_path / "skills").exists()


# ------------------------------------------------------------- failures/report


@pytest.mark.asyncio
async def test_run_thread_writes_report_even_when_failing_and_reraises(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(tmp_path, fake_llm_cls, "not json", "still not json")

    with pytest.raises(ClassifyParseError, match="after one retry"):
        await harness.thread(thread)

    (report,) = harness.reports()
    assert report["failed"].startswith("ClassifyParseError: classify: reply is not valid JSON after one retry")
    assert [(r["code"], r["subject"]) for r in report["code_rejections"]] == [("parse_error", "classify")]
    assert {"stage": "classify", "status": "failed", "detail": report["code_rejections"][0]["detail"]} in report[
        "stages"
    ]
    assert report["llm"]["calls_used"] == 2 and report["proposals"] == []
    assert harness.sink.info == []


@pytest.mark.asyncio
async def test_classify_template_without_policy_placeholder_is_an_error(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    prompts = _prompts(classify="Library:\n{skill_library}\n\nBundle:\n{evidence_bundle}\n")
    harness = _Harness(tmp_path, fake_llm_cls, prompts=prompts)

    with pytest.raises(ValueError, match=re.escape("classify template has no {selection_policy} placeholder")):
        await harness.thread(thread)

    assert harness.llm.payloads == []
    (report,) = harness.reports()
    assert report["failed"].startswith("ValueError: classify template has no")


@pytest.mark.asyncio
async def test_no_report_is_written_without_a_report_dir(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(tmp_path, fake_llm_cls, _classify_reply(verdict="nothing"), report=False)

    result = await harness.thread(thread)

    assert result.report_path is None and not (tmp_path / "state").exists()


# -------------------------------------------------------------- policy/prompts


@pytest.mark.asyncio
async def test_classify_payload_gets_exactly_one_policy_block_and_hashes_recorded(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    skills = [_library_skill(tmp_path, "real")]
    harness = _Harness(
        tmp_path,
        fake_llm_cls,
        _classify_reply(_candidate(_refs(thread))),
        VALID_SKILL,
        REVIEW_PASS,
        cfg=DirectSkillGenerationConfig(save_evidence_text=True),
        skills=skills,
    )

    result = await harness.thread(thread)

    payload = harness.payload(0)
    assert payload.count("# Selection policy:") == 1
    assert selection_policy_block("strict_knowledge") in payload
    assert "{selection_policy}" not in payload and "{skill_library}" not in payload
    assert "- real: Use when testing." in payload
    assert "Selection policy: ordinary." in harness.payload(1) and "Review policy: ordinary." in harness.payload(2)
    hashes = harness.run.prompts.hashes()
    assert set(hashes) == {"classify", "render", "review"} and all(h.startswith("sha256:") for h in hashes.values())
    assert _provenance(harness.proposal(SKILL_NAME))["prompt_hashes"] == hashes
    (report,) = harness.reports()
    assert report["prompts"] == {"contract_version": 2, "variants": harness.run.prompts.variants(), "hashes": hashes}
    evidence = result.report_path.with_name(result.report_path.stem + ".evidence.md")
    assert report["evidence_text_file"] == evidence.name
    assert evidence.read_text(encoding="utf-8").startswith("# Round 1\n\n### Episode E1")


@pytest.mark.asyncio
async def test_provenance_and_report_record_the_prompts_contract_from_one_source(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    harness = _Harness(
        tmp_path,
        fake_llm_cls,
        _classify_reply(_candidate(_refs(thread))),
        VALID_SKILL,
        REVIEW_PASS,
        prompts=_prompts(contract_version=7),
    )

    await harness.thread(thread)

    (report,) = harness.reports()
    provenance = _provenance(harness.proposal(SKILL_NAME))
    assert provenance["versions"]["prompts_contract"] == report["prompts"]["contract_version"] == 7


@pytest.mark.asyncio
async def test_reusable_workflow_policy_block_and_provenance(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    refs = _refs(thread)
    configured = _Harness(
        tmp_path / "a",
        fake_llm_cls,
        _classify_reply(_candidate(refs)),
        VALID_SKILL,
        REVIEW_PASS,
        cfg=DirectSkillGenerationConfig(policy="reusable_workflow"),
    )
    await configured.thread(thread)
    payload = configured.payload(0)
    assert "# Selection policy: reusable_workflow" in payload and "# Selection policy: strict_knowledge" not in payload
    assert _provenance(configured.proposal(SKILL_NAME))["policy"] == {
        "requested": "reusable_workflow",
        "selection": "reusable_workflow",
        "source": "config",
    }

    overridden = _Harness(
        tmp_path / "b",
        fake_llm_cls,
        _classify_reply(_candidate(refs)),
        VALID_SKILL,
        REVIEW_PASS,
        cfg=apply_overrides(DirectSkillGenerationConfig(), policy="reusable_workflow"),
    )
    await overridden.thread(thread)
    assert _provenance(overridden.proposal(SKILL_NAME))["policy"]["source"] == "--policy"
    (report,) = overridden.reports()
    assert report["policy"]["overrides"] == ["--policy reusable_workflow"]


@pytest.mark.asyncio
async def test_single_step_skill_with_evidence_backed_prohibition_is_written(tmp_path: Path, fake_llm_cls) -> None:
    thread = _thread()
    candidate = _candidate(
        _refs(thread),
        title="Profile without force",
        rule="Run the profiler without --force.",
        constraints=["Never use --force; it deleted the saved profile."],
    )
    harness = _Harness(tmp_path, fake_llm_cls, _classify_reply(candidate), ONE_STEP_SKILL, REVIEW_PASS)

    result = await harness.thread(thread)

    assert harness.sink.error == [] and result.tally.proposals == 1
    assert not any(REJECTED in line for line in harness.sink.error)
    proposal = harness.proposal("profile-without-force")
    assert proposal.read_text(encoding="utf-8") == ONE_STEP_SKILL
    assert "Never use --force" in harness.payload(1)


# -------------------------------------------------------- helpers (unit)


def test_read_existing_skill_hashes_and_truncates(tmp_path: Path) -> None:
    skill = _library_skill(tmp_path, "real")
    content = skill.path.read_bytes()

    text = read_existing_skill(skill)
    assert text.display_name == "real" and text.path == skill.path
    assert text.sha256 == hashlib.sha256(content).hexdigest()
    assert text.text == content.decode("utf-8") and text.truncated is False

    clipped = read_existing_skill(skill, max_chars=10)
    assert clipped.truncated is True and clipped.sha256 == text.sha256
    assert clipped.text == content.decode("utf-8")[:10] + f"\n… [truncated: {len(content) - 10} more characters]"

    assert read_existing_skill(Skill(name="ghost", description="", category="default", path=tmp_path / "no")) is None


def test_resolve_library_skill_and_covering_skill_record(tmp_path: Path) -> None:
    alpha = _library_skill(tmp_path, "alpha")
    beta_profiler = _library_skill(tmp_path, "beta", category="profiler")
    beta_modeling = _library_skill(tmp_path, "beta", category="modeling")
    skills = [alpha, beta_profiler, beta_modeling]

    assert resolve_library_skill("alpha", skills) is alpha
    assert resolve_library_skill("beta", skills) is None
    assert resolve_library_skill("profiler/beta", skills) is beta_profiler
    assert resolve_library_skill(None, skills) is None and resolve_library_skill(" ", skills) is None

    assert covering_skill_record(None, skills) == (None, None)
    record, warning = covering_skill_record("alpha", skills)
    assert warning is None
    assert record == {
        "display_name": "alpha",
        "path": str(alpha.path),
        "sha256": hashlib.sha256(alpha.path.read_bytes()).hexdigest(),
    }
    record, warning = covering_skill_record("ghost", skills)
    assert record == {"display_name": "ghost", "path": None, "sha256": None}
    assert warning == "covered_by 'ghost' names no library skill; recorded as the classifier's claim"


def test_reference_coverage_reads_text_or_marks_unverified(tmp_path: Path) -> None:
    from msagent.skill_evolver.classify import Candidate

    real = _library_skill(tmp_path, "real")
    ghost = Skill(name="ghost", description="", category="default", path=tmp_path / "missing" / "SKILL.md")
    candidate = Candidate.model_validate(_candidate(["ev1"], title="Seen"))

    rows = reference_coverage([(candidate, real), (candidate, ghost)])

    assert [(row["skill"], row["title"], row["coverage"]) for row in rows] == [
        ("real", "Seen", "text_read"),
        ("ghost", "Seen", "unverified"),
    ]
    assert rows[0]["sha256"] == hashlib.sha256(real.path.read_bytes()).hexdigest() and rows[1]["sha256"] is None


@pytest.mark.asyncio
async def test_load_prompts_hashes_the_returned_text_and_records_the_config_contract(tmp_path: Path) -> None:
    async def loader(_root: Path, _cfg: DirectSkillGenerationConfig, stage: str) -> tuple[str, str]:
        return f"text of {stage}", f"user/{stage}/prompt_v2.md"

    prompts = await module.load_prompts(loader, tmp_path, DirectSkillGenerationConfig())

    assert prompts.variants() == {stage: f"user/{stage}/prompt_v2.md" for stage in ("classify", "render", "review")}
    assert prompts.hashes() == {stage: prompt_sha256(f"text of {stage}") for stage in ("classify", "render", "review")}
    assert prompts.review.contract_version == 2


def test_print_policy_block_and_config_error_lines() -> None:
    from msagent.skill_evolver.config import SkillEvolverConfigError
    from msagent.skill_evolver.prompts import PromptContractError

    sink = _Sink()
    cfg = apply_overrides(DirectSkillGenerationConfig(notes=("active: legacy replay field, ignored",)), demo=True)
    module.print_policy_block(sink, cfg, effective_rules(cfg, working_dir=Path("/w")))
    assert sink.plain == [
        "[muted]Requested policy: strict_knowledge (config.skill.evolver.yml)[/muted]",
        "[muted]Demo mode: true (--demo)[/muted]",
        "[muted]Effective selection: demo_workflow[/muted]",
        f"[muted]{module.DEMO_OVERLAY_LINE}[/muted]",
        "[muted]Config note: active: legacy replay field, ignored[/muted]",
    ]

    sink = _Sink()
    module.report_config_error(sink, SkillEvolverConfigError([("gate.min_evidence_score", "required")], file=None))
    assert sink.error == ["config: gate.min_evidence_score: required"]
    assert sink.plain == [f"[muted]{module.CONFIG_HINT}[/muted]"]

    sink = _Sink()
    module.report_config_error(sink, PromptContractError("x declares contract_version 1"))
    assert sink.error == ["x declares contract_version 1"] and len(sink.plain) == 1


def test_bundle_preview_lines_for_a_demo_thread() -> None:
    thread = _thread(DEMO, demo=True)
    rules = effective_rules(DirectSkillGenerationConfig(demo_mode=True), working_dir=Path("/w"))

    lines = module.bundle_preview(thread, rules)

    assert lines[0].startswith("bundle: 7 fragments, ") and lines[0].endswith(
        "(cap 60000); episodes shown 1, trimmed 0, excluded 0"
    )
    assert lines[1:] == [
        "observed_procedure: read_file → bash → bash (7 events)",
        "observed_procedure candidates: 1 selected; dropped: none",
    ]


def test_report_plans_and_rejection_lines() -> None:
    from msagent.skill_evolver.classify import Candidate
    from msagent.skill_evolver.render import RenderPlan, RenderPlans

    candidate = Candidate.model_validate(_candidate(["ev1"], title="Seen"))
    skill = Skill(name="real", description="", category="profiler", path=Path("/lib/real/SKILL.md"))
    plans = RenderPlans(plans=[], deferred=[], references=[(candidate, skill), (candidate, skill)], rejected=[])

    _warnings, notes = module.report_plans(plans, ["text_read"])

    assert notes == [
        "reference profiler/real: Seen (classifier's claim; skill text read, coverage not verified)",
        "reference profiler/real: Seen (classifier's claim; skill text unreadable, coverage not verified)",
    ]

    plan = RenderPlan(candidates=[candidate], existing=None)
    budget = module.PlanOutcome(
        plan,
        0,
        errors=["c1: required evidence ['ev1'] does not fit"],
        rejection=CodeRejection("insufficient_context_budget", plan.label, "x"),
    )
    assert module.rejection_lines(plan.label, budget) == [
        "plan create: Seen: required evidence does not fit the render budget (insufficient_context_budget); nothing written:",
        "  - c1: required evidence ['ev1'] does not fit",
    ]
    secrets = module.PlanOutcome(
        plan,
        2,
        rejection=CodeRejection(
            "secrets_detected",
            plan.label,
            "proposal package contains possible secrets; nothing written: SKILL.md:3: possible secret (jwt)",
        ),
    )
    assert module.rejection_lines(plan.label, secrets) == [
        "plan create: Seen: proposal package contains possible secrets; nothing written: SKILL.md:3: possible secret (jwt)",
    ]
