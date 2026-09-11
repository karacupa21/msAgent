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

"""Tests for the generation stage on a scripted fake LLM (no network)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pytest

from msagent.skill_evolver.bundle import ShownFragment
from msagent.skill_evolver.classify import EMPTY_REPLY, Candidate
from msagent.skill_evolver.generate import (
    AMBIGUOUS_TARGET,
    CANDIDATES_PLACEHOLDER,
    EVIDENCE_LINE_OVERHEAD,
    EXISTING_SKILL_PLACEHOLDER,
    GENERATION_POLICY_PLACEHOLDER,
    INVALID_TARGET,
    NO_EXISTING_SKILL,
    EvidenceSelection,
    GenerationPlan,
    GenerationPlans,
    InsufficientContextBudget,
    format_candidates,
    format_existing_skill,
    generate_skill_md,
    plan_generation,
    resolve_library_skill,
    revise_skill_md,
    select_generation_evidence,
    select_plan_evidence,
)
from msagent.skills.factory import Skill
from msagent.trajectory_recorder.model import EvidenceRef

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = (
    REPO_ROOT / "resources" / "configs" / "default" / "skill-evolver" / "prompts" / "generate" / "prompt_v2.md"
)
LOGGER = "msagent.skill_evolver.generate"

TEMPLATE = (
    "Generate.\n\n# Policy\n\n{generation_policy}\n\n# Candidates\n\n{candidates}\n\n"
    "# Existing\n\n{existing_skill}\n\nReply with SKILL.md."
)
POLICY = "Selection policy: test. Name it after the task class."
VALID = "\n".join(
    [
        "---",
        "name: build-before-test",
        "description: Use when the test suite needs generated code.",
        "---",
        "",
        "# Build before test",
        "",
        "## Inputs",
        "",
        "The repository.",
        "",
        "## Workflow",
        "",
        "1. Run make.",
        "2. Run the tests.",
        "",
        "## Outputs",
        "",
        "A green run.",
        "",
    ]
)
INVALID = "---\nname: fix-build\ndescription: Instructions for debugging\n---\n\n## Workflow\n\n1. Only one.\n"


# ------------------------------------------------------------------ builders


def _candidate(**overrides: Any) -> Candidate:
    data: dict[str, Any] = {
        "title": "Build before test",
        "rule": "Run make before invoking the test suite.",
        "evidence_refs": ["ev1", "ev2"],
        "future_applicability": "high",
        "target": {"action": "create", "existing_skill": None},
    }
    data.update(overrides)
    return Candidate.model_validate(data)


def _fragment(fragment_id: str, text: str, *, required: bool = True) -> ShownFragment:
    seq = int(fragment_id[2:])
    return ShownFragment(
        id=fragment_id,
        ref=EvidenceRef(source="thread-t.jsonl", line=seq, seq=seq),
        role="event",
        required=required,
        text=text,
    )


SHOWN = {
    "ev1": _fragment("ev1", "tool.error bash (error): CalledProcessError: exit 2 missing dep"),
    "ev2": _fragment("ev2", 'tool.start bash: {"cmd": "make deps && make"}', required=False),
    "ev3": _fragment("ev3", "tool.result bash (ok): ok built 42 targets"),
    "ev4": _fragment("ev4", 'user: "please build it"', required=False),
}


def _cost(*ids: str) -> int:
    """Budget cost of quoting the given SHOWN fragments."""
    return sum(len(SHOWN[fragment_id].text) + EVIDENCE_LINE_OVERHEAD for fragment_id in ids)


def _update(existing_skill: str, **overrides: Any) -> Candidate:
    return _candidate(target={"action": "update", "existing_skill": existing_skill}, **overrides)


def _reference(existing_skill: str, **overrides: Any) -> Candidate:
    return _candidate(target={"action": "reference", "existing_skill": existing_skill}, **overrides)


def _plan(candidates: list[Candidate], skills: list[Skill], max_plans: int = 3) -> GenerationPlans:
    return plan_generation(candidates, skills, max_plans=max_plans)


def _titles(plan: GenerationPlan) -> list[str]:
    return [candidate.title for candidate in plan.candidates]


def _skill(name: str, category: str = "default") -> Skill:
    return Skill(
        name=name,
        description="Use when testing.",
        category=category,
        path=Path("/tmp") / category / name / "SKILL.md",
    )


# ---------------------------------------------------------------- formatting


def test_format_candidates() -> None:
    text = format_candidates([_candidate(), _update("real", title="Second", future_applicability="low")])

    assert text == "\n".join(
        [
            "1. Build before test (future applicability: high)",
            "   Rule: Run make before invoking the test suite.",
            "   Target: create a new skill",
            "2. Second (future applicability: low)",
            "   Rule: Run make before invoking the test suite.",
            "   Target: update `real`",
        ]
    )


def test_format_candidates_with_conditions_and_evidence() -> None:
    candidate = _candidate(
        evidence_refs=["ev2", "ev1"],
        applies_when="the test suite depends on generated code",
        constraints=["keep generated files out of version control", "run make once per checkout"],
        expected_outcome="a green test run without manual generation",
    )

    text = format_candidates([candidate], SHOWN)

    assert text == "\n".join(
        [
            "1. Build before test (future applicability: high)",
            "   Rule: Run make before invoking the test suite.",
            "   When: the test suite depends on generated code",
            "   Constraints:",
            "   - keep generated files out of version control",
            "   - run make once per checkout",
            "   Expected outcome: a green test run without manual generation",
            "   Target: create a new skill",
            "   Evidence:",
            "   - tool.error bash (error): CalledProcessError: exit 2 missing dep",
            '   - tool.start bash: {"cmd": "make deps && make"}',
        ]
    )


def test_select_generation_evidence_orders_required_first_without_cap() -> None:
    candidate = _candidate(evidence_refs=["ev4", "ev2", "ev3", "ev1", "ev99"])

    selection = select_generation_evidence(candidate, SHOWN)

    assert selection.ids == ["ev3", "ev1", "ev4", "ev2"]
    assert selection.complete and selection.usable
    assert selection.omitted == [] and selection.missing_required == []
    assert select_generation_evidence(candidate, {}) == EvidenceSelection([], [], [])


def test_select_plan_evidence_drops_optional_first_under_budget() -> None:
    candidate = _candidate(evidence_refs=["ev4", "ev2", "ev3", "ev1"])

    (selection,) = select_plan_evidence([candidate], SHOWN, budget_chars=_cost("ev3", "ev1", "ev4"))

    assert selection.ids == ["ev3", "ev1", "ev4"]
    assert selection.omitted == ["ev2"]
    assert selection.missing_required == []
    assert selection.usable and not selection.complete


def test_select_plan_evidence_reports_every_missing_required() -> None:
    candidate = _candidate(evidence_refs=["ev4", "ev2", "ev3", "ev1"])

    (selection,) = select_plan_evidence([candidate], SHOWN, budget_chars=5)

    assert selection.ids == []
    assert selection.missing_required == ["ev3", "ev1"]
    assert selection.omitted == ["ev4", "ev2"]
    assert not selection.usable


def test_select_plan_evidence_places_all_required_before_any_optional() -> None:
    first = _candidate(evidence_refs=["ev2", "ev1"])
    second = _candidate(title="Second", evidence_refs=["ev4", "ev3"])
    # Room for both required fragments and the first candidate's optional one only.
    budget = _cost("ev1", "ev3", "ev2")

    one, two = select_plan_evidence([first, second], SHOWN, budget_chars=budget)

    assert one.ids == ["ev1", "ev2"] and one.omitted == []
    assert two.ids == ["ev3"] and two.omitted == ["ev4"]
    assert one.usable and two.usable
    # A fragment cited by two candidates is costed and quoted for each.
    twin = _candidate(title="Twin", evidence_refs=["ev1"])
    both = select_plan_evidence([first, twin], SHOWN, budget_chars=_cost("ev1"))
    assert both[0].ids == ["ev1"] and both[1].missing_required == ["ev1"]


def test_format_candidates_shows_covered_by_and_omitted_count() -> None:
    candidate = _candidate(evidence_refs=["ev1", "ev2", "ev4"], covered_by="csv-column-sum")
    selections = select_plan_evidence([candidate], SHOWN, budget_chars=_cost("ev1", "ev2"))

    text = format_candidates([candidate], SHOWN, selections=selections)

    assert text == "\n".join(
        [
            "1. Build before test (future applicability: high)",
            "   Rule: Run make before invoking the test suite.",
            "   Target: create a new skill",
            "   Covered by library skill: csv-column-sum (write a separate teaching skill; do not copy the library text)",
            "   Evidence:",
            "   - tool.error bash (error): CalledProcessError: exit 2 missing dep",
            '   - tool.start bash: {"cmd": "make deps && make"}',
            "   - (1 context excerpts omitted for the prompt budget)",
        ]
    )
    # Unbudgeted: every cited fragment, no omitted line, no covered-by line.
    plain = format_candidates([_candidate(evidence_refs=["ev1", "ev2", "ev4"])], SHOWN)
    assert plain.count("   - ") == 3 and "omitted" not in plain and "Covered by" not in plain


def test_resolve_library_skill() -> None:
    profiler = _skill("real", category="profiler")
    modeling = _skill("real", category="modeling")
    other = _skill("other")

    assert resolve_library_skill("profiler/real", [profiler, modeling, other]) is profiler
    assert resolve_library_skill("other", [profiler, modeling, other]) is other
    assert resolve_library_skill("real", [profiler, modeling]) is None
    assert resolve_library_skill("ghost", [other]) is None
    assert resolve_library_skill(None, [other]) is None and resolve_library_skill("  ", [other]) is None


def test_generation_payload_has_no_evidence_ids() -> None:
    candidate = _candidate(evidence_refs=["ev1", "ev2", "ev3", "ev4"])

    text = format_candidates([candidate], SHOWN)

    assert "Evidence:" in text
    assert not re.search(r"\bev\d+\b", text)
    assert not re.search(r"\bseq\b", text)


def test_format_existing_skill() -> None:
    text = format_existing_skill("profiler/real", "---\nname: real\n---\nbody\n\n")

    assert text.startswith("The candidates update the existing skill `profiler/real`; keep its name.")
    assert text.endswith("Current text:\n\n---\nname: real\n---\nbody\n")


# ----------------------------------------------------------- plan_generation


def test_plan_create_only() -> None:
    plans = _plan([_candidate()], [_skill("real")])

    (plan,) = plans.plans
    assert _titles(plan) == ["Build before test"]
    assert plan.existing is None
    assert plan.label == "create: Build before test"
    assert plans.deferred == [] and plans.references == [] and plans.rejected == []


def test_plan_update_by_display_name() -> None:
    skill = _skill("real", category="profiler")

    plans = _plan([_update("profiler/real")], [skill, _skill("other")])

    (plan,) = plans.plans
    assert plan.existing is skill
    assert len(plan.candidates) == 1
    assert plan.label == "update profiler/real (1 candidate)"


def test_plan_update_by_unique_bare_name() -> None:
    skill = _skill("real", category="profiler")

    plans = _plan([_update("real")], [skill, _skill("other")])

    assert plans.plans[0].existing is skill
    assert plans.rejected == []


def test_plan_update_ambiguous_bare_name_rejected(caplog: pytest.LogCaptureFixture) -> None:
    skills = [_skill("real", category="profiler"), _skill("real", category="modeling")]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plans = _plan([_update("real")], skills)

    assert plans.plans == []
    (rejection,) = plans.rejected
    assert rejection.candidate.title == "Build before test"
    assert rejection.code == AMBIGUOUS_TARGET
    assert rejection.detail == "existing_skill 'real' is ambiguous"
    assert "dropped candidate 'Build before test'" in caplog.text


def test_plan_update_unknown_skill_rejected() -> None:
    plans = _plan([_update("ghost")], [_skill("real")])

    assert plans.plans == []
    (rejection,) = plans.rejected
    assert rejection.code == INVALID_TARGET
    assert rejection.detail == "existing_skill 'ghost' is not in the skill library"


def test_plan_reference_to_library_skill_is_informational() -> None:
    skill = _skill("real")
    candidate = _reference("real")

    plans = _plan([candidate], [skill])

    assert plans.plans == []
    assert plans.references == [(candidate, skill)]
    assert plans.rejected == []


def test_plan_reference_to_missing_skill_rejected(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plans = _plan([_reference("ghost")], [_skill("real")])

    assert plans.plans == [] and plans.references == []
    (rejection,) = plans.rejected
    assert rejection.code == INVALID_TARGET
    assert rejection.detail == "existing_skill 'ghost' is not in the skill library"
    assert "dropped candidate" in caplog.text


def test_plan_ambiguous_bare_name_rejects_update_and_reference() -> None:
    skills = [_skill("real", category="profiler"), _skill("real", category="modeling")]

    plans = _plan([_update("real", title="U"), _reference("real", title="R")], skills)

    assert plans.plans == [] and plans.references == []
    assert [(r.candidate.title, r.code) for r in plans.rejected] == [
        ("U", AMBIGUOUS_TARGET),
        ("R", AMBIGUOUS_TARGET),
    ]


def test_plan_display_name_wins_over_ambiguous_bare_name() -> None:
    profiler = _skill("real", category="profiler")
    modeling = _skill("real", category="modeling")

    plans = _plan([_update("profiler/real"), _reference("modeling/real")], [profiler, modeling])

    (plan,) = plans.plans
    assert plan.existing is profiler
    assert [skill for _, skill in plans.references] == [modeling]
    assert plans.rejected == []


def test_plan_create_and_updates_of_one_skill_are_two_plans() -> None:
    skill = _skill("real")
    candidates = [_candidate(), _update("real", title="Two"), _update("real", title="Three")]

    plans = _plan(candidates, [skill])

    assert [plan.existing for plan in plans.plans] == [None, skill]
    assert _titles(plans.plans[0]) == ["Build before test"]
    assert _titles(plans.plans[1]) == ["Two", "Three"]
    assert plans.plans[1].label == "update real (2 candidates)"


def test_plan_two_updates_of_one_skill_share_a_plan() -> None:
    skill = _skill("real")
    candidates = [_update("real", title="One"), _candidate(title="Mid"), _update("real", title="Two")]

    plans = _plan(candidates, [skill])

    # Plan order is the first appearance; later updates still join their skill.
    assert len(plans.plans) == 2
    assert plans.plans[0].existing is skill and _titles(plans.plans[0]) == ["One", "Two"]
    assert plans.plans[1].existing is None and _titles(plans.plans[1]) == ["Mid"]


def test_plan_updates_of_two_skills_are_two_plans() -> None:
    one, two = _skill("one"), _skill("two")

    plans = _plan([_update("one"), _update("two")], [one, two])

    assert [plan.existing for plan in plans.plans] == [one, two]
    assert all(len(plan.candidates) == 1 for plan in plans.plans)
    assert plans.deferred == []


def test_plan_each_create_is_its_own_plan() -> None:
    plans = _plan([_candidate(title="A"), _candidate(title="B")], [])

    assert [plan.label for plan in plans.plans] == ["create: A", "create: B"]
    assert all(plan.existing is None for plan in plans.plans)


def test_plan_over_limit_is_deferred_with_targets_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    one, two = _skill("one"), _skill("two")
    candidates = [
        _update("one", title="A1"),
        _candidate(title="New"),
        _update("two", title="B1"),
        _update("one", title="A2"),
    ]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plans = _plan(candidates, [one, two], max_plans=2)

    assert [plan.label for plan in plans.plans] == ["update one (2 candidates)", "create: New"]
    ((plan, reason),) = plans.deferred
    assert plan.existing is two and _titles(plan) == ["B1"]
    assert reason == "max_plans 2 reached"
    assert "deferred plan update two (1 candidate)" in caplog.text


def test_plan_every_candidate_lands_in_exactly_one_bucket() -> None:
    skills = [_skill("one"), _skill("two"), _skill("real")]
    candidates = [
        _candidate(candidate_id="c1"),
        _update("one", candidate_id="c2"),
        _update("two", candidate_id="c3"),
        _reference("real", candidate_id="c4"),
        _update("ghost", candidate_id="c5"),
        _reference("real", candidate_id="c6"),
    ]

    plans = _plan(candidates, skills, max_plans=1)

    ids = [c.candidate_id for plan in plans.plans for c in plan.candidates]
    ids += [c.candidate_id for plan, _ in plans.deferred for c in plan.candidates]
    ids += [c.candidate_id for c, _ in plans.references]
    ids += [r.candidate.candidate_id for r in plans.rejected]
    assert sorted(ids) == ["c1", "c2", "c3", "c4", "c5", "c6"]


def test_plan_empty() -> None:
    plans = _plan([], [])

    assert plans.plans == [] and plans.deferred == []
    assert plans.references == [] and plans.rejected == []


# --------------------------------------------------------- generate_skill_md


@pytest.mark.asyncio
async def test_valid_first_reply_single_call(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID)

    result = await generate_skill_md([_candidate()], llm=llm, policy_text=POLICY, template=TEMPLATE)

    assert result.ok and result.calls == 1
    assert result.content == VALID.strip()
    (payload,) = llm.payloads
    ((role, instruction),) = payload
    assert role == "human"
    assert format_candidates([_candidate()]) in instruction
    assert NO_EXISTING_SKILL in instruction
    assert CANDIDATES_PLACEHOLDER not in instruction
    assert EXISTING_SKILL_PLACEHOLDER not in instruction
    assert GENERATION_POLICY_PLACEHOLDER not in instruction
    assert result.transcript == [*payload, ("ai", VALID.strip())]
    assert result.generation_evidence == {"": []} and result.generation_evidence_omitted == {"": []}


@pytest.mark.asyncio
async def test_evidence_texts_reach_the_generation_payload(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID)
    candidate = _candidate(applies_when="the build is stale")

    result = await generate_skill_md([candidate], llm=llm, policy_text=POLICY, template=TEMPLATE, evidence=SHOWN)

    assert result.ok
    instruction = llm.payloads[0][0][1]
    assert "   When: the build is stale" in instruction
    assert "   - " + SHOWN["ev1"].text in instruction
    assert "   - " + SHOWN["ev2"].text in instruction
    assert "ev1" not in instruction and "ev2" not in instruction


@pytest.mark.asyncio
async def test_fenced_and_think_reply_cleaned(fake_llm_cls) -> None:
    llm = fake_llm_cls("<think>hmm</think>\n```markdown\n" + VALID + "```")

    result = await generate_skill_md([_candidate()], llm=llm, policy_text=POLICY, template=TEMPLATE)

    assert result.ok
    assert result.content == VALID.strip()


@pytest.mark.asyncio
async def test_crlf_reply_normalized(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID.replace("\n", "\r\n"))

    result = await generate_skill_md([_candidate()], llm=llm, policy_text=POLICY, template=TEMPLATE)

    assert result.ok
    assert "\r" not in result.content


@pytest.mark.asyncio
async def test_invalid_then_valid_retries_once(fake_llm_cls) -> None:
    llm = fake_llm_cls(INVALID, VALID)

    result = await generate_skill_md([_candidate()], llm=llm, policy_text=POLICY, template=TEMPLATE)

    assert result.ok and result.calls == 2
    first, second = llm.payloads
    assert second[:1] == first
    assert second[1] == ("ai", INVALID.strip())
    role, correction = second[2]
    assert role == "human"
    assert correction.startswith("Your previous reply is not a valid SKILL.md:\n- ")
    for fragment in (
        "task identifier",
        "description: must start with",
        "missing section '## Inputs'",
        "missing section '## Outputs'",
    ):
        assert fragment in correction
    assert "Workflow: needs at least" not in correction
    assert result.transcript == [*second, ("ai", VALID.strip())]


@pytest.mark.asyncio
async def test_blank_first_reply_replayed_as_empty(fake_llm_cls) -> None:
    llm = fake_llm_cls("   ", VALID)

    result = await generate_skill_md([_candidate()], llm=llm, policy_text=POLICY, template=TEMPLATE)

    assert result.ok
    assert llm.payloads[1][1] == ("ai", EMPTY_REPLY)


@pytest.mark.asyncio
async def test_invalid_twice_returns_errors_without_third_call(fake_llm_cls) -> None:
    llm = fake_llm_cls(INVALID, "no frontmatter at all")

    result = await generate_skill_md([_candidate()], llm=llm, policy_text=POLICY, template=TEMPLATE)

    assert not result.ok and result.calls == 2
    assert len(llm.payloads) == 2 and llm.replies == []
    assert result.content == "no frontmatter at all"
    assert any("no YAML frontmatter" in error for error in result.validation.errors)


@pytest.mark.asyncio
async def test_update_passes_existing_text_and_enforces_name(fake_llm_cls) -> None:
    existing = format_existing_skill("real", "---\nname: real\n---\nold body\n")
    llm = fake_llm_cls(VALID, VALID.replace("name: build-before-test", "name: real"))

    result = await generate_skill_md(
        [_update("real")], llm=llm, policy_text=POLICY, template=TEMPLATE, existing_skill=existing, expected_name="real"
    )

    assert result.ok and result.calls == 2
    assert existing in llm.payloads[0][0][1]
    assert "name: must stay 'real' (update), got 'build-before-test'" in llm.payloads[1][2][1]
    assert "name: real" in result.content


@pytest.mark.asyncio
async def test_taken_name_rejected_then_corrected(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID, VALID.replace("build-before-test", "build-first"))

    result = await generate_skill_md(
        [_candidate()], llm=llm, policy_text=POLICY, template=TEMPLATE, taken_names={"build-before-test"}
    )

    assert result.ok and result.calls == 2
    assert "already exists in the skill library" in llm.payloads[1][2][1]


@pytest.mark.asyncio
async def test_guards_raise_before_any_call(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID)

    with pytest.raises(ValueError, match="no candidates"):
        await generate_skill_md([], llm=llm, policy_text=POLICY, template=TEMPLATE)
    with pytest.raises(ValueError, match="placeholder"):
        await generate_skill_md([_candidate()], llm=llm, policy_text=POLICY, template="no placeholders here")
    with pytest.raises(ValueError, match="go together"):
        await generate_skill_md([_candidate()], llm=llm, policy_text=POLICY, template=TEMPLATE, expected_name="real")
    with pytest.raises(ValueError, match="new skills only"):
        await generate_skill_md(
            [_candidate()],
            llm=llm,
            policy_text=POLICY,
            template=TEMPLATE,
            existing_skill="x",
            expected_name="real",
            required_prefix="demo-",
        )
    assert llm.payloads == []


@pytest.mark.asyncio
async def test_generation_policy_placeholder_is_substituted_once_and_mandatory(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID)

    await generate_skill_md([_candidate()], llm=llm, template=TEMPLATE, policy_text=POLICY)

    instruction = llm.payloads[0][0][1]
    assert instruction.count(POLICY) == 1
    assert "{generation_policy}" not in instruction
    without = TEMPLATE.replace("{generation_policy}", "")
    with pytest.raises(ValueError, match=r"placeholder") as info:
        await generate_skill_md([_candidate()], llm=llm, template=without, policy_text=POLICY)
    assert "{generation_policy}" in str(info.value)
    assert len(llm.payloads) == 1


@pytest.mark.asyncio
async def test_generation_quotes_all_required_fragments_beyond_three(fake_llm_cls) -> None:
    shown = {f"ev{n}": _fragment(f"ev{n}", f"tool.result step{n} (ok): output number {n}") for n in range(1, 6)}
    candidate = _candidate(evidence_refs=list(shown), candidate_id="c1")
    llm = fake_llm_cls(VALID)

    result = await generate_skill_md([candidate], llm=llm, template=TEMPLATE, policy_text=POLICY, evidence=shown)

    instruction = llm.payloads[0][0][1]
    for fragment in shown.values():
        assert "   - " + fragment.text in instruction
    assert result.generation_evidence == {"c1": ["ev1", "ev2", "ev3", "ev4", "ev5"]}
    assert result.generation_evidence_omitted == {"c1": []}


@pytest.mark.asyncio
async def test_generation_refuses_before_llm_when_required_evidence_does_not_fit(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID)
    candidate = _candidate(evidence_refs=["ev3", "ev1", "ev2"], candidate_id="c7")

    with pytest.raises(InsufficientContextBudget) as info:
        await generate_skill_md(
            [candidate], llm=llm, template=TEMPLATE, policy_text=POLICY, evidence=SHOWN, evidence_budget_chars=5
        )

    assert info.value.candidate_id == "c7"
    assert info.value.missing == ["ev3", "ev1"]
    assert info.value.budget_chars == 5
    assert "'c7'" in str(info.value) and "5 chars" in str(info.value)
    assert llm.payloads == []


@pytest.mark.asyncio
async def test_generation_records_omitted_optional_fragments(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID)
    candidate = _candidate(evidence_refs=["ev4", "ev2", "ev3", "ev1"], candidate_id="c1")

    result = await generate_skill_md(
        [candidate],
        llm=llm,
        template=TEMPLATE,
        policy_text=POLICY,
        evidence=SHOWN,
        evidence_budget_chars=_cost("ev3", "ev1", "ev4"),
    )

    assert result.generation_evidence == {"c1": ["ev3", "ev1", "ev4"]}
    assert result.generation_evidence_omitted == {"c1": ["ev2"]}
    instruction = llm.payloads[0][0][1]
    assert SHOWN["ev2"].text not in instruction
    assert "(1 context excerpts omitted for the prompt budget)" in instruction


@pytest.mark.asyncio
async def test_required_prefix_reaches_validation_and_correction(fake_llm_cls) -> None:
    llm = fake_llm_cls(
        VALID.replace("name: build-before-test", "name: csv-column-sum"),
        VALID.replace("name: build-before-test", "name: demo-csv-column-sum"),
    )

    result = await generate_skill_md(
        [_candidate()], llm=llm, template=TEMPLATE, policy_text=POLICY, required_prefix="demo-"
    )

    assert result.ok and result.calls == 2
    assert "name: 'csv-column-sum' must start with 'demo-' (demo proposal)" in llm.payloads[1][2][1]
    assert "name: demo-csv-column-sum" in result.content


@pytest.mark.asyncio
async def test_revise_skill_md_extends_transcript_with_review_issues(fake_llm_cls) -> None:
    previous = await generate_skill_md(
        [_candidate(candidate_id="c1")], llm=fake_llm_cls(VALID), template=TEMPLATE, policy_text=POLICY, evidence=SHOWN
    )
    fixed = VALID.replace("1. Run make.", "1. Run make deps && make.")
    llm = fake_llm_cls(fixed)
    issues = ["command_mismatch: the evidence shows `make deps && make`, not `make`", "other: x"]

    result = await revise_skill_md(previous, issues, llm=llm)

    (payload,) = llm.payloads
    assert payload[:-1] == previous.transcript
    role, correction = payload[-1]
    assert role == "human"
    assert correction.startswith("A reviewer compared your SKILL.md with the accepted candidates and their evidence")
    assert "- command_mismatch: the evidence shows `make deps && make`, not `make`\n- other: x" in correction
    assert "without adding anything the evidence does not support" in correction
    assert result.ok and result.calls == 1
    assert result.content == fixed.strip()
    assert result.transcript == [*payload, ("ai", fixed.strip())]
    assert result.generation_evidence == previous.generation_evidence == {"c1": ["ev1", "ev2"]}
    assert result.generation_evidence_omitted == previous.generation_evidence_omitted == {"c1": []}


@pytest.mark.asyncio
async def test_revise_skill_md_validates_once_without_retry(fake_llm_cls) -> None:
    previous = await generate_skill_md([_candidate()], llm=fake_llm_cls(VALID), template=TEMPLATE, policy_text=POLICY)
    llm = fake_llm_cls(INVALID, VALID)

    result = await revise_skill_md(previous, ["other: x"], llm=llm, taken_names={"fix-build"})

    assert not result.ok and result.calls == 1
    assert len(llm.payloads) == 1 and llm.replies == [VALID]
    assert any("task identifier" in error for error in result.validation.errors)


@pytest.mark.asyncio
async def test_placeholder_text_inside_a_rule_is_not_resubstituted(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID)
    candidate = _candidate(rule="Keep the literal {existing_skill} marker.")

    await generate_skill_md([candidate], llm=llm, policy_text=POLICY, template=TEMPLATE)

    instruction = llm.payloads[0][0][1]
    assert "Keep the literal {existing_skill} marker." in instruction
    assert instruction.count(NO_EXISTING_SKILL) == 1


# ------------------------------------------------------------------- prompt


def test_packaged_generate_prompt_contract() -> None:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    assert lines[0].startswith("## description: ")
    assert lines[1] == "## contract_version: 2"
    for placeholder in (CANDIDATES_PLACEHOLDER, EXISTING_SKILL_PLACEHOLDER, GENERATION_POLICY_PLACEHOLDER):
        assert text.count(placeholder) == 1, placeholder
    assert "# Selection policy" in text
    assert "# REQUIRED SKILL.md STRUCTURE" in text
    assert "## Description must be proactive" in text
    assert "Do not create an empty `Examples` section." in text
    assert "# Output contract" in text
    assert text.index("# Selection policy") < text.index("# Task") < text.index("# Accepted candidates")
    assert text.index("# REQUIRED SKILL.md STRUCTURE") < text.index("# Output contract")
    # The candidate block carries conditions and evidence text, never ids.
    for present in ("When", "Constraints", "Expected outcome", "Evidence", "evidence ids"):
        assert present in text, present
    # Evidence reading rules, parameterisation, no unsupported additions, no claimed verification, one step allowed.
    for present in (
        "tool.start <tool>",
        "tool.result <tool> (ok)",
        "not proof",
        "<parameter>",
        "durable file names, column names and environment conventions",
        "Never add a command, flag, API, version, unit, dependency or guarantee of success",
        "Never claim that a test or verification ran unless the evidence shows it ran",
        "exactly one `## Workflow` step",
        "one step when the procedure is one operation",
        "Prohibitions are allowed when the evidence shows the failure they prevent",
    ):
        assert present in text, present
    for absent in ("{evidence_bundle}", "{skill_library}", "Nothing to save", "skills_list", "must never be used"):
        assert absent not in text, absent


def test_packaged_v1_generate_prompt_stays_for_hash_comparison() -> None:
    v1 = PROMPT_PATH.with_name("prompt_v1.md")
    assert v1.is_file()
    assert "contract_version" not in v1.read_text(encoding="utf-8")
