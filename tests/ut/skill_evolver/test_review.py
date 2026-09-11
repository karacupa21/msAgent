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

"""Tests for the semantic review stage and the per-plan generate_and_review chain (scripted LLM)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from msagent.skill_evolver.bundle import ShownFragment
from msagent.skill_evolver.classify import EMPTY_REPLY, Candidate
from msagent.skill_evolver.generate import INSUFFICIENT_CONTEXT_BUDGET, NO_EXISTING_SKILL, format_existing_skill
from msagent.skill_evolver.review import (
    CANDIDATES_PLACEHOLDER,
    EVIDENCE_PLACEHOLDER,
    EXISTING_SKILL_PLACEHOLDER,
    GENERATION_INVALID,
    ISSUE_CODES,
    QUALITY_REVIEW_FAILED,
    REVIEW_POLICY_PLACEHOLDER,
    SKILL_MD_PLACEHOLDER,
    VERIFICATION_EVIDENCE_SUPPORTED,
    ReviewContractError,
    ReviewIssue,
    ReviewParseError,
    ReviewResult,
    format_review_candidates,
    format_review_evidence,
    generate_and_review,
    parse_review_reply,
    review_skill_md,
)
from msagent.trajectory_recorder.model import EvidenceRef

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = REPO_ROOT / "resources" / "configs" / "default" / "skill-evolver" / "prompts" / "review" / "prompt_v2.md"
LOGGER = "msagent.skill_evolver.review"

GENERATION_TEMPLATE = "Policy:\n{generation_policy}\n\nCandidates:\n{candidates}\n\nExisting:\n{existing_skill}\n"
REVIEW_TEMPLATE = (
    "Policy:\n{review_policy}\n\nSkill:\n{skill_md}\n\nCandidates:\n{candidates}\n\n"
    "Evidence:\n{evidence}\n\nExisting:\n{existing_skill}\n"
)
GENERATION_POLICY = "Selection policy: test generation."
REVIEW_POLICY = "Review policy: test review."

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
        "1. Run `make deps && make`.",
        "2. Run the tests.",
        "",
        "## Outputs",
        "",
        "A green run.",
        "",
    ]
)
SKILL_WITH_FLAG = VALID.replace("1. Run `make deps && make`.", "1. Run `make deps && make --jobs 8`.")
INVALID = "---\nname: fix-build\ndescription: Instructions for debugging\n---\n\n## Workflow\n\n1. Only one.\n"
REVIEW_PASS = json.dumps({"verdict": "pass", "issues": []})


def _fail(*issues: dict[str, Any]) -> str:
    return json.dumps({"verdict": "fail", "issues": list(issues)})


def _issue(code: str = "unsupported_addition", detail: str = "the evidence shows no --jobs flag", **kw: Any) -> dict:
    return {"code": code, "detail": detail, **kw}


REVIEW_FAIL_FLAG = _fail(_issue(evidence_refs=["ev2"]))


# ------------------------------------------------------------------ builders


def _candidate(**overrides: Any) -> Candidate:
    data: dict[str, Any] = {
        "title": "Build before test",
        "rule": "Run make deps && make before invoking the test suite.",
        "evidence_refs": ["ev1", "ev2"],
        "future_applicability": "high",
        "target": {"action": "create", "existing_skill": None},
        "candidate_id": "c1",
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
FRAGMENTS = [SHOWN["ev1"], SHOWN["ev2"]]


async def _review(llm: Any, skill_md: str = VALID, existing: str | None = None, **overrides: Any) -> ReviewResult:
    kwargs: dict[str, Any] = {
        "llm": llm,
        "template": REVIEW_TEMPLATE,
        "policy_text": REVIEW_POLICY,
    }
    kwargs.update(overrides)
    return await review_skill_md(skill_md, [_candidate()], FRAGMENTS, existing, **kwargs)


async def _chain(llm: Any, **overrides: Any):
    kwargs: dict[str, Any] = {
        "llm": llm,
        "generation_template": GENERATION_TEMPLATE,
        "review_template": REVIEW_TEMPLATE,
        "generation_policy": GENERATION_POLICY,
        "review_policy": REVIEW_POLICY,
        "evidence": SHOWN,
    }
    kwargs.update(overrides)
    return await generate_and_review([_candidate()], **kwargs)


# ------------------------------------------------------------------- parsing


def test_parse_review_reply_strict() -> None:
    reply = parse_review_reply("<think>x</think>\n```json\n" + REVIEW_FAIL_FLAG + "\n```")
    assert reply.verdict == "fail"
    assert reply.issues == [
        ReviewIssue(code="unsupported_addition", detail="the evidence shows no --jobs flag", evidence_refs=["ev2"])
    ]

    with pytest.raises(ReviewParseError, match="invalid JSON"):
        parse_review_reply("not json")
    with pytest.raises(ReviewParseError, match="must be an object, got list"):
        parse_review_reply("[]")
    with pytest.raises(ReviewParseError, match="schema violation"):
        parse_review_reply(json.dumps({"verdict": "maybe", "issues": []}))
    with pytest.raises(ReviewParseError, match="schema violation"):
        parse_review_reply(_fail(_issue(code="typo")))
    with pytest.raises(ReviewParseError, match="schema violation"):
        parse_review_reply(_fail(_issue(evidence_refs=[3])))


def test_inconsistent_verdicts_are_schema_violations() -> None:
    with pytest.raises(ReviewParseError, match="verdict 'pass' requires an empty issues list"):
        parse_review_reply(json.dumps({"verdict": "pass", "issues": [_issue()]}))
    with pytest.raises(ReviewParseError, match="verdict 'fail' requires at least one issue"):
        parse_review_reply(json.dumps({"verdict": "fail", "issues": []}))


def test_issue_codes_are_the_nine_of_the_contract() -> None:
    assert ISSUE_CODES == {
        "lost_condition",
        "order_changed",
        "command_mismatch",
        "unsupported_addition",
        "missing_rule",
        "unsafe_claim",
        "one_time_value",
        "name_policy",
        "other",
    }
    assert VERIFICATION_EVIDENCE_SUPPORTED == {"level": "evidence_supported", "note": "not executed by generator"}


# ---------------------------------------------------------------- formatters


def test_format_review_candidates_shows_ids_conditions_and_citations() -> None:
    plain = _candidate()
    rich = _candidate(
        candidate_id="c2",
        title="Second",
        target={"action": "update", "existing_skill": "profiler/real"},
        applies_when="the build is stale",
        constraints=["run make once"],
        expected_outcome="green tests",
        evidence_refs=["ev3"],
    )

    text = format_review_candidates([plain, rich])

    assert text == "\n".join(
        [
            "1. [c1] Build before test — target create",
            "   Rule: Run make deps && make before invoking the test suite.",
            "   Cites: ev1, ev2",
            "2. [c2] Second — target update `profiler/real`",
            "   Rule: Run make deps && make before invoking the test suite.",
            "   When: the build is stale",
            "   Constraints:",
            "   - run make once",
            "   Expected outcome: green tests",
            "   Cites: ev3",
        ]
    )


def test_format_review_evidence_shows_ids_and_kind_deduplicated() -> None:
    text = format_review_evidence([SHOWN["ev2"], SHOWN["ev1"], SHOWN["ev2"]])

    assert text == "\n".join(
        [
            '- [ev2] (context) tool.start bash: {"cmd": "make deps && make"}',
            "- [ev1] (required) tool.error bash (error): CalledProcessError: exit 2 missing dep",
        ]
    )


# ----------------------------------------------------------- review_skill_md


@pytest.mark.asyncio
async def test_review_pass_single_call(fake_llm_cls) -> None:
    llm = fake_llm_cls(REVIEW_PASS)

    result = await _review(llm)

    assert result.verdict == "pass" and result.issues == [] and result.unknown_refs == [] and result.calls == 1
    assert result.issue_lines() == []
    assert result.record() == {"verdict": "pass", "issues": [], "unknown_refs": [], "calls": 1}
    (payload,) = llm.payloads
    ((role, instruction),) = payload
    assert role == "human"
    assert VALID in instruction
    assert "1. [c1] Build before test — target create" in instruction
    assert "- [ev1] (required) " + SHOWN["ev1"].text in instruction
    assert "- [ev2] (context) " + SHOWN["ev2"].text in instruction
    assert REVIEW_POLICY in instruction and NO_EXISTING_SKILL in instruction
    for placeholder in (
        SKILL_MD_PLACEHOLDER,
        CANDIDATES_PLACEHOLDER,
        EVIDENCE_PLACEHOLDER,
        EXISTING_SKILL_PLACEHOLDER,
        REVIEW_POLICY_PLACEHOLDER,
    ):
        assert placeholder not in instruction


@pytest.mark.asyncio
async def test_review_existing_skill_text_and_braces_in_skill_are_inert(fake_llm_cls) -> None:
    llm = fake_llm_cls(REVIEW_PASS)
    existing = format_existing_skill("real", "---\nname: real\n---\nold body\n")
    skill = VALID.replace("A green run.", "A green run with a literal {evidence} marker.")

    await _review(llm, skill_md=skill, existing=existing)

    instruction = llm.payloads[0][0][1]
    assert existing in instruction and NO_EXISTING_SKILL not in instruction
    assert "a literal {evidence} marker" in instruction
    assert instruction.count("- [ev1] (required)") == 1


@pytest.mark.asyncio
async def test_review_fail_returns_issues(fake_llm_cls) -> None:
    llm = fake_llm_cls(REVIEW_FAIL_FLAG)

    result = await _review(llm, skill_md=SKILL_WITH_FLAG)

    assert result.verdict == "fail" and result.calls == 1
    assert result.issue_lines() == ["unsupported_addition: the evidence shows no --jobs flag"]
    assert result.unknown_refs == []
    assert result.record()["issues"] == [
        {"code": "unsupported_addition", "detail": "the evidence shows no --jobs flag", "evidence_refs": ["ev2"]}
    ]


@pytest.mark.asyncio
async def test_review_parse_error_retried_once_then_contract_error(fake_llm_cls) -> None:
    llm = fake_llm_cls("not json", "still not json")

    with pytest.raises(ReviewContractError, match="after 2 attempt") as info:
        await _review(llm)

    assert info.value.calls == 2
    assert len(llm.payloads) == 2
    first, second = llm.payloads
    assert second[0] == first[0]
    assert second[1] == ("ai", "not json")
    role, correction = second[2]
    assert role == "human"
    assert correction.startswith("Your previous reply is not the JSON object the review contract requires.")
    assert "invalid JSON" in correction

    llm = fake_llm_cls("not json", REVIEW_PASS)
    with pytest.raises(ReviewContractError, match="after 1 attempt") as info:
        await _review(llm, corrective_retry=False)
    assert info.value.calls == 1 and len(llm.payloads) == 1


@pytest.mark.asyncio
async def test_review_blank_reply_replayed_as_empty_then_parsed(fake_llm_cls) -> None:
    llm = fake_llm_cls("   ", REVIEW_PASS)

    result = await _review(llm)

    assert result.verdict == "pass" and result.calls == 2
    assert llm.payloads[1][1] == ("ai", EMPTY_REPLY)


@pytest.mark.asyncio
async def test_review_inconsistent_verdict_is_a_parse_error(fake_llm_cls) -> None:
    llm = fake_llm_cls(json.dumps({"verdict": "pass", "issues": [_issue()]}), REVIEW_PASS)

    result = await _review(llm)

    assert result.verdict == "pass" and result.calls == 2
    assert "verdict 'pass' requires an empty issues list" in llm.payloads[1][2][1]

    llm = fake_llm_cls(
        json.dumps({"verdict": "fail", "issues": []}), json.dumps({"verdict": "pass", "issues": [_issue()]})
    )
    with pytest.raises(ReviewContractError, match="schema violation"):
        await _review(llm)


@pytest.mark.asyncio
async def test_review_unknown_evidence_ref_is_flagged_not_dropped(
    fake_llm_cls, caplog: pytest.LogCaptureFixture
) -> None:
    llm = fake_llm_cls(_fail(_issue(evidence_refs=["ev99", "ev2", "ev99", "ev7"])))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await _review(llm, skill_md=SKILL_WITH_FLAG)

    assert result.verdict == "fail"
    assert result.unknown_refs == ["ev99", "ev7"]
    assert len(result.issues) == 1 and result.issues[0].evidence_refs == ["ev99", "ev2", "ev99", "ev7"]
    assert "cite evidence not shown" in caplog.text and "ev99" in caplog.text
    assert result.record()["unknown_refs"] == ["ev99", "ev7"]


@pytest.mark.asyncio
async def test_review_guards_raise_before_any_call(fake_llm_cls) -> None:
    llm = fake_llm_cls(REVIEW_PASS)
    kwargs: dict[str, Any] = {"llm": llm, "template": REVIEW_TEMPLATE, "policy_text": REVIEW_POLICY}

    with pytest.raises(ValueError, match="SKILL.md is empty"):
        await review_skill_md("  \n", [_candidate()], FRAGMENTS, None, **kwargs)
    with pytest.raises(ValueError, match="no candidates"):
        await review_skill_md(VALID, [], FRAGMENTS, None, **kwargs)
    with pytest.raises(ValueError, match="no evidence fragments"):
        await review_skill_md(VALID, [_candidate()], [], None, **kwargs)
    for placeholder in (
        SKILL_MD_PLACEHOLDER,
        CANDIDATES_PLACEHOLDER,
        EVIDENCE_PLACEHOLDER,
        EXISTING_SKILL_PLACEHOLDER,
        REVIEW_POLICY_PLACEHOLDER,
    ):
        broken = REVIEW_TEMPLATE.replace(placeholder, "")
        with pytest.raises(ValueError, match="placeholder") as info:
            await review_skill_md(VALID, [_candidate()], FRAGMENTS, None, **{**kwargs, "template": broken})
        assert placeholder in str(info.value)
    assert llm.payloads == []


# -------------------------------------------------------- generate_and_review


@pytest.mark.asyncio
async def test_generate_and_review_pass_is_two_calls(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID, REVIEW_PASS)

    result = await _chain(llm)

    assert result.ok and result.code is None and result.errors == []
    assert result.calls == 2 and len(llm.payloads) == 2
    assert result.content == VALID.strip()
    assert result.validation is not None and result.validation.ok
    assert result.review is not None and result.review.verdict == "pass"
    assert result.corrected is False and result.initial_issues == []
    assert result.generation_evidence == {"c1": ["ev1", "ev2"]} and result.generation_evidence_omitted == {"c1": []}
    assert result.review_record() == {
        "verdict": "pass",
        "issues": [],
        "unknown_refs": [],
        "calls": 1,
        "corrected": False,
        "initial_issues": [],
    }
    generation_instruction, review_instruction = llm.payloads[0][0][1], llm.payloads[1][0][1]
    assert GENERATION_POLICY in generation_instruction and REVIEW_POLICY in review_instruction
    # The reviewer sees exactly what the generation call was quoted, with ids.
    assert "- [ev1] (required) " + SHOWN["ev1"].text in review_instruction
    assert "- [ev2] (context) " + SHOWN["ev2"].text in review_instruction
    assert "[ev3]" not in review_instruction and "[ev4]" not in review_instruction


@pytest.mark.asyncio
async def test_generate_and_review_unknown_flag_blocked_then_corrected(fake_llm_cls) -> None:
    llm = fake_llm_cls(SKILL_WITH_FLAG, REVIEW_FAIL_FLAG, VALID, REVIEW_PASS)

    result = await _chain(llm)

    assert result.ok and result.calls == 4 and len(llm.payloads) == 4
    assert result.corrected is True
    assert result.initial_issues == ["unsupported_addition: the evidence shows no --jobs flag"]
    assert result.content == VALID.strip()
    assert result.review is not None and result.review.verdict == "pass"
    # The corrective revision continues the generation transcript with the reviewer's bullet.
    revise = llm.payloads[2]
    assert revise[0] == llm.payloads[0][0]
    assert revise[1] == ("ai", SKILL_WITH_FLAG.strip())
    assert revise[2][0] == "human"
    assert revise[2][1].endswith("nothing before or after it.")
    assert "- unsupported_addition: the evidence shows no --jobs flag" in revise[2][1]
    # The second review sees the corrected skill and gets no parse retry budget.
    assert VALID in llm.payloads[3][0][1]
    assert result.review_record()["corrected"] is True
    assert result.review_record()["initial_issues"] == result.initial_issues


@pytest.mark.asyncio
async def test_generate_and_review_lost_constraint_fails_after_one_correction(fake_llm_cls) -> None:
    lost = _fail(_issue("lost_condition", "the missing-dependency error condition is gone", evidence_refs=["ev1"]))
    still = _fail(_issue("lost_condition", "the condition is still missing", evidence_refs=["ev1"]))
    llm = fake_llm_cls(VALID, lost, SKILL_WITH_FLAG, still)

    result = await _chain(llm)

    assert not result.ok and result.code == QUALITY_REVIEW_FAILED
    assert result.calls == 4 and len(llm.payloads) == 4 and llm.replies == []
    assert result.errors == ["lost_condition: the condition is still missing"]
    assert result.content is None
    assert result.review is not None and result.review.verdict == "fail"
    assert result.corrected is False
    # The first round's findings and the corrected draft survive for the record.
    assert result.initial_issues == ["lost_condition: the missing-dependency error condition is gone"]
    assert result.review_record()["initial_issues"] == result.initial_issues
    assert result.draft is not None and result.draft.strip() == SKILL_WITH_FLAG.strip()


@pytest.mark.asyncio
async def test_generate_and_review_invalid_corrective_revision_is_quality_review_failed(fake_llm_cls) -> None:
    llm = fake_llm_cls(SKILL_WITH_FLAG, REVIEW_FAIL_FLAG, INVALID, REVIEW_PASS)

    result = await _chain(llm)

    assert result.code == QUALITY_REVIEW_FAILED and result.calls == 3
    assert len(llm.payloads) == 3 and llm.replies == [REVIEW_PASS]
    assert result.errors and all(error.startswith("corrective SKILL.md invalid: ") for error in result.errors)
    assert any("task identifier" in error for error in result.errors)
    assert result.content is None
    # The first review is kept for the record.
    assert result.review is not None and result.review.verdict == "fail"


@pytest.mark.asyncio
async def test_generate_and_review_reviewer_invalid_twice_is_quality_review_failed(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID, "nope", "still nope")

    result = await _chain(llm)

    assert result.code == QUALITY_REVIEW_FAILED and result.calls == 3 and len(llm.payloads) == 3
    assert result.errors == [
        "reviewer reply invalid: review: reply is not the required JSON after 2 attempt(s): invalid JSON: "
        + result.errors[0].split("invalid JSON: ", 1)[1]
    ]
    assert result.content is None and result.review is None
    assert result.generation_evidence == {"c1": ["ev1", "ev2"]}


@pytest.mark.asyncio
async def test_generate_and_review_second_reviewer_reply_invalid_is_quality_review_failed(fake_llm_cls) -> None:
    llm = fake_llm_cls(SKILL_WITH_FLAG, REVIEW_FAIL_FLAG, VALID, "garbage")

    result = await _chain(llm)

    assert result.code == QUALITY_REVIEW_FAILED and result.calls == 4 and len(llm.payloads) == 4
    assert result.errors[0].startswith(
        "reviewer reply invalid: review: reply is not the required JSON after 1 attempt(s)"
    )
    assert result.content is None


@pytest.mark.asyncio
async def test_generate_and_review_generation_invalid_makes_no_review_call(fake_llm_cls) -> None:
    llm = fake_llm_cls(INVALID, "no frontmatter at all")

    result = await _chain(llm)

    assert result.code == GENERATION_INVALID and result.calls == 2 and len(llm.payloads) == 2
    assert any("no YAML frontmatter" in error for error in result.errors)
    assert result.content is None and result.review is None
    assert result.validation is not None and not result.validation.ok


@pytest.mark.asyncio
async def test_generate_and_review_budget_rejection_makes_no_call(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID, REVIEW_PASS)

    result = await _chain(llm, evidence_budget_chars=1)

    assert result.code == INSUFFICIENT_CONTEXT_BUDGET and result.calls == 0
    assert llm.payloads == []
    assert result.errors == ["candidate c1: required evidence ['ev1'] does not fit the evidence budget"]
    assert result.content is None and result.review is None
    assert result.generation_evidence == {} and result.generation_evidence_omitted == {}


@pytest.mark.asyncio
async def test_generate_and_review_empty_selection_is_budget_rejection_before_llm(fake_llm_cls) -> None:
    llm = fake_llm_cls(VALID, REVIEW_PASS)
    only_optional = [_candidate(evidence_refs=["ev2", "ev4"])]

    result = await generate_and_review(
        only_optional,
        llm=llm,
        generation_template=GENERATION_TEMPLATE,
        review_template=REVIEW_TEMPLATE,
        generation_policy=GENERATION_POLICY,
        review_policy=REVIEW_POLICY,
        evidence=SHOWN,
        evidence_budget_chars=1,
    )

    assert result.code == INSUFFICIENT_CONTEXT_BUDGET and result.calls == 0 and llm.payloads == []
    assert result.errors == [
        "no evidence fragment fits the evidence budget; nothing to generate from or review against"
    ]


@pytest.mark.asyncio
async def test_generate_and_review_max_six_calls(fake_llm_cls) -> None:
    fixed = VALID.replace("A green run.", "A green run of the test suite.")
    llm = fake_llm_cls(INVALID, SKILL_WITH_FLAG, "not json", REVIEW_FAIL_FLAG, fixed, REVIEW_PASS)

    result = await _chain(llm)

    assert result.ok and result.corrected is True
    assert result.calls == 6 and len(llm.payloads) == 6 and llm.replies == []
    assert result.content == fixed.strip()
    assert result.review is not None and result.review.calls == 1
    assert result.initial_issues == ["unsupported_addition: the evidence shows no --jobs flag"]
    # generate 2 (validator correction) + review 2 (parse retry) + revise 1 + review 1
    assert [len(payload) for payload in llm.payloads] == [1, 3, 1, 3, 5, 1]


@pytest.mark.asyncio
async def test_generate_and_review_passes_update_and_prefix_arguments_through(fake_llm_cls) -> None:
    existing = format_existing_skill("real", "---\nname: real\n---\nold body\n")
    llm = fake_llm_cls(VALID.replace("name: build-before-test", "name: real"), REVIEW_PASS)

    result = await _chain(llm, existing_skill=existing, existing_skill_text=existing, expected_name="real")

    assert result.ok and result.calls == 2
    assert existing in llm.payloads[0][0][1] and existing in llm.payloads[1][0][1]

    llm = fake_llm_cls(VALID, VALID.replace("name: build-before-test", "name: demo-build-before-test"), REVIEW_PASS)
    result = await _chain(llm, required_prefix="demo-", taken_names={"other"})
    assert result.ok and result.calls == 3
    assert "must start with 'demo-'" in llm.payloads[1][2][1]
    assert "name: demo-build-before-test" in result.content


# ------------------------------------------------------------------- prompt


def test_packaged_review_prompt_contract() -> None:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    assert lines[0].startswith("## description: ")
    assert lines[1] == "## contract_version: 2"
    for placeholder in (
        SKILL_MD_PLACEHOLDER,
        CANDIDATES_PLACEHOLDER,
        EVIDENCE_PLACEHOLDER,
        EXISTING_SKILL_PLACEHOLDER,
        REVIEW_POLICY_PLACEHOLDER,
    ):
        assert text.count(placeholder) == 1, placeholder
    for code in sorted(ISSUE_CODES):
        assert f"`{code}`" in text, code
    assert '"verdict"' in text and '"issues"' in text and '"evidence_refs"' in text
    assert "a claimed test or verification not present in the evidence" in text
    assert "does not mean the procedure was executed" in text
    assert "[ev<k>]" in text
    for absent in ("{evidence_bundle}", "{skill_library}", "{selection_policy}", "{generation_policy}"):
        assert absent not in text, absent
