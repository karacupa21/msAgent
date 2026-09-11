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

"""Tests for the JSON classify stage (contract v2) on a scripted fake LLM (no network)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from msagent.skill_evolver.classify import (
    BUNDLE_PLACEHOLDER,
    CODE_CONTRACT_ERROR,
    CODE_INVALID_EVIDENCE,
    CODE_PARSE_ERROR,
    CONTRACT_VERSION,
    EMPTY_REPLY,
    EXPLANATION_LIMIT,
    LIBRARY_PLACEHOLDER,
    POLICY_PLACEHOLDER,
    REASON_CODES,
    ClassifyContractError,
    ClassifyParseError,
    classify,
    parse_classify_reply,
    strip_code_fence,
    strip_think_blocks,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
PROMPT_PATH = (
    REPO_ROOT / "resources" / "configs" / "default" / "skill-evolver" / "prompts" / "classify" / "prompt_v2.md"
)
LOGGER = "msagent.skill_evolver.classify"

TEMPLATE = "Classify the evidence.\n\n{evidence_bundle}\n\nReply with JSON only."
BUNDLE = (
    "### Episode E1 — error_recovery (weight 0.60, thread thread-t)\n"
    "Excerpts:\n"
    '- [ev1] tool.start bash: {"cmd": "make"}\n'
    "- [ev2] tool.error bash (error): exit 2\n"
    '- [ev3] tool.start bash: {"cmd": "make deps && make"}'
)
VALID_REFS = {"ev1", "ev2", "ev3"}
EPISODE_IDS = {"E1"}


class _FakeLLM:
    """Records every payload and answers with the scripted replies in order.

    A ``str`` reply is wrapped into an ``AIMessage`` unless ``raw`` is set;
    any other object is returned as is.
    """

    def __init__(self, *replies: Any, raw: bool = False) -> None:
        self.replies = list(replies)
        self.raw = raw
        self.payloads: list[list[tuple[str, str]]] = []

    async def ainvoke(self, payload: list[tuple[str, str]]) -> Any:
        self.payloads.append(list(payload))
        assert self.replies, "fake LLM called more often than scripted"
        reply = self.replies.pop(0)
        if isinstance(reply, str) and not self.raw:
            return AIMessage(content=reply)
        return reply


# ------------------------------------------------------------------ builders


def _candidate(**overrides: Any) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "title": "Build before test",
        "rule": "Run make before invoking the test suite.",
        "evidence_refs": ["ev1", "ev2"],
        "future_applicability": "high",
        "target": {"action": "create", "existing_skill": None},
    }
    candidate.update(overrides)
    return candidate


def _decision(**overrides: Any) -> dict[str, Any]:
    decision: dict[str, Any] = {
        "episode_ids": ["E1"],
        "decision": "accept",
        "reason_code": "accepted",
        "explanation": "the argument diff shows the missing build step",
        "evidence_refs": ["ev1", "ev2"],
    }
    decision.update(overrides)
    return decision


def _reply(
    *candidates: dict[str, Any],
    verdict: str = "save",
    decisions: list[dict[str, Any]] | None = None,
    **top_level: Any,
) -> str:
    """A contract v2 reply; the default decision on E1 follows the candidates."""
    if decisions is None:
        decisions = [_decision()] if candidates else [_decision(decision="reject", reason_code="routine_activity")]
    body: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "verdict": verdict,
        "candidates": list(candidates),
        "decisions": decisions,
    }
    body.update(top_level)
    return json.dumps(body)


def _drop(reply: str, key: str) -> str:
    body = json.loads(reply)
    del body[key]
    return json.dumps(body)


async def _classify(llm: _FakeLLM, *, demo: bool = False, episode_ids: set[str] = EPISODE_IDS) -> Any:
    return await classify(BUNDLE, VALID_REFS, llm, TEMPLATE, episode_ids=episode_ids, demo=demo)


# ---------------------------------------------------------------- constants


def test_contract_constants() -> None:
    assert CONTRACT_VERSION == 2
    assert REASON_CODES == {
        "accepted",
        "routine_activity",
        "insufficient_evidence",
        "transient_observation",
        "already_covered",
        "no_transferable_rule",
        "unsafe_procedure",
    }
    assert (CODE_PARSE_ERROR, CODE_CONTRACT_ERROR, CODE_INVALID_EVIDENCE) == (
        "parse_error",
        "contract_error",
        "invalid_evidence",
    )
    assert (LIBRARY_PLACEHOLDER, POLICY_PLACEHOLDER) == ("{skill_library}", "{selection_policy}")
    assert EXPLANATION_LIMIT == 500
    assert issubclass(ClassifyContractError, ValueError)
    assert issubclass(ClassifyParseError, ValueError)


# ---------------------------------------------------------------- happy path


@pytest.mark.asyncio
async def test_valid_json_single_call() -> None:
    llm = _FakeLLM(_reply(_candidate(evidence_refs=["ev3", "ev1", "ev1"])))

    result = await _classify(llm)

    assert llm.payloads == [[("human", TEMPLATE.replace(BUNDLE_PLACEHOLDER, BUNDLE))]]
    assert result.verdict == "save"
    assert result.rejected == []
    (candidate,) = result.candidates
    assert candidate.title == "Build before test"
    assert candidate.rule == "Run make before invoking the test suite."
    assert candidate.evidence_refs == ["ev3", "ev1"]  # deduplicated, order kept
    assert candidate.future_applicability == "high"
    assert candidate.target.action == "create"
    assert candidate.target.existing_skill is None
    assert candidate.candidate_id == "c1"
    # Conditions the model did not state stay empty, never invented.
    assert (candidate.applies_when, candidate.constraints, candidate.expected_outcome) == (None, [], None)


@pytest.mark.asyncio
async def test_v2_reply_returns_decisions_calls_and_ids() -> None:
    decisions = [
        _decision(evidence_refs=["ev1", "ev2"]),
        _decision(episode_ids=["E1"], decision="reject", reason_code="transient_observation", evidence_refs=[]),
    ]
    llm = _FakeLLM(_reply(_candidate(), _candidate(title="Covered", covered_by="build-basics"), decisions=decisions))

    result = await _classify(llm)

    assert (result.verdict, result.model_verdict, result.calls) == ("save", "save", 1)
    assert [(c.candidate_id, c.covered_by) for c in result.candidates] == [("c1", None), ("c2", "build-basics")]
    assert [(d.episode_ids, d.decision, d.reason_code) for d in result.decisions] == [
        (["E1"], "accept", "accepted"),
        (["E1"], "reject", "transient_observation"),
    ]
    assert result.decisions[0].explanation == "the argument diff shows the missing build step"
    assert result.decisions[0].evidence_refs == ["ev1", "ev2"]
    assert result.insufficient_evidence is False


@pytest.mark.asyncio
async def test_conditions_are_kept_when_the_model_states_them() -> None:
    stated = _candidate(
        applies_when="the test suite depends on generated code",
        constraints=["keep generated files out of version control"],
        expected_outcome="a green test run without manual generation",
    )
    llm = _FakeLLM(_reply(stated))

    (candidate,) = (await _classify(llm)).candidates

    assert candidate.applies_when == "the test suite depends on generated code"
    assert candidate.constraints == ["keep generated files out of version control"]
    assert candidate.expected_outcome == "a green test run without manual generation"


@pytest.mark.asyncio
async def test_candidate_ids_number_kept_candidates_only() -> None:
    llm = _FakeLLM(
        _reply(
            _candidate(title="First", candidate_id="ignored"),
            _candidate(title="Fabricated", evidence_refs=["ev9"]),
            _candidate(title="Second"),
        )
    )

    result = await _classify(llm)

    assert [(c.title, c.candidate_id) for c in result.candidates] == [("First", "c1"), ("Second", "c2")]
    assert [c.title for c, _ in result.rejected] == ["Fabricated"]


@pytest.mark.asyncio
async def test_update_target_keeps_existing_skill() -> None:
    target = {"action": "update", "existing_skill": "profiling-bottleneck"}
    llm = _FakeLLM(_reply(_candidate(target=target)))

    result = await _classify(llm)

    (candidate,) = result.candidates
    assert candidate.target.action == "update"
    assert candidate.target.existing_skill == "profiling-bottleneck"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrap",
    [
        "```json\n{body}\n```",
        "```\n{body}\n```",
        "<think>\nlet me think {about} it\n</think>\n{body}",
        "<THINK>x</THINK>\n```json\n{body}```",
    ],
)
async def test_fences_and_think_blocks_are_stripped(wrap: str) -> None:
    llm = _FakeLLM(wrap.replace("{body}", _reply(_candidate())))

    result = await _classify(llm)

    assert len(llm.payloads) == 1
    assert result.verdict == "save"
    assert [c.evidence_refs for c in result.candidates] == [["ev1", "ev2"]]


@pytest.mark.asyncio
async def test_nothing_verdict_without_candidates() -> None:
    llm = _FakeLLM(_reply(verdict="nothing"))

    result = await _classify(llm)

    assert (result.verdict, result.model_verdict, result.candidates, result.rejected) == ("nothing", "nothing", [], [])
    assert [(d.decision, d.reason_code) for d in result.decisions] == [("reject", "routine_activity")]
    assert result.insufficient_evidence is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(content=_reply(_candidate())),
        SimpleNamespace(text=lambda: "method, not property", content=_reply(_candidate())),
        AIMessage(content=[{"type": "text", "text": _reply(_candidate())}]),
        _reply(_candidate()),
    ],
    ids=["content-only", "text-method", "content-blocks", "plain-str"],
)
async def test_reply_shapes_without_text_property(response: Any) -> None:
    llm = _FakeLLM(response, raw=True)

    result = await _classify(llm)

    assert result.verdict == "save"
    assert len(result.candidates) == 1


# --------------------------------------------------------------------- retry


@pytest.mark.asyncio
async def test_invalid_json_then_valid_retries_once_with_error_text() -> None:
    llm = _FakeLLM("not json at all", _reply(_candidate()))

    result = await _classify(llm)

    assert (result.verdict, result.calls) == ("save", 2)
    assert len(llm.payloads) == 2
    first_prompt = llm.payloads[0][0]
    assert llm.payloads[1][0] == first_prompt
    assert llm.payloads[1][1] == ("ai", "not json at all")
    role, correction = llm.payloads[1][2]
    assert role == "human"
    assert "could not be parsed" in correction
    assert "invalid JSON" in correction
    assert "Expecting value" in correction  # the json module's own error text


@pytest.mark.asyncio
async def test_invalid_twice_raises_after_exactly_two_calls() -> None:
    llm = _FakeLLM("garbage", "still garbage")

    with pytest.raises(ClassifyParseError, match="after one retry"):
        await _classify(llm)

    assert len(llm.payloads) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad_reply", "fragment"),
    [
        ("[]", "must be an object"),
        ('"save"', "must be an object"),
        (_reply(_candidate(evidence_refs=[True, "ev1"])), "string_type"),
        (_reply(_candidate(evidence_refs=[4])), "string_type"),
        (_reply(_candidate(evidence_refs=[4.0])), "string_type"),
        (_reply(_candidate(future_applicability="certain")), "literal_error"),
        (_reply(_candidate(target={"action": "update"})), "requires existing_skill"),
        (
            _reply(_candidate(target={"action": "reference", "existing_skill": " "})),
            "requires existing_skill",
        ),
        (_reply(_candidate(covered_by=7)), "string_type"),
        (_reply(_candidate(), verdict="maybe"), "literal_error"),
        ('{"verdict": "nothing"}', "candidates"),
        (_drop(_reply(_candidate()), "contract_version"), "contract_version"),
        (_reply(_candidate(), contract_version=1), "literal_error"),
        (_reply(_candidate(), contract_version="2"), "literal_error"),
        (_drop(_reply(_candidate()), "decisions"), "decisions"),
        (_reply(_candidate(), decisions=[]), "too_short"),
        (_reply(_candidate(), decisions=[_decision(episode_ids=[])]), "too_short"),
        (_reply(_candidate(), decisions=[_decision(episode_ids=[1])]), "string_type"),
        (_reply(_candidate(), decisions=[_decision(decision="maybe")]), "literal_error"),
        (_reply(_candidate(), decisions=[_decision(reason_code="boring")]), "literal_error"),
        (
            _reply(_candidate(), decisions=[_decision(reason_code="routine_activity")]),
            "requires reason_code 'accepted'",
        ),
        (
            _reply(_candidate(), decisions=[_decision(decision="reject")]),
            "requires reason_code 'accepted'",
        ),
        (_reply(_candidate(), decisions=[_decision(explanation="x" * (EXPLANATION_LIMIT + 1))]), "string_too_long"),
        (_reply(_candidate(), decisions=[_decision(evidence_refs=[3])]), "string_type"),
    ],
    ids=[
        "list",
        "string",
        "bool-ref",
        "int-ref",
        "float-ref",
        "bad-applicability",
        "update-without-skill",
        "reference-blank-skill",
        "int-covered-by",
        "bad-verdict",
        "missing-candidates",
        "missing-contract-version",
        "contract-version-1",
        "contract-version-string",
        "missing-decisions",
        "empty-decisions",
        "empty-episode-ids",
        "int-episode-id",
        "bad-decision",
        "unknown-reason-code",
        "accept-with-reject-code",
        "reject-with-accepted-code",
        "explanation-too-long",
        "int-decision-ref",
    ],
)
async def test_schema_violations_go_through_the_retry(bad_reply: str, fragment: str) -> None:
    llm = _FakeLLM(bad_reply, _reply(_candidate()))

    result = await _classify(llm)

    assert (result.verdict, result.calls) == ("save", 2)
    assert len(llm.payloads) == 2
    correction = llm.payloads[1][2][1]
    assert "could not be parsed" in correction
    assert fragment in correction


@pytest.mark.asyncio
async def test_explanation_at_the_limit_is_accepted() -> None:
    llm = _FakeLLM(_reply(_candidate(), decisions=[_decision(explanation="x" * EXPLANATION_LIMIT)]))

    result = await _classify(llm)

    assert result.calls == 1
    assert len(result.decisions[0].explanation) == EXPLANATION_LIMIT


@pytest.mark.asyncio
async def test_blank_first_reply_is_replayed_as_placeholder_turn() -> None:
    llm = _FakeLLM("  \n", _reply(_candidate()))

    result = await _classify(llm)

    assert result.verdict == "save"
    assert llm.payloads[1][1] == ("ai", EMPTY_REPLY)


# ------------------------------------------------------------------ contract

# (reply, demo, text the correction and the final error must contain)
_CONTRACT_CASES = [
    pytest.param(_reply(verdict="save"), False, "verdict 'save' requires at least one candidate", id="save-empty"),
    pytest.param(
        _reply(_candidate(), verdict="nothing"),
        False,
        "verdict 'nothing' requires an empty candidates list",
        id="nothing-with-candidate",
    ),
    pytest.param(
        _reply(_candidate(), decisions=[_decision(episode_ids=["E1", "E9"])]),
        False,
        "decisions cite unknown episode ids ['E9']; the bundle shows ['E1']",
        id="unknown-episode",
    ),
    pytest.param(
        _reply(_candidate(), decisions=[_decision(evidence_refs=["ev1", "ev41", "ev41", "4"])]),
        False,
        "decisions cite evidence ids not shown in the bundle: ['ev41', '4']",
        id="unknown-decision-ref",
    ),
    pytest.param(
        _reply(_candidate(title="Known", target={"action": "update", "existing_skill": "build-basics"})),
        True,
        "candidate 'Known': target.action 'update' is not allowed in demo mode; "
        "use 'create' and name the covering skill in covered_by",
        id="demo-update",
    ),
    pytest.param(
        _reply(_candidate(title="Seen", target={"action": "reference", "existing_skill": "build-basics"})),
        True,
        "candidate 'Seen': target.action 'reference' is not allowed in demo mode",
        id="demo-reference",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("bad_reply", "demo", "violation"), _CONTRACT_CASES)
async def test_contract_violation_gets_one_corrective_retry(bad_reply: str, demo: bool, violation: str) -> None:
    good = _reply(_candidate(covered_by="build-basics" if demo else None))
    llm = _FakeLLM(bad_reply, good)

    result = await _classify(llm, demo=demo)

    assert (result.verdict, result.model_verdict, result.calls) == ("save", "save", 2)
    assert [c.candidate_id for c in result.candidates] == ["c1"]
    assert len(llm.payloads) == 2
    assert llm.payloads[1][0] == llm.payloads[0][0]
    assert llm.payloads[1][1] == ("ai", bad_reply)
    role, correction = llm.payloads[1][2]
    assert role == "human"
    assert "violates the output contract" in correction
    assert f"- {violation}" in correction
    assert "could not be parsed" not in correction
    assert ('use target.action "create" for every candidate' in correction) is demo


@pytest.mark.asyncio
@pytest.mark.parametrize(("bad_reply", "demo", "violation"), _CONTRACT_CASES)
async def test_contract_violation_twice_raises_after_exactly_two_calls(
    bad_reply: str, demo: bool, violation: str
) -> None:
    llm = _FakeLLM(bad_reply, bad_reply)

    with pytest.raises(ClassifyContractError, match="after one retry") as info:
        await _classify(llm, demo=demo)

    assert violation in str(info.value)
    assert len(llm.payloads) == 2


@pytest.mark.asyncio
async def test_contract_correction_lists_every_violation_and_sorts_shown_ids_numerically() -> None:
    bad = _reply(
        verdict="save",
        decisions=[
            _decision(episode_ids=["E10", "E3", "E9"], decision="reject", reason_code="routine_activity"),
            _decision(episode_ids=["E9"], decision="reject", reason_code="no_transferable_rule", evidence_refs=["ev7"]),
        ],
    )
    llm = _FakeLLM(bad, bad)

    with pytest.raises(ClassifyContractError) as info:
        await _classify(llm, episode_ids={"E1", "E2", "E10"})

    correction = llm.payloads[1][2][1]
    assert correction.index("- verdict 'save' requires at least one candidate") < correction.index(
        "- decisions cite unknown episode ids ['E3', 'E9']; the bundle shows ['E1', 'E2', 'E10']"
    )
    assert "- decisions cite evidence ids not shown in the bundle: ['ev7']" in correction
    assert str(info.value) == (
        "classify: reply violates the contract after one retry: "
        "verdict 'save' requires at least one candidate; "
        "decisions cite unknown episode ids ['E3', 'E9']; the bundle shows ['E1', 'E2', 'E10']; "
        "decisions cite evidence ids not shown in the bundle: ['ev7']"
    )


@pytest.mark.asyncio
async def test_parse_error_then_contract_violation_raises_contract_error() -> None:
    llm = _FakeLLM("garbage", _reply(verdict="save"))

    with pytest.raises(ClassifyContractError, match="after one retry"):
        await _classify(llm)

    assert len(llm.payloads) == 2


@pytest.mark.asyncio
async def test_contract_violation_then_parse_error_raises_parse_error() -> None:
    llm = _FakeLLM(_reply(verdict="save"), "garbage")

    with pytest.raises(ClassifyParseError, match="after one retry"):
        await _classify(llm)

    assert len(llm.payloads) == 2


@pytest.mark.asyncio
async def test_demo_accepts_create_with_covered_by_in_one_call() -> None:
    llm = _FakeLLM(_reply(_candidate(covered_by="build-basics")))

    result = await _classify(llm, demo=True)

    assert (result.verdict, result.calls) == ("save", 1)
    assert result.candidates[0].covered_by == "build-basics"


# -------------------------------------------------------------- verification


@pytest.mark.asyncio
async def test_ref_not_shown_is_rejected_with_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # "ev41" was never shown (excluded or invented) and "4" is a seq number,
    # which is not something the model has seen either.
    llm = _FakeLLM(
        _reply(
            _candidate(title="Grounded", evidence_refs=["ev1"]),
            _candidate(title="Fabricated", evidence_refs=["ev1", "ev41", "4"]),
        )
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await _classify(llm)

    assert result.verdict == "save"
    assert [c.title for c in result.candidates] == ["Grounded"]
    ((rejected, reason),) = result.rejected
    assert rejected.title == "Fabricated"
    assert reason == "evidence not shown in the bundle: ['ev41', '4']"
    assert "rejected candidate 'Fabricated': evidence not shown in the bundle: ['ev41', '4']" in caplog.text
    assert "rejected candidate 'Grounded'" not in caplog.text


@pytest.mark.asyncio
async def test_empty_evidence_refs_drops_candidate_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _FakeLLM(_reply(_candidate(title="Hollow", evidence_refs=[]), _candidate(title="Solid")))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await _classify(llm)

    assert [c.title for c in result.candidates] == ["Solid"]
    assert [(c.title, reason) for c, reason in result.rejected] == [("Hollow", "empty evidence_refs")]
    assert "rejected candidate 'Hollow': empty evidence_refs" in caplog.text


@pytest.mark.asyncio
async def test_candidate_with_unknown_refs_is_rejected_without_retry(caplog: pytest.LogCaptureFixture) -> None:
    # Per-candidate evidence failures are code rejections, not contract violations:
    # no retry, the verdict is forced to nothing while the model's verdict is kept.
    llm = _FakeLLM(_reply(_candidate(evidence_refs=[]), _candidate(title="Fabricated", evidence_refs=["ev123"])))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await _classify(llm)

    assert len(llm.payloads) == 1
    assert (result.verdict, result.model_verdict, result.calls) == ("nothing", "save", 1)
    assert result.candidates == []
    assert [(c.title, reason) for c, reason in result.rejected] == [
        ("Build before test", "empty evidence_refs"),
        ("Fabricated", "evidence not shown in the bundle: ['ev123']"),
    ]
    assert caplog.text.count("rejected candidate") == 2
    assert result.insufficient_evidence is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            _reply(verdict="nothing", decisions=[_decision(decision="reject", reason_code="insufficient_evidence")]),
            True,
        ),
        (
            _reply(
                verdict="nothing",
                decisions=[
                    _decision(decision="reject", reason_code="routine_activity"),
                    _decision(decision="reject", reason_code="insufficient_evidence", evidence_refs=[]),
                ],
            ),
            True,
        ),
        (_reply(verdict="nothing", decisions=[_decision(decision="reject", reason_code="routine_activity")]), False),
        (
            _reply(
                _candidate(evidence_refs=["ev99"]),
                decisions=[_decision(), _decision(decision="reject", reason_code="insufficient_evidence")],
            ),
            False,
        ),
    ],
    ids=["nothing-insufficient", "nothing-mixed", "nothing-routine", "save-all-code-rejected"],
)
async def test_insufficient_evidence_signal_requires_model_nothing(reply: str, expected: bool) -> None:
    llm = _FakeLLM(reply)

    result = await _classify(llm)

    assert result.verdict == "nothing"
    assert result.insufficient_evidence is expected


# -------------------------------------------------------------------- guards


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bundle", "valid_refs", "template", "episode_ids", "fragment"),
    [
        ("", VALID_REFS, TEMPLATE, EPISODE_IDS, "bundle is empty"),
        ("  \n", VALID_REFS, TEMPLATE, EPISODE_IDS, "bundle is empty"),
        (BUNDLE, set(), TEMPLATE, EPISODE_IDS, "valid_refs is empty"),
        (BUNDLE, VALID_REFS, TEMPLATE, set(), "episode_ids is empty"),
        (BUNDLE, VALID_REFS, "no placeholder here", EPISODE_IDS, "placeholder"),
        (BUNDLE, VALID_REFS, "{skill_library}\n" + TEMPLATE, EPISODE_IDS, "still contains {skill_library}"),
        (BUNDLE, VALID_REFS, TEMPLATE + "\n{selection_policy}", EPISODE_IDS, "still contains {selection_policy}"),
    ],
    ids=[
        "empty-bundle",
        "blank-bundle",
        "empty-valid-refs",
        "empty-episode-ids",
        "no-placeholder",
        "library-left",
        "policy-left",
    ],
)
async def test_guards_raise_before_calling_the_llm(
    bundle: str, valid_refs: set[str], template: str, episode_ids: set[str], fragment: str
) -> None:
    llm = _FakeLLM(_reply(_candidate()))

    with pytest.raises(ValueError, match=fragment.replace("{", r"\{").replace("}", r"\}")):
        await classify(bundle, valid_refs, llm, template, episode_ids=episode_ids)

    assert llm.payloads == []


def test_episode_ids_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        classify(BUNDLE, VALID_REFS, _FakeLLM(), TEMPLATE, EPISODE_IDS)  # type: ignore[misc]


# ------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('  {"a": 1}  ', '{"a": 1}'),
        ('```json\n{"a": 1}```', '{"a": 1}'),
        ('```{"a": 1}```', '```{"a": 1}```'),  # no newline after the fence: left alone
    ],
)
def test_strip_code_fence(text: str, expected: str) -> None:
    assert strip_code_fence(text) == expected


def test_strip_think_blocks() -> None:
    text = "<think>\nplan\n</think>\n{}\n<THINK>more</THINK>"
    assert strip_think_blocks(text).strip() == "{}"


def test_parse_classify_reply_errors() -> None:
    with pytest.raises(ClassifyParseError, match="invalid JSON"):
        parse_classify_reply("{")
    with pytest.raises(ClassifyParseError, match="must be an object, got list"):
        parse_classify_reply("[1]")
    with pytest.raises(ClassifyParseError, match="schema violation"):
        parse_classify_reply('{"verdict": "save"}')
    with pytest.raises(ClassifyParseError, match="schema violation"):
        parse_classify_reply('{"verdict": "nothing", "candidates": []}')  # v1 shape: no version, no decisions
    parsed = parse_classify_reply(_reply(verdict="nothing"))
    assert (parsed.contract_version, parsed.verdict, parsed.candidates) == (2, "nothing", [])
    assert [d.reason_code for d in parsed.decisions] == ["routine_activity"]


# -------------------------------------------------------------------- prompt


def test_packaged_classify_prompt_contract() -> None:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    assert len(lines) <= 170
    assert lines[0].startswith("## description: ")
    assert lines[1] == "## contract_version: 2"
    for placeholder in (BUNDLE_PLACEHOLDER, LIBRARY_PLACEHOLDER, POLICY_PLACEHOLDER):
        assert text.count(placeholder) == 1, placeholder
    # The model cites fragment ids, never bare event numbers.
    assert "[ev" in text
    assert "E<n>" in text
    assert "Evidence: seq" not in text
    assert "Observed procedure:" in text
    assert "observed_procedure" in text
    for key in (
        '"contract_version"',
        '"verdict"',
        '"candidates"',
        '"title"',
        '"rule"',
        '"evidence_refs"',
        '"applies_when"',
        '"constraints"',
        '"expected_outcome"',
        '"future_applicability"',
        '"target"',
        '"action"',
        '"existing_skill"',
        '"covered_by"',
        '"decisions"',
        '"episode_ids"',
        '"decision"',
        '"reason_code"',
        '"explanation"',
        '"save"',
        '"nothing"',
        '"accept"',
        '"reject"',
        '"create"',
        '"update"',
        '"reference"',
        '"high"',
        '"medium"',
        '"low"',
    ):
        assert key in text, key
    for code in sorted(REASON_CODES):
        assert f"`{code}`" in text, code
    # Selection criteria live only in the inserted policy block, so the fixed
    # text can never contradict a demo policy.
    for forbidden in ("Non-obviousness", "Novelty", "trivial"):
        assert forbidden not in text, forbidden
    for generation_marker in (
        "# Required SKILL.md structure",
        "## Inputs",
        "## Workflow",
        "## Outputs",
        "frontmatter",
        "Nothing to save.",
    ):
        assert generation_marker not in text, generation_marker


def test_packaged_v1_prompt_stays_for_hash_comparison() -> None:
    v1 = PROMPT_PATH.with_name("prompt_v1.md")
    assert v1.is_file()
    assert "contract_version" not in v1.read_text(encoding="utf-8")


# ----------------------------------------------------------------- isolation

_ISOLATION_PROBE = """
import sys
import msagent.skill_evolver.bundle as bundle
import msagent.skill_evolver.classify as classify
assert bundle.__file__.startswith(sys.argv[1]), bundle.__file__
assert classify.__file__.startswith(sys.argv[1]), classify.__file__
banned = ("langchain", "langgraph", "httpx", "requests", "urllib3", "aiohttp",
          "urllib.request", "http.client")
leaked = sorted(m for m in sys.modules if m.startswith(banned))
assert not leaked, leaked
"""


def test_bundle_and_classify_import_no_langchain_or_network() -> None:
    # Same setup as test_features: the pytest process already has langchain
    # loaded, so the check runs in a fresh interpreter with src/ first on the path.
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", _ISOLATION_PROBE, str(SRC_DIR)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
