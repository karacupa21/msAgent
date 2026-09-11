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

"""LLM classification of an evidence bundle into knowledge candidates (contract v2).

The model receives the bundle built by :mod:`msagent.skill_evolver.bundle`
and answers with one JSON object (see the ``classify`` prompt). Everything
that makes the answer trustworthy happens here, in code:

- the reply is parsed strictly against :class:`ClassifyResult`;
- the parsed reply must honour the output contract: ``save`` needs at least
  one candidate, ``nothing`` an empty list, every decision cites only the
  episode ids (``E<n>``) and fragment ids (``ev<k>``) the bundle showed, and
  in demo mode every candidate is a ``create``;
- one corrective retry is spent per round on either failure kind, then
  :class:`ClassifyParseError` / :class:`ClassifyContractError` is raised —
  the reply is never repaired;
- every candidate must cite only fragment ids the bundle actually showed — a
  bare event number is not something the model has seen. Candidates failing
  that check are rejected individually with a reason (kept in
  :class:`Classification.rejected`, code ``invalid_evidence``) so the others
  survive; when none survives the verified verdict is ``nothing`` while
  ``model_verdict`` keeps what the model said.

Kept candidates get a ``candidate_id`` (``c1``, ``c2``, ...) that provenance
and the generation stage use to join them. The model's per-episode
``decisions`` (with a ``reason_code``) travel unchanged for the decision
report; a ``nothing`` verdict explained by ``insufficient_evidence`` is the
signal the pipeline may answer with one context-expanded round.

The template reaches :func:`classify` with ``{skill_library}`` and
``{selection_policy}`` already substituted by the caller (``str.replace``);
only ``{evidence_bundle}`` is filled here, also with ``str.replace`` so
braces inside the bundle stay inert.

The LLM is any object with ``async ainvoke(payload)`` that accepts a list of
``(role, text)`` pairs and returns an object exposing ``.text`` or
``.content`` (langchain chat models do, without being imported here).
Stdlib + pydantic only; this module never writes files.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, StrictStr, ValidationError, model_validator

logger = logging.getLogger(__name__)

# Version of the reply contract this module parses and verifies.
CONTRACT_VERSION = 2
# Placeholder of the prompt template that receives the bundle text.
BUNDLE_PLACEHOLDER = "{evidence_bundle}"
# Placeholders the caller must have substituted before calling classify().
LIBRARY_PLACEHOLDER = "{skill_library}"
POLICY_PLACEHOLDER = "{selection_policy}"
# Longest error text quoted back to the model in the corrective retry.
ERROR_TEXT_LIMIT = 2000
# Longest ``Decision.explanation`` the contract accepts.
EXPLANATION_LIMIT = 500
# Assistant turn replayed in the retry when the first reply was blank: some
# providers reject an empty assistant message.
EMPTY_REPLY = "(empty reply)"
# Rejection reasons of the evidence check (``Classification.rejected``).
REJECT_NO_EVIDENCE = "empty evidence_refs"
REJECT_NOT_SHOWN = "evidence not shown in the bundle"
# Code-rejection names the pipeline records for this stage.
CODE_PARSE_ERROR = "parse_error"
CODE_CONTRACT_ERROR = "contract_error"
CODE_INVALID_EVIDENCE = "invalid_evidence"

ReasonCode = Literal[
    "accepted",
    "routine_activity",
    "insufficient_evidence",
    "transient_observation",
    "already_covered",
    "no_transferable_rule",
    "unsafe_procedure",
]
REASON_CODES: frozenset[str] = frozenset(get_args(ReasonCode))

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CORRECTION = (
    "Your previous reply could not be parsed as the required JSON object.\n"
    "Error: {error}\n\n"
    "Reply again with only the JSON object described in the instructions: "
    "no markdown fences, no prose, no comments, nothing before or after it."
)
_CONTRACT_CORRECTION = (
    "Your previous reply is valid JSON but violates the output contract.\n"
    "Violations:\n{violations}\n\n"
    "Reply again with only the corrected JSON object described in the instructions: "
    "keep verdict and candidates consistent, cite only the E<n> and ev<k> ids shown in the bundle{demo_hint}."
)
_DEMO_HINT = ', and use target.action "create" for every candidate'


class ClassifyParseError(ValueError):
    """The reply is not the JSON object the prompt requires."""


class ClassifyContractError(ValueError):
    """Valid JSON that violates the output contract, still after one corrective retry."""


# ------------------------------------------------------------------- models


class CandidateTarget(BaseModel):
    """Where the knowledge belongs: a new skill or an existing one."""

    action: Literal["create", "update", "reference"]
    existing_skill: str | None = None

    @model_validator(mode="after")
    def require_existing_skill(self) -> CandidateTarget:
        """``update`` and ``reference`` point at a named library entry."""
        if self.action != "create" and not (self.existing_skill or "").strip():
            raise ValueError(f"target.action {self.action!r} requires existing_skill")
        return self


class Candidate(BaseModel):
    """One durable rule the model distilled from the bundle.

    ``evidence_refs`` are fragment ids of the bundle (``ev<N>``); StrictStr,
    so a seq number or a boolean never passes as a citation. The conditions
    of the rule — ``applies_when``, ``constraints``, ``expected_outcome`` —
    are optional: the prompt tells the model to leave them empty rather than
    invent them, and the generation prompt shows only the conditions that
    are filled. ``covered_by`` names the library skill that already covers
    the procedure when the policy still asks for a candidate (demo keeps
    ``target.action`` ``create``). ``candidate_id`` is assigned by
    :func:`classify` to the kept candidates, never by the model.
    """

    title: str
    rule: str
    evidence_refs: list[StrictStr]
    future_applicability: Literal["high", "medium", "low"]
    target: CandidateTarget
    applies_when: str | None = None
    constraints: list[str] = Field(default_factory=list)
    expected_outcome: str | None = None
    covered_by: str | None = None
    candidate_id: str = ""


class Decision(BaseModel):
    """The model's decision about one episode (or a group about the same rule).

    ``explanation`` is a short diagnostic for the decision report, not a
    reasoning trace; ``evidence_refs`` are the fragment ids it rests on.
    """

    episode_ids: list[StrictStr] = Field(min_length=1)
    decision: Literal["accept", "reject"]
    reason_code: ReasonCode
    explanation: str = Field(max_length=EXPLANATION_LIMIT)
    evidence_refs: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def accept_matches_reason(self) -> Decision:
        """``accept`` ⇔ ``accepted``."""
        if (self.decision == "accept") != (self.reason_code == "accepted"):
            raise ValueError(
                "decision 'accept' requires reason_code 'accepted', and 'accepted' requires decision 'accept'"
            )
        return self


class ClassifyResult(BaseModel):
    """The model's reply as replied; the contract between its parts is checked in code."""

    contract_version: Literal[2]
    verdict: Literal["save", "nothing"]
    candidates: list[Candidate]
    decisions: list[Decision] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class Classification:
    """The verified verdict: kept candidates (with ids), the rejected ones with why, and the model's decisions.

    ``verdict`` is forced to ``nothing`` when no candidate survives the
    evidence check; ``model_verdict`` keeps what the model replied. ``calls``
    counts the ``ainvoke`` calls this round cost (1 or 2).
    """

    verdict: Literal["save", "nothing"]
    candidates: list[Candidate]
    rejected: list[tuple[Candidate, str]]
    decisions: list[Decision] = field(default_factory=list)
    calls: int = 1
    model_verdict: Literal["save", "nothing"] = "nothing"

    @property
    def insufficient_evidence(self) -> bool:
        """The model itself found nothing and blamed the shown evidence for at least one episode."""
        return self.model_verdict == "nothing" and any(
            decision.reason_code == "insufficient_evidence" for decision in self.decisions
        )


# ------------------------------------------------------------------ parsing


def strip_think_blocks(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning blocks some models emit."""
    return _THINK_BLOCK.sub("", text)


def strip_code_fence(text: str) -> str:
    """Unwrap the whole text if the model fenced it despite instructions."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            return stripped[first_newline + 1 : -3].strip()
    return stripped


def parse_classify_reply(raw: str) -> ClassifyResult:
    """Parse one model reply strictly; raise ClassifyParseError otherwise."""
    text = strip_code_fence(strip_think_blocks(raw))
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClassifyParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        kind = type(data).__name__
        raise ClassifyParseError(f"top-level JSON value must be an object, got {kind}")
    try:
        return ClassifyResult.model_validate(data)
    except ValidationError as exc:
        raise ClassifyParseError(f"schema violation: {exc}") from exc


def reply_text(response: Any) -> str:
    """Text of an ``ainvoke`` result: ``.text``, else ``.content``, else str."""
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(response, "content", None)
    if content is not None:
        return content if isinstance(content, str) else str(content)
    return str(response)


# ------------------------------------------------------------- verification


def _unknown(cited: list[str], known: Collection[str]) -> list[str]:
    """Cited ids not in ``known``, deduplicated in citation order."""
    return [ref for ref in dict.fromkeys(cited) if ref not in known]


def _contract_violations(
    result: ClassifyResult,
    shown: Collection[str],
    episode_ids: Collection[str],
    *,
    demo: bool,
) -> list[str]:
    """Every way the parsed reply breaks the output contract; empty when it holds.

    A candidate citing unknown fragments is not a violation: :func:`_verify`
    rejects it individually so the other candidates survive.
    """
    violations: list[str] = []
    if result.verdict == "save" and not result.candidates:
        violations.append("verdict 'save' requires at least one candidate")
    if result.verdict == "nothing" and result.candidates:
        violations.append("verdict 'nothing' requires an empty candidates list")
    unknown_episodes = _unknown([eid for d in result.decisions for eid in d.episode_ids], episode_ids)
    if unknown_episodes:
        # "E2" before "E10": sort by length, then text.
        shown_ids = sorted(episode_ids, key=lambda eid: (len(eid), eid))
        violations.append(f"decisions cite unknown episode ids {unknown_episodes}; the bundle shows {shown_ids}")
    unknown_refs = _unknown([ref for d in result.decisions for ref in d.evidence_refs], shown)
    if unknown_refs:
        violations.append(f"decisions cite evidence ids not shown in the bundle: {unknown_refs}")
    if demo:
        for candidate in result.candidates:
            action = candidate.target.action
            if action != "create":
                violations.append(
                    f"candidate {candidate.title!r}: target.action {action!r} is not allowed in demo mode; "
                    "use 'create' and name the covering skill in covered_by"
                )
    return violations


def _verify(result: ClassifyResult, shown: Collection[str], *, calls: int) -> Classification:
    """Reject candidates whose evidence is empty or cites a fragment the bundle did not show."""
    kept: list[Candidate] = []
    rejected: list[tuple[Candidate, str]] = []
    for candidate in result.candidates:
        # Order-preserving dedupe: "ev10" must not sort before "ev2".
        refs = list(dict.fromkeys(candidate.evidence_refs))
        if not refs:
            rejected.append((candidate, REJECT_NO_EVIDENCE))
            continue
        unknown = [ref for ref in refs if ref not in shown]
        if unknown:
            rejected.append((candidate, f"{REJECT_NOT_SHOWN}: {unknown}"))
            continue
        update = {"evidence_refs": refs, "candidate_id": f"c{len(kept) + 1}"}
        kept.append(candidate.model_copy(update=update))
    for candidate, reason in rejected:
        logger.warning("classify: rejected candidate %r: %s", candidate.title, reason)
    verdict = result.verdict if kept else "nothing"
    return Classification(
        verdict,
        kept,
        rejected,
        decisions=list(result.decisions),
        calls=calls,
        model_verdict=result.verdict,
    )


# --------------------------------------------------------------- public API


async def classify(
    bundle: str,
    valid_refs: Collection[str],
    llm: Any,
    template: str,
    *,
    episode_ids: Collection[str],
    demo: bool = False,
) -> Classification:
    """Ask the LLM to classify ``bundle``; keep only verifiable candidates.

    ``valid_refs`` are the fragment ids of ``EvidenceBundle.shown`` and
    ``episode_ids`` the episode ids (``EvidenceBundle.episode_ids``) of this
    text. A candidate whose ``evidence_refs`` is empty, or cites any id
    outside ``valid_refs`` (an excluded event, a seq number, anything the
    model was not shown), is rejected with a reason and a warning; when
    nothing survives the verdict is ``nothing``.

    One corrective retry is spent on the first reply that is not the
    required JSON or breaks the contract (see :func:`_contract_violations`);
    a second failure raises :class:`ClassifyParseError` or
    :class:`ClassifyContractError`. Raises ``ValueError`` before any call
    when the bundle is blank, when ``valid_refs`` or ``episode_ids`` is
    empty, when the template has no ``{evidence_bundle}`` placeholder, or
    when it still contains ``{skill_library}`` / ``{selection_policy}``.
    """
    if not bundle.strip():
        raise ValueError("classify: the evidence bundle is empty")
    if not valid_refs:
        raise ValueError("classify: valid_refs is empty for a non-empty bundle")
    if not episode_ids:
        raise ValueError("classify: episode_ids is empty for a non-empty bundle")
    if BUNDLE_PLACEHOLDER not in template:
        raise ValueError(f"classify: template has no {BUNDLE_PLACEHOLDER} placeholder")
    for placeholder in (LIBRARY_PLACEHOLDER, POLICY_PLACEHOLDER):
        if placeholder in template:
            raise ValueError(f"classify: template still contains {placeholder}; substitute it before calling classify")

    # str.replace, never str.format: braces inside the bundle must stay inert.
    payload = [("human", template.replace(BUNDLE_PLACEHOLDER, bundle))]
    raw = reply_text(await llm.ainvoke(payload))
    try:
        result = parse_classify_reply(raw)
        violations = _contract_violations(result, valid_refs, episode_ids, demo=demo)
    except ClassifyParseError as first:
        logger.warning("classify: unparseable reply, retrying once: %s", first)
        correction = _CORRECTION.format(error=str(first)[:ERROR_TEXT_LIMIT])
    else:
        if not violations:
            return _verify(result, valid_refs, calls=1)
        logger.warning("classify: contract violation, retrying once: %s", "; ".join(violations))
        correction = _CONTRACT_CORRECTION.format(
            violations="\n".join(f"- {violation}" for violation in violations),
            demo_hint=_DEMO_HINT if demo else "",
        )

    payload = [
        *payload,
        ("ai", strip_think_blocks(raw).strip() or EMPTY_REPLY),
        ("human", correction),
    ]
    raw = reply_text(await llm.ainvoke(payload))
    try:
        result = parse_classify_reply(raw)
    except ClassifyParseError as second:
        raise ClassifyParseError(f"classify: reply is not valid JSON after one retry: {second}") from second
    violations = _contract_violations(result, valid_refs, episode_ids, demo=demo)
    if violations:
        raise ClassifyContractError("classify: reply violates the contract after one retry: " + "; ".join(violations))
    return _verify(result, valid_refs, calls=2)
