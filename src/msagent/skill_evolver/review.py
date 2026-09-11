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

"""Semantic review of a generated SKILL.md against its candidates and evidence.

Rules:

* :func:`review_skill_md` shows the reviewer the accepted candidates (with
  their ids), the evidence fragments the generation call was quoted (with
  their ``ev`` ids — the reviewer must cite them), the existing skill text
  for an update, and the SKILL.md; it answers with one strict JSON object
  (:class:`ReviewReply`): ``pass`` with no issues, or ``fail`` with at least
  one coded issue. One parse retry is spent (the same idiom as classify and
  generate); a second failure is :class:`ReviewContractError`. An issue
  citing an id the generation call never saw is kept and reported in
  ``unknown_refs`` with a warning, never dropped.
* A ``pass`` means "faithful to the candidates and evidence", never "the
  procedure was executed" or "works everywhere": the generator runs nothing,
  which :data:`VERIFICATION_EVIDENCE_SUPPORTED` records in provenance.
* :func:`generate_and_review` is the per-plan stage both commands call:
  generate (≤ 2 calls) → validate → review (≤ 2) → on ``fail`` one corrective
  revision (1) → validate → review (1); at most 6 ``ainvoke`` per plan. A plan
  whose required evidence does not fit the budget, or whose selection is
  empty, is refused before any call (``insufficient_context_budget``).
  :class:`~msagent.skill_evolver.budget.LlmBudgetExhausted` is not caught
  here; it propagates before anything is written.
* The ``{review_policy}`` placeholder is mandatory; the policy text is
  inserted with the one-pass regex, never ``str.format``.

The LLM is duck-typed as in :mod:`msagent.skill_evolver.classify`. Stdlib +
pydantic; this module never writes files.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, StrictStr, ValidationError, model_validator

from msagent.skill_evolver.bundle import ShownFragment
from msagent.skill_evolver.classify import (
    EMPTY_REPLY,
    ERROR_TEXT_LIMIT,
    Candidate,
    reply_text,
    strip_code_fence,
    strip_think_blocks,
)
from msagent.skill_evolver.generate import (
    INSUFFICIENT_CONTEXT_BUDGET,
    NO_EXISTING_SKILL,
    GenerationResult,
    generate_skill_md,
    revise_skill_md,
    select_plan_evidence,
)
from msagent.skill_evolver.validator import ValidationResult

logger = logging.getLogger(__name__)

SKILL_MD_PLACEHOLDER = "{skill_md}"
CANDIDATES_PLACEHOLDER = "{candidates}"
EVIDENCE_PLACEHOLDER = "{evidence}"
EXISTING_SKILL_PLACEHOLDER = "{existing_skill}"
REVIEW_POLICY_PLACEHOLDER = "{review_policy}"
_PLACEHOLDERS = (
    SKILL_MD_PLACEHOLDER,
    CANDIDATES_PLACEHOLDER,
    EVIDENCE_PLACEHOLDER,
    EXISTING_SKILL_PLACEHOLDER,
    REVIEW_POLICY_PLACEHOLDER,
)
# One pass, so a placeholder-looking string inside the SKILL.md or a rule is never substituted.
_PLACEHOLDER_RE = re.compile(r"\{(skill_md|candidates|evidence|existing_skill|review_policy)\}")
# Code rejections of generate_and_review.
GENERATION_INVALID = "generation_invalid"
QUALITY_REVIEW_FAILED = "quality_review_failed"
# Provenance ``verification`` of every proposal this stage passes: nothing was executed.
VERIFICATION_EVIDENCE_SUPPORTED: dict[str, str] = {"level": "evidence_supported", "note": "not executed by generator"}

IssueCode = Literal[
    "lost_condition",
    "order_changed",
    "command_mismatch",
    "unsupported_addition",
    "missing_rule",
    "unsafe_claim",
    "one_time_value",
    "name_policy",
    "other",
]
ISSUE_CODES: frozenset[str] = frozenset(get_args(IssueCode))

_CORRECTION = (
    "Your previous reply is not the JSON object the review contract requires.\n"
    "Error: {error}\n\n"
    "Reply again with only the JSON object described in the instructions: "
    "no markdown fences, no prose, nothing before or after it."
)


class ReviewParseError(ValueError):
    """The reply is not the JSON object the review prompt requires."""


class ReviewContractError(ValueError):
    """The reviewer gave no usable reply within the allowed attempts; ``calls`` were spent."""

    def __init__(self, message: str, *, calls: int) -> None:
        self.calls = calls
        super().__init__(message)


# ------------------------------------------------------------------- models


class ReviewIssue(BaseModel):
    """One concrete deviation of the SKILL.md from the candidates or evidence."""

    code: IssueCode
    detail: str
    evidence_refs: list[StrictStr] = Field(default_factory=list)


class ReviewReply(BaseModel):
    """The reviewer's reply as replied; ``pass`` and ``issues`` must agree."""

    verdict: Literal["pass", "fail"]
    issues: list[ReviewIssue]

    @model_validator(mode="after")
    def verdict_matches_issues(self) -> ReviewReply:
        if self.verdict == "pass" and self.issues:
            raise ValueError("verdict 'pass' requires an empty issues list")
        if self.verdict == "fail" and not self.issues:
            raise ValueError("verdict 'fail' requires at least one issue")
        return self


@dataclass(frozen=True, slots=True)
class ReviewResult:
    """One review round: the verdict, its issues, the cited ids the generation call never saw, the calls spent."""

    verdict: Literal["pass", "fail"]
    issues: list[ReviewIssue]
    unknown_refs: list[str]
    calls: int

    def issue_lines(self) -> list[str]:
        return [f"{issue.code}: {issue.detail}" for issue in self.issues]

    def record(self) -> dict[str, Any]:
        """JSON-safe record for provenance and the decision report."""
        return {
            "verdict": self.verdict,
            "issues": [issue.model_dump() for issue in self.issues],
            "unknown_refs": list(self.unknown_refs),
            "calls": self.calls,
        }


@dataclass(frozen=True, slots=True)
class GeneratedSkill:
    """Outcome of :func:`generate_and_review` for one plan; ``ok`` when ``code`` is None."""

    content: str | None
    validation: ValidationResult | None
    review: ReviewResult | None
    generation_evidence: dict[str, list[str]]
    generation_evidence_omitted: dict[str, list[str]]
    calls: int
    # GENERATION_INVALID | QUALITY_REVIEW_FAILED | INSUFFICIENT_CONTEXT_BUDGET, None on success.
    code: str | None = None
    errors: list[str] = field(default_factory=list)
    # ``corrected``: the plan passed only after the one corrective revision. ``initial_issues`` is
    # what the first review found whenever a corrective revision ran — on the pass and the fail path.
    corrected: bool = False
    initial_issues: list[str] = field(default_factory=list)
    # The last SKILL.md the model produced when the plan was refused (generation_invalid, quality
    # review failed): kept for the decision report's rejected-draft file, never written as a proposal.
    draft: str | None = None

    @property
    def ok(self) -> bool:
        return self.code is None

    def review_record(self) -> dict[str, Any]:
        """Provenance ``quality_review``: the final review plus the correction history."""
        record = self.review.record() if self.review is not None else {}
        return {**record, "corrected": self.corrected, "initial_issues": list(self.initial_issues)}


# ------------------------------------------------------------------ parsing


def parse_review_reply(raw: str) -> ReviewReply:
    """Parse one reviewer reply strictly; raise ReviewParseError otherwise."""
    text = strip_code_fence(strip_think_blocks(raw))
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReviewParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ReviewParseError(f"top-level JSON value must be an object, got {type(data).__name__}")
    try:
        return ReviewReply.model_validate(data)
    except ValidationError as exc:
        raise ReviewParseError(f"schema violation: {exc}") from exc


# --------------------------------------------------------------- formatters


def format_review_candidates(candidates: Sequence[Candidate]) -> str:
    """Candidate blocks for the reviewer: the same conditions as the generation call saw, plus ids and citations."""
    blocks: list[str] = []
    for number, candidate in enumerate(candidates, start=1):
        target = candidate.target
        where = target.action if target.action == "create" else f"{target.action} `{target.existing_skill}`"
        lines = [
            f"{number}. [{candidate.candidate_id}] {candidate.title} — target {where}",
            f"   Rule: {candidate.rule}",
        ]
        if candidate.applies_when:
            lines.append(f"   When: {candidate.applies_when}")
        if candidate.constraints:
            lines.append("   Constraints:")
            lines.extend(f"   - {constraint}" for constraint in candidate.constraints)
        if candidate.expected_outcome:
            lines.append(f"   Expected outcome: {candidate.expected_outcome}")
        lines.append(f"   Cites: {', '.join(candidate.evidence_refs)}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def format_review_evidence(fragments: Sequence[ShownFragment]) -> str:
    """One line per fragment the generation call saw, ids shown, order kept, duplicates by id dropped."""
    lines: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        if fragment.id in seen:
            continue
        seen.add(fragment.id)
        kind = "required" if fragment.required else "context"
        lines.append(f"- [{fragment.id}] ({kind}) {fragment.text}")
    return "\n".join(lines)


# --------------------------------------------------------------- public API


async def review_skill_md(
    skill_md: str,
    candidates: Sequence[Candidate],
    fragments: Sequence[ShownFragment],
    existing_skill_text: str | None,
    *,
    llm: Any,
    template: str,
    policy_text: str,
    corrective_retry: bool = True,
) -> ReviewResult:
    """Ask the reviewer whether ``skill_md`` is faithful to ``candidates`` and ``fragments``.

    ``fragments`` are exactly what the generation call was quoted;
    ``existing_skill_text`` is the formatted existing skill of an update (None for
    a new skill); ``policy_text`` fills ``{review_policy}``. One parse retry when
    ``corrective_retry``; a further failure raises :class:`ReviewContractError`.
    Raises ``ValueError`` before any call for a blank SKILL.md, no candidates,
    no fragments or a template missing a placeholder.
    """
    if not skill_md.strip():
        raise ValueError("review: SKILL.md is empty")
    if not candidates:
        raise ValueError("review: no candidates to review")
    if not fragments:
        raise ValueError("review: no evidence fragments to review against")
    missing = [p for p in _PLACEHOLDERS if p not in template]
    if missing:
        raise ValueError(f"review: template has no {missing} placeholder")

    values = {
        "skill_md": skill_md,
        "candidates": format_review_candidates(candidates),
        "evidence": format_review_evidence(fragments),
        "existing_skill": existing_skill_text or NO_EXISTING_SKILL,
        "review_policy": policy_text,
    }
    payload: list[tuple[str, str]] = [("human", _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template))]
    raw = reply_text(await llm.ainvoke(payload))
    calls = 1
    try:
        reply = parse_review_reply(raw)
    except ReviewParseError as first:
        if not corrective_retry:
            raise ReviewContractError(
                f"review: reply is not the required JSON after 1 attempt(s): {first}", calls=calls
            ) from first
        logger.warning("review: unparseable reply, retrying once: %s", first)
        payload = [
            *payload,
            ("ai", strip_think_blocks(raw).strip() or EMPTY_REPLY),
            ("human", _CORRECTION.format(error=str(first)[:ERROR_TEXT_LIMIT])),
        ]
        raw = reply_text(await llm.ainvoke(payload))
        calls = 2
        try:
            reply = parse_review_reply(raw)
        except ReviewParseError as second:
            raise ReviewContractError(
                f"review: reply is not the required JSON after 2 attempt(s): {second}", calls=calls
            ) from second

    known = {fragment.id for fragment in fragments}
    cited = [ref for issue in reply.issues for ref in issue.evidence_refs]
    unknown_refs = [ref for ref in dict.fromkeys(cited) if ref not in known]
    if unknown_refs:
        logger.warning("review: issues cite evidence not shown to the generation call: %s", unknown_refs)
    return ReviewResult(reply.verdict, list(reply.issues), unknown_refs, calls)


def _quoted_fragments(candidates: Sequence[Candidate], selections: Sequence[Any]) -> list[ShownFragment]:
    """The union of what the generation call saw, in quoting order, deduplicated by id."""
    fragments: list[ShownFragment] = []
    seen: set[str] = set()
    for selection in selections:
        for fragment in selection.fragments:
            if fragment.id not in seen:
                seen.add(fragment.id)
                fragments.append(fragment)
    return fragments


def _failed(
    code: str,
    errors: list[str],
    *,
    generated: GenerationResult | None,
    review: ReviewResult | None,
    calls: int,
    initial_issues: Sequence[str] = (),
) -> GeneratedSkill:
    """A rejected plan: nothing may be written.

    ``generated`` keeps the selection, validation and draft for the record.
    """
    return GeneratedSkill(
        content=None,
        validation=generated.validation if generated is not None else None,
        review=review,
        generation_evidence=dict(generated.generation_evidence) if generated is not None else {},
        generation_evidence_omitted=dict(generated.generation_evidence_omitted) if generated is not None else {},
        calls=calls,
        code=code,
        errors=errors,
        initial_issues=list(initial_issues),
        draft=generated.content if generated is not None else None,
    )


async def generate_and_review(
    candidates: Sequence[Candidate],
    *,
    llm: Any,
    generation_template: str,
    review_template: str,
    generation_policy: str,
    review_policy: str,
    evidence: Mapping[str, ShownFragment],
    existing_skill: str | None = None,
    existing_skill_text: str | None = None,
    expected_name: str | None = None,
    taken_names: Collection[str] = (),
    required_prefix: str | None = None,
    evidence_budget_chars: int | None = None,
) -> GeneratedSkill:
    """Generate one plan's SKILL.md, validate it, review it, correct once; at most 6 LLM calls.

    ``existing_skill`` is the formatted existing skill for the generation
    call and ``existing_skill_text`` the same for the reviewer (both None
    for a new skill). The outcome's ``code`` is None when the content may
    be written, else ``generation_invalid``, ``quality_review_failed`` or
    ``insufficient_context_budget`` (the last before any call).
    """
    selections = select_plan_evidence(candidates, evidence, budget_chars=evidence_budget_chars)
    short = [
        f"candidate {c.candidate_id}: required evidence {s.missing_required} does not fit the evidence budget"
        for c, s in zip(candidates, selections)
        if not s.usable
    ]
    if short:
        return _failed(INSUFFICIENT_CONTEXT_BUDGET, short, generated=None, review=None, calls=0)
    fragments = _quoted_fragments(candidates, selections)
    if not fragments:
        errors = ["no evidence fragment fits the evidence budget; nothing to generate from or review against"]
        return _failed(INSUFFICIENT_CONTEXT_BUDGET, errors, generated=None, review=None, calls=0)

    generated = await generate_skill_md(
        candidates,
        llm=llm,
        template=generation_template,
        policy_text=generation_policy,
        existing_skill=existing_skill,
        expected_name=expected_name,
        taken_names=taken_names,
        evidence=evidence,
        required_prefix=required_prefix,
        evidence_budget_chars=evidence_budget_chars,
    )
    calls = generated.calls
    if not generated.ok:
        return _failed(
            GENERATION_INVALID, list(generated.validation.errors), generated=generated, review=None, calls=calls
        )

    try:
        first = await review_skill_md(
            generated.content,
            candidates,
            fragments,
            existing_skill_text,
            llm=llm,
            template=review_template,
            policy_text=review_policy,
        )
    except ReviewContractError as exc:
        calls += exc.calls
        return _failed(
            QUALITY_REVIEW_FAILED, [f"reviewer reply invalid: {exc}"], generated=generated, review=None, calls=calls
        )
    calls += first.calls
    if first.verdict == "pass":
        return GeneratedSkill(
            content=generated.content,
            validation=generated.validation,
            review=first,
            generation_evidence=dict(generated.generation_evidence),
            generation_evidence_omitted=dict(generated.generation_evidence_omitted),
            calls=calls,
        )

    logger.warning("review: SKILL.md failed the quality review, correcting once: %s", first.issue_lines())
    revised = await revise_skill_md(
        generated,
        first.issue_lines(),
        llm=llm,
        expected_name=expected_name,
        taken_names=taken_names,
        required_prefix=required_prefix,
    )
    calls += revised.calls
    initial = first.issue_lines()
    if not revised.ok:
        errors = [f"corrective SKILL.md invalid: {error}" for error in revised.validation.errors]
        return _failed(
            QUALITY_REVIEW_FAILED, errors, generated=revised, review=first, calls=calls, initial_issues=initial
        )
    try:
        second = await review_skill_md(
            revised.content,
            candidates,
            fragments,
            existing_skill_text,
            llm=llm,
            template=review_template,
            policy_text=review_policy,
            corrective_retry=False,
        )
    except ReviewContractError as exc:
        calls += exc.calls
        errors = [f"reviewer reply invalid: {exc}"]
        return _failed(
            QUALITY_REVIEW_FAILED, errors, generated=revised, review=first, calls=calls, initial_issues=initial
        )
    calls += second.calls
    if second.verdict == "fail":
        return _failed(
            QUALITY_REVIEW_FAILED,
            second.issue_lines(),
            generated=revised,
            review=second,
            calls=calls,
            initial_issues=initial,
        )
    return GeneratedSkill(
        content=revised.content,
        validation=revised.validation,
        review=second,
        generation_evidence=dict(revised.generation_evidence),
        generation_evidence_omitted=dict(revised.generation_evidence_omitted),
        calls=calls,
        corrected=True,
        initial_issues=first.issue_lines(),
    )
