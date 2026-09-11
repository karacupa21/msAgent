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

"""LLM generation of validated SKILL.md files from planned candidates.

:func:`plan_generation` splits the candidates the classify stage kept into
generation plans: one per new skill, one per library skill being updated;
every named target is resolved against the library once, and a candidate
whose target is unknown or ambiguous gets a rejection code, never a new plan.
:func:`generate_skill_md` turns one plan into a ``SKILL.md``: the model
receives its candidates (and, for an update, the text of the existing skill)
and answers with the complete file. The reply is checked by
:mod:`msagent.skill_evolver.validator`; on failure the model gets exactly one
corrective turn listing every error, and a second failure is handed back to
the caller, who writes nothing. :func:`revise_skill_md` is the single
corrective turn after a failed semantic review.

Evidence rules: there is no fixed cap on the fragments quoted per candidate.
:func:`select_plan_evidence` quotes every cited fragment when no budget is
given; under ``budget_chars`` the required fragments of all candidates are
placed first and the optional ones after, so a shortage drops context before
proof. A required fragment that does not fit makes the plan unusable
(:class:`InsufficientContextBudget`, raised before any LLM call); the
selection is recorded in provenance as ``generation_evidence`` /
``generation_evidence_omitted``, so an incomplete set is never marked
complete. The ``{generation_policy}`` placeholder is mandatory: the policy
text is inserted here with the same one-pass regex as the other placeholders.

The LLM is duck-typed exactly as in :mod:`msagent.skill_evolver.classify`.
Stdlib + pydantic; this module never writes files.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from msagent.skill_evolver.bundle import ShownFragment
from msagent.skill_evolver.classify import (
    EMPTY_REPLY,
    ERROR_TEXT_LIMIT,
    Candidate,
    reply_text,
    strip_code_fence,
    strip_think_blocks,
)
from msagent.skill_evolver.validator import ValidationResult, validate_skill_md
from msagent.skills.factory import Skill

logger = logging.getLogger(__name__)

CANDIDATES_PLACEHOLDER = "{candidates}"
EXISTING_SKILL_PLACEHOLDER = "{existing_skill}"
GENERATION_POLICY_PLACEHOLDER = "{generation_policy}"
# Text of the "Existing skill" section when the proposal is a new skill.
NO_EXISTING_SKILL = "None. Create a new skill."
# Rejection codes of plan_generation: the candidate names no library skill, or
# a bare name that exists in several categories.
INVALID_TARGET = "invalid_target"
AMBIGUOUS_TARGET = "ambiguous_target"
# Code rejection of a plan whose required evidence does not fit the evidence budget.
INSUFFICIENT_CONTEXT_BUDGET = "insufficient_context_budget"
# Characters one quoted evidence line costs beyond its text ("   - " + newline).
EVIDENCE_LINE_OVERHEAD = 6

# One pass over the template, so a placeholder-looking string inside a rule
# or inside the existing skill text is never substituted.
_PLACEHOLDER_RE = re.compile(r"\{(candidates|existing_skill|generation_policy)\}")
_PLACEHOLDERS = (CANDIDATES_PLACEHOLDER, EXISTING_SKILL_PLACEHOLDER, GENERATION_POLICY_PLACEHOLDER)
_CORRECTION = (
    "Your previous reply is not a valid SKILL.md:\n{errors}\n\n"
    "Reply again with the complete corrected SKILL.md: frontmatter and every "
    "section, no code fences, nothing before or after it."
)
_REVIEW_CORRECTION = (
    "A reviewer compared your SKILL.md with the accepted candidates and their evidence and found:\n"
    "{issues}\n\n"
    "Reply again with the complete corrected SKILL.md that fixes every point without adding anything the "
    "evidence does not support: frontmatter and every section, no code fences, nothing before or after it."
)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The last reply, its validation, the calls spent and what the generation call was quoted."""

    content: str
    validation: ValidationResult
    # 1 or 2 for generate_skill_md, 1 for revise_skill_md.
    calls: int
    # The whole conversation of the last attempt, the final ("ai", content) turn included.
    transcript: list[tuple[str, str]]
    # candidate_id -> fragment ids quoted / dropped for the budget.
    generation_evidence: dict[str, list[str]]
    generation_evidence_omitted: dict[str, list[str]]

    @property
    def ok(self) -> bool:
        return self.validation.ok


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    """The fragments quoted to the generation call for one candidate, and what was left out."""

    # Required first, then optional, each in citation order.
    fragments: list[ShownFragment]
    # Optional fragment ids dropped for the budget.
    omitted: list[str]
    # Required fragment ids that did not fit: the plan is unusable.
    missing_required: list[str]

    @property
    def ids(self) -> list[str]:
        return [fragment.id for fragment in self.fragments]

    @property
    def complete(self) -> bool:
        return not self.omitted and not self.missing_required

    @property
    def usable(self) -> bool:
        return not self.missing_required


class InsufficientContextBudget(ValueError):
    """Required generation evidence of a candidate does not fit ``budget_chars``."""

    def __init__(self, candidate_id: str, missing: list[str], budget_chars: int) -> None:
        self.candidate_id = candidate_id
        self.missing = list(missing)
        self.budget_chars = budget_chars
        super().__init__(
            f"generate: required evidence {self.missing} of candidate {candidate_id!r} does not fit the evidence "
            f"budget ({budget_chars} chars)"
        )


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """One generation call: its candidates and the library skill they revise."""

    # Classification order; one candidate for a create, every kept update of
    # ``existing`` otherwise.
    candidates: list[Candidate]
    # The library skill every candidate updates; None for a new skill.
    existing: Skill | None

    @property
    def label(self) -> str:
        """Console/error line: ``update <skill> (N candidates)`` or ``create: <title>``."""
        if self.existing is None:
            return f"create: {self.candidates[0].title}"
        count = len(self.candidates)
        noun = "candidate" if count == 1 else "candidates"
        return f"update {self.existing.display_name} ({count} {noun})"


@dataclass(frozen=True, slots=True)
class PlanRejection:
    """A candidate no plan takes: ``code`` is INVALID_TARGET or AMBIGUOUS_TARGET."""

    candidate: Candidate
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class GenerationPlans:
    """What plan_generation decided for one thread; every candidate is in exactly one list."""

    # Plans to generate a SKILL.md for, ordered by the first appearance of
    # their first candidate.
    plans: list[GenerationPlan]
    # Plans past ``max_plans`` with the reason; content and target untouched.
    deferred: list[tuple[GenerationPlan, str]]
    # ``reference`` candidates naming a library skill: reported; no SKILL.md
    # is generated for them.
    references: list[tuple[Candidate, Skill]]
    # Update/reference candidates whose target is unknown or ambiguous.
    rejected: list[PlanRejection]


def _resolve_target(
    wanted: str,
    by_display: Mapping[str, Skill],
    by_name: Mapping[str, Skill | None],
) -> Skill | str:
    """The library skill ``wanted`` names, or a rejection code.

    A display name wins; a bare name resolves only when it is unique across
    categories (``by_name`` stores None for a name found in several).
    """
    skill = by_display.get(wanted)
    if skill is not None:
        return skill
    if wanted in by_name:
        return by_name[wanted] or AMBIGUOUS_TARGET
    return INVALID_TARGET


def _rejection_detail(code: str, wanted: str) -> str:
    """Human text of a rejection code for one target name."""
    if code == AMBIGUOUS_TARGET:
        return f"existing_skill '{wanted}' is ambiguous"
    return f"existing_skill '{wanted}' is not in the skill library"


def plan_generation(
    candidates: Sequence[Candidate],
    skills: Sequence[Skill],
    *,
    max_plans: int,
) -> GenerationPlans:
    """Split the kept candidates into generation plans; resolve every named target once.

    Each ``create`` is its own plan (no merging), every ``update`` of one
    library skill shares a plan, and ``reference`` candidates are checked
    with the same resolver and reported; no SKILL.md is generated for them.
    A target must name a library skill (display name, or a bare name that is
    unique across categories); otherwise the candidate is rejected with a
    warning and never turned into a ``create``. Plans beyond ``max_plans``
    are deferred; nothing is generated for them.
    """
    by_display = {skill.display_name: skill for skill in skills}
    by_name: dict[str, Skill | None] = {}
    for skill in skills:
        by_name[skill.name] = None if skill.name in by_name else skill
    groups: list[tuple[Skill | None, list[Candidate]]] = []
    updates: dict[str, list[Candidate]] = {}
    references: list[tuple[Candidate, Skill]] = []
    rejected: list[PlanRejection] = []
    for candidate in candidates:
        target = candidate.target
        if target.action == "create":
            groups.append((None, [candidate]))
            continue
        wanted = (target.existing_skill or "").strip()
        resolved = _resolve_target(wanted, by_display, by_name)
        if isinstance(resolved, str):
            detail = _rejection_detail(resolved, wanted)
            logger.warning(
                "generate: dropped candidate %r: %s",
                candidate.title,
                detail,
            )
            rejected.append(PlanRejection(candidate, resolved, detail))
            continue
        if target.action == "reference":
            references.append((candidate, resolved))
            continue
        group = updates.get(resolved.display_name)
        if group is None:
            # Registered on the first update, so the plan keeps the position
            # of its first candidate while later updates still join it.
            group = updates[resolved.display_name] = []
            groups.append((resolved, group))
        group.append(candidate)
    ordered = [GenerationPlan(candidates=list(group), existing=skill) for skill, group in groups]
    reason = f"max_plans {max_plans} reached"
    deferred = [(plan, reason) for plan in ordered[max_plans:]]
    for plan, _ in deferred:
        logger.warning("generate: deferred plan %s: %s", plan.label, reason)
    return GenerationPlans(
        plans=ordered[:max_plans],
        deferred=deferred,
        references=references,
        rejected=rejected,
    )


def resolve_library_skill(name: str | None, skills: Sequence[Skill]) -> Skill | None:
    """The library skill ``name`` denotes (display name, or unique bare name), else None."""
    wanted = (name or "").strip()
    if not wanted:
        return None
    by_display = {skill.display_name: skill for skill in skills}
    by_name: dict[str, Skill | None] = {}
    for skill in skills:
        by_name[skill.name] = None if skill.name in by_name else skill
    resolved = _resolve_target(wanted, by_display, by_name)
    return None if isinstance(resolved, str) else resolved


def select_plan_evidence(
    candidates: Sequence[Candidate],
    evidence: Mapping[str, ShownFragment],
    *,
    budget_chars: int | None = None,
) -> list[EvidenceSelection]:
    """What each candidate of one plan gets quoted, under one shared budget.

    Unknown ids are ignored. Pass 1 places the required fragments of every
    candidate (citation order), pass 2 the optional ones; a fragment costs
    ``len(text) + EVIDENCE_LINE_OVERHEAD``. A required fragment that does not
    fit is reported in ``missing_required`` (every miss, not just the first),
    an optional one in ``omitted``. ``budget_chars`` None means no limit. A
    fragment two candidates cite is costed and quoted for each.
    """
    required: list[list[ShownFragment]] = []
    optional: list[list[ShownFragment]] = []
    for candidate in candidates:
        cited = [evidence[ref] for ref in candidate.evidence_refs if ref in evidence]
        required.append([fragment for fragment in cited if fragment.required])
        optional.append([fragment for fragment in cited if not fragment.required])
    taken: list[list[ShownFragment]] = [[] for _ in candidates]
    missing: list[list[str]] = [[] for _ in candidates]
    omitted: list[list[str]] = [[] for _ in candidates]
    remaining = budget_chars

    def place(fragment: ShownFragment) -> bool:
        nonlocal remaining
        cost = len(fragment.text) + EVIDENCE_LINE_OVERHEAD
        if remaining is not None and cost > remaining:
            return False
        if remaining is not None:
            remaining -= cost
        return True

    for index, fragments in enumerate(required):
        for fragment in fragments:
            (taken[index] if place(fragment) else missing[index]).append(fragment)
    for index, fragments in enumerate(optional):
        for fragment in fragments:
            if place(fragment):
                taken[index].append(fragment)
            else:
                omitted[index].append(fragment.id)
    return [
        EvidenceSelection(fragments, omitted[index], [fragment.id for fragment in missing[index]])
        for index, fragments in enumerate(taken)
    ]


def select_generation_evidence(
    candidate: Candidate,
    evidence: Mapping[str, ShownFragment],
    *,
    budget_chars: int | None = None,
) -> EvidenceSelection:
    """:func:`select_plan_evidence` for one candidate alone (provenance's unbudgeted fallback)."""
    return select_plan_evidence([candidate], evidence, budget_chars=budget_chars)[0]


def format_candidates(
    candidates: Sequence[Candidate],
    evidence: Mapping[str, ShownFragment] | None = None,
    *,
    selections: Sequence[EvidenceSelection] | None = None,
) -> str:
    """Numbered candidate blocks for the ``{candidates}`` placeholder.

    Each block carries the rule and its conditions (``When``,
    ``Constraints``, ``Expected outcome`` — only when the classifier filled
    them), the target, the library skill that already covers the procedure
    (``covered_by``), and the text of every selected evidence fragment plus
    a count of the fragments omitted for the budget. ``selections`` defaults
    to the unbudgeted selection. Fragment ids never appear: they belong in
    provenance, not in a prompt whose reply is the user-facing SKILL.md.
    """
    if selections is None:
        selections = select_plan_evidence(candidates, evidence or {})
    blocks: list[str] = []
    for number, (candidate, selection) in enumerate(zip(candidates, selections), start=1):
        target = candidate.target
        if target.action == "create":
            where = "create a new skill"
        else:
            where = f"{target.action} `{target.existing_skill}`"
        applicability = candidate.future_applicability
        lines = [
            f"{number}. {candidate.title} (future applicability: {applicability})",
            f"   Rule: {candidate.rule}",
        ]
        if candidate.applies_when:
            lines.append(f"   When: {candidate.applies_when}")
        if candidate.constraints:
            lines.append("   Constraints:")
            lines.extend(f"   - {constraint}" for constraint in candidate.constraints)
        if candidate.expected_outcome:
            lines.append(f"   Expected outcome: {candidate.expected_outcome}")
        lines.append(f"   Target: {where}")
        if candidate.covered_by:
            lines.append(
                f"   Covered by library skill: {candidate.covered_by} "
                "(write a separate teaching skill; do not copy the library text)"
            )
        if selection.fragments or selection.omitted:
            lines.append("   Evidence:")
            lines.extend(f"   - {fragment.text}" for fragment in selection.fragments)
            if selection.omitted:
                lines.append(f"   - ({len(selection.omitted)} context excerpts omitted for the prompt budget)")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def format_existing_skill(display_name: str, text: str) -> str:
    """Text of the ``{existing_skill}`` placeholder for an update."""
    intro = f"The candidates update the existing skill `{display_name}`; keep its name."
    return f"{intro} Current text:\n\n{text.strip()}\n"


def _clean(raw: str) -> str:
    """Strip reasoning blocks and a whole-reply fence; unify line endings."""
    text = strip_code_fence(strip_think_blocks(raw))
    return text.replace("\r\n", "\n").replace("\r", "\n")


async def _ask(llm: Any, payload: list[tuple[str, str]]) -> tuple[str, str]:
    """One LLM turn: the raw reply and its cleaned SKILL.md text."""
    raw = reply_text(await llm.ainvoke(payload))
    return raw, _clean(raw)


async def generate_skill_md(
    candidates: Sequence[Candidate],
    *,
    llm: Any,
    template: str,
    policy_text: str,
    existing_skill: str | None = None,
    expected_name: str | None = None,
    taken_names: Collection[str] = (),
    evidence: Mapping[str, ShownFragment] | None = None,
    required_prefix: str | None = None,
    evidence_budget_chars: int | None = None,
) -> GenerationResult:
    """Ask the LLM for a SKILL.md, validate it, correct once, return the last try.

    ``policy_text`` fills ``{generation_policy}``. ``existing_skill`` is the
    formatted text of the skill being updated and ``expected_name`` its name
    (both or neither). ``taken_names`` are library names a new skill must not
    reuse; ``required_prefix`` (new skills only) must start its name.
    ``evidence`` is the bundle's registry of shown fragments; each candidate
    is formatted with its own selection (:func:`select_plan_evidence` under
    ``evidence_budget_chars``). Raises ``ValueError`` before any LLM call when
    there is nothing to generate from, the template lacks a placeholder or the
    update arguments disagree, and :class:`InsufficientContextBudget` when a
    required fragment does not fit. Whether the content may be written is
    ``result.validation.ok``.
    """
    if not candidates:
        raise ValueError("generate: no candidates to generate from")
    missing = [p for p in _PLACEHOLDERS if p not in template]
    if missing:
        raise ValueError(f"generate: template has no {missing} placeholder")
    if (existing_skill is None) != (expected_name is None):
        raise ValueError("generate: existing_skill and expected_name go together")
    if required_prefix is not None and expected_name is not None:
        raise ValueError("generate: required_prefix applies to new skills only")
    selections = select_plan_evidence(candidates, evidence or {}, budget_chars=evidence_budget_chars)
    for candidate, selection in zip(candidates, selections):
        if not selection.usable:
            raise InsufficientContextBudget(candidate.candidate_id, selection.missing_required, evidence_budget_chars)
    generation_evidence = {c.candidate_id: s.ids for c, s in zip(candidates, selections)}
    generation_evidence_omitted = {c.candidate_id: list(s.omitted) for c, s in zip(candidates, selections)}

    values = {
        "candidates": format_candidates(candidates, evidence, selections=selections),
        "existing_skill": existing_skill or NO_EXISTING_SKILL,
        "generation_policy": policy_text,
    }
    instruction = _PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], template)
    payload: list[tuple[str, str]] = [("human", instruction)]
    raw, content = await _ask(llm, payload)
    result = validate_skill_md(
        content,
        expected_name=expected_name,
        taken_names=taken_names,
        required_prefix=required_prefix,
    )
    if result.ok:
        return GenerationResult(
            content,
            result,
            1,
            [*payload, ("ai", content)],
            generation_evidence,
            generation_evidence_omitted,
        )

    logger.warning("generate: invalid SKILL.md, retrying once: %s", result.errors)
    bullets = "\n".join(f"- {error[:ERROR_TEXT_LIMIT]}" for error in result.errors)
    payload = [
        *payload,
        ("ai", strip_think_blocks(raw).strip() or EMPTY_REPLY),
        ("human", _CORRECTION.format(errors=bullets)),
    ]
    raw, content = await _ask(llm, payload)
    result = validate_skill_md(
        content,
        expected_name=expected_name,
        taken_names=taken_names,
        required_prefix=required_prefix,
    )
    return GenerationResult(
        content,
        result,
        2,
        [*payload, ("ai", content or EMPTY_REPLY)],
        generation_evidence,
        generation_evidence_omitted,
    )


async def revise_skill_md(
    previous: GenerationResult,
    issues: Sequence[str],
    *,
    llm: Any,
    expected_name: str | None = None,
    taken_names: Collection[str] = (),
    required_prefix: str | None = None,
) -> GenerationResult:
    """The single corrective turn after a failed semantic review: one call, validated once.

    The reviewer's ``issues`` are appended to ``previous.transcript``; the
    evidence selection is the one ``previous`` was generated with.
    """
    bullets = "\n".join(f"- {issue[:ERROR_TEXT_LIMIT]}" for issue in issues)
    payload = [*previous.transcript, ("human", _REVIEW_CORRECTION.format(issues=bullets))]
    _, content = await _ask(llm, payload)
    result = validate_skill_md(
        content,
        expected_name=expected_name,
        taken_names=taken_names,
        required_prefix=required_prefix,
    )
    return GenerationResult(
        content,
        result,
        1,
        [*payload, ("ai", content or EMPTY_REPLY)],
        dict(previous.generation_evidence),
        dict(previous.generation_evidence_omitted),
    )
