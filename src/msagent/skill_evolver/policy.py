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

"""Policy blocks inserted into the classify, generate and review prompts.

Rules:

* The fixed prompt texts carry no selection criteria; everything that
  decides *which* evidence deserves a candidate lives in exactly one block
  per stage, inserted at ``{selection_policy}`` (classify),
  ``{generation_policy}`` (generate) and ``{review_policy}`` (review). The
  ``demo_workflow`` block is the only one that admits a trivial procedure,
  so a prompt can never demand non-obviousness and allow triviality at once.
* The classify block is chosen by the effective selection
  (``strict_knowledge`` | ``reusable_workflow`` | ``demo_workflow``); the
  generation and review blocks by the effective demo flag. Callers pass
  plain values: this module imports nothing from the config.
* The evidence standard is the same under every block; demo lowers novelty,
  never evidence.
* Blocks contain no ``{`` / ``}`` and are inserted with ``str.replace`` or a
  one-pass regex, never ``str.format``.

Stdlib only.
"""

from __future__ import annotations

SELECTION_POLICY_PLACEHOLDER = "{selection_policy}"
GENERATION_POLICY_PLACEHOLDER = "{generation_policy}"
REVIEW_POLICY_PLACEHOLDER = "{review_policy}"
# Name prefix every demo proposal must carry (validator ``required_prefix``).
DEMO_NAME_PREFIX = "demo-"

STRICT_KNOWLEDGE_BLOCK = """# Selection policy: strict_knowledge

Save only proven, transferable, durable knowledge that is not an obvious use of ordinary tools and is not already covered by the library. The value is a non-obvious rule, condition, limit, technical mechanism or a substantial correction of a procedure.
- One episode is enough when it carries direct confirmation: an error followed by the fixed call and its result, an explicit correction, a mechanism visible in the excerpts. Several sessions are not required.
- A matching tool sequence is not proof of usefulness by itself; execution that merely ended well is `routine_activity`.
- A rule the library already states (same decision scope and workflow, not just a similar name) is `already_covered`: answer `reference`, or `update` only when the evidence changes the existing rule.
- `no_transferable_rule` for knowledge tied to one run (a path, id, output of that session); `transient_observation` for a state that may not hold next time; `unsafe_procedure` for destructive or credential-exposing steps.
- Keep the conditions under which the rule applies, replace one-time values with parameters, and cite the excerpts that prove the rule."""

REUSABLE_WORKFLOW_BLOCK = """# Selection policy: reusable_workflow

Save a useful, re-applicable procedure. Its individual steps may be standard tools; the value may lie in their selection, order, branching, applicability conditions or completion criteria. A new technical discovery is not required.
- A bare list of tools without inputs, decision logic and a checked result is not a procedure: `routine_activity`.
- Every step you keep must be visible in the excerpts (arguments and results). Do not add steps you did not see; `insufficient_evidence` when a step's result is missing, failed or only claimed.
- A procedure the library already describes without a substantive difference is `already_covered`: answer `reference`, or `update` only when the evidence shows a real change.
- Keep conditions, order and limits as observed, replace one-time values with parameters, and state the completion criterion the evidence shows."""

DEMO_WORKFLOW_BLOCK = """# Selection policy: demo_workflow (demonstration mode)

Demonstration mode: the rule does not have to be new, surprising or absent from the library. A trivial but confirmed procedure is a valid candidate, including one the library already covers.
- Accept an episode (typically `observed_procedure`) when the excerpts show the task context, the real arguments of each operation and its real, meaningful result; use `reason_code` `accepted`.
- A procedure a library skill already covers is still accepted as a separate teaching candidate: `target.action` must be `create`, `existing_skill` null, and the covering skill named in `covered_by`. Never answer `update` or `reference` in this mode.
- The evidence standard is unchanged: an orphan call, a missing or failing result, `status: ok` without a meaningful output, or an unconfirmed claim of success is `insufficient_evidence`; destructive or credential-exposing steps are `unsafe_procedure`.
- Keep all observed conditions, order and limits; replace one-time values with parameters; state how the result is checked."""

_SELECTION_BLOCKS = {
    "strict_knowledge": STRICT_KNOWLEDGE_BLOCK,
    "reusable_workflow": REUSABLE_WORKFLOW_BLOCK,
    "demo_workflow": DEMO_WORKFLOW_BLOCK,
}

GENERATION_POLICY_NORMAL = (
    "Selection policy: ordinary. The candidates passed the evidence gate on their own merit. "
    "Name a new skill after its task class as described in the Task section."
)
GENERATION_POLICY_DEMO = (
    "Selection policy: demo. The candidates describe a confirmed procedure that may be trivial or already "
    "common knowledge. Write it anyway, as a teaching skill: concrete inputs with their format, the exact "
    "operation with its real argument shape, the observed result and how to check it — never generic advice. "
    "A new skill's name MUST start with `demo-` followed by the task class (for example `demo-csv-column-sum`) "
    "and must not reuse the name of a library skill; the proposal is always a new skill, never an update. "
    "Do not mention demo mode inside the file."
)
REVIEW_POLICY_NORMAL = (
    "Review policy: ordinary. Judge fidelity, not value: check faithfulness to the candidates and evidence "
    "only; do not report triviality, novelty or library overlap as an issue."
)
REVIEW_POLICY_DEMO = (
    "Review policy: demo. Judge fidelity, not value: the skill may be trivial or already covered by the "
    "library; that is not an issue. Report `name_policy` when the name does not start with `demo-` or when "
    "the proposal is not a new skill (its target is not create). Judge faithfulness to the evidence exactly "
    "as in ordinary mode."
)


def selection_policy_block(selection: str) -> str:
    """The classify block of one effective selection; ``ValueError`` for an unknown one."""
    block = _SELECTION_BLOCKS.get(selection)
    if block is None:
        raise ValueError(f"unknown selection policy {selection!r}; expected one of {', '.join(_SELECTION_BLOCKS)}")
    return block


def generation_policy_block(demo: bool) -> str:
    """The ``{generation_policy}`` text for the effective demo flag."""
    return GENERATION_POLICY_DEMO if demo else GENERATION_POLICY_NORMAL


def review_policy_block(demo: bool) -> str:
    """The ``{review_policy}`` text for the effective demo flag."""
    return REVIEW_POLICY_DEMO if demo else REVIEW_POLICY_NORMAL
