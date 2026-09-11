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

"""Tests for the policy blocks and their assembly with the packaged prompts."""

from __future__ import annotations

from pathlib import Path

import pytest

from msagent.skill_evolver.policy import (
    DEMO_NAME_PREFIX,
    DEMO_WORKFLOW_BLOCK,
    GENERATION_POLICY_DEMO,
    GENERATION_POLICY_NORMAL,
    GENERATION_POLICY_PLACEHOLDER,
    REUSABLE_WORKFLOW_BLOCK,
    REVIEW_POLICY_DEMO,
    REVIEW_POLICY_NORMAL,
    REVIEW_POLICY_PLACEHOLDER,
    SELECTION_POLICY_PLACEHOLDER,
    STRICT_KNOWLEDGE_BLOCK,
    generation_policy_block,
    review_policy_block,
    selection_policy_block,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPTS = REPO_ROOT / "resources" / "configs" / "default" / "skill-evolver" / "prompts"
CLASSIFY_PROMPT = PROMPTS / "classify" / "prompt_v2.md"
GENERATION_PROMPT = PROMPTS / "generate" / "prompt_v2.md"
REVIEW_PROMPT = PROMPTS / "review" / "prompt_v2.md"

SELECTIONS = ("strict_knowledge", "reusable_workflow", "demo_workflow")
# A demo assembly must never carry a novelty or non-obviousness requirement
# next to the permission to save a trivial procedure (spec §3). Compared casefolded.
FORBIDDEN_IN_DEMO = ("non-obvious", "novelty", "not already in the library", "must be new")
MAX_BLOCK_LINES = 14
MAX_ASSEMBLY_LINES = 165


def _assert_none_of(text: str, phrases: tuple[str, ...], where: str) -> None:
    lowered = text.casefold()
    for phrase in phrases:
        assert phrase.casefold() not in lowered, f"{where} contains {phrase!r}"


# -------------------------------------------------------------------- blocks


def test_selection_policy_block_picks_one_block_per_selection() -> None:
    assert selection_policy_block("strict_knowledge") is STRICT_KNOWLEDGE_BLOCK
    assert selection_policy_block("reusable_workflow") is REUSABLE_WORKFLOW_BLOCK
    assert selection_policy_block("demo_workflow") is DEMO_WORKFLOW_BLOCK
    with pytest.raises(ValueError, match="unknown selection policy 'lenient'"):
        selection_policy_block("lenient")


@pytest.mark.parametrize("selection", SELECTIONS)
def test_selection_blocks_have_a_heading_and_no_placeholders(selection: str) -> None:
    block = selection_policy_block(selection)
    lines = block.splitlines()

    assert lines[0].startswith(f"# Selection policy: {selection}")
    assert len(lines) <= MAX_BLOCK_LINES
    assert "{" not in block and "}" not in block
    assert block == block.strip()


@pytest.mark.parametrize(
    "block", [GENERATION_POLICY_NORMAL, GENERATION_POLICY_DEMO, REVIEW_POLICY_NORMAL, REVIEW_POLICY_DEMO]
)
def test_generation_and_review_blocks_have_no_placeholders(block: str) -> None:
    assert "{" not in block and "}" not in block
    assert block == block.strip() and "\n" not in block


def test_strict_block_demands_non_obviousness_and_names_the_reason_codes() -> None:
    assert "non-obvious" in STRICT_KNOWLEDGE_BLOCK
    for code in (
        "routine_activity",
        "already_covered",
        "no_transferable_rule",
        "transient_observation",
        "unsafe_procedure",
    ):
        assert f"`{code}`" in STRICT_KNOWLEDGE_BLOCK, code
    assert "Several sessions are not required" in STRICT_KNOWLEDGE_BLOCK


def test_reusable_block_admits_standard_tools_and_demands_visible_steps() -> None:
    assert "A new technical discovery is not required" in REUSABLE_WORKFLOW_BLOCK
    assert "visible in the excerpts" in REUSABLE_WORKFLOW_BLOCK
    for code in ("routine_activity", "insufficient_evidence", "already_covered"):
        assert f"`{code}`" in REUSABLE_WORKFLOW_BLOCK, code
    _assert_none_of(REUSABLE_WORKFLOW_BLOCK, ("non-obvious",), "reusable block")


def test_demo_block_allows_triviality_and_keeps_the_evidence_standard() -> None:
    _assert_none_of(DEMO_WORKFLOW_BLOCK, FORBIDDEN_IN_DEMO, "demo block")
    assert "trivial" in DEMO_WORKFLOW_BLOCK
    assert "`create`" in DEMO_WORKFLOW_BLOCK and "`covered_by`" in DEMO_WORKFLOW_BLOCK
    assert "Never answer `update` or `reference`" in DEMO_WORKFLOW_BLOCK
    assert "The evidence standard is unchanged" in DEMO_WORKFLOW_BLOCK
    assert "`insufficient_evidence`" in DEMO_WORKFLOW_BLOCK and "`unsafe_procedure`" in DEMO_WORKFLOW_BLOCK


def test_generation_and_review_blocks_follow_the_demo_flag() -> None:
    assert generation_policy_block(False) == GENERATION_POLICY_NORMAL
    assert generation_policy_block(True) == GENERATION_POLICY_DEMO
    assert review_policy_block(False) == REVIEW_POLICY_NORMAL
    assert review_policy_block(True) == REVIEW_POLICY_DEMO
    assert DEMO_NAME_PREFIX == "demo-"
    assert f"`{DEMO_NAME_PREFIX}`" in GENERATION_POLICY_DEMO and "demo-csv-column-sum" in GENERATION_POLICY_DEMO
    assert "new skill" in GENERATION_POLICY_DEMO and "trivial" in GENERATION_POLICY_DEMO
    assert "demo" not in GENERATION_POLICY_NORMAL
    assert "Judge fidelity, not value" in REVIEW_POLICY_NORMAL and "Judge fidelity, not value" in REVIEW_POLICY_DEMO
    assert "triviality" in REVIEW_POLICY_NORMAL
    assert "`name_policy`" in REVIEW_POLICY_DEMO and f"`{DEMO_NAME_PREFIX}`" in REVIEW_POLICY_DEMO
    assert "create" in REVIEW_POLICY_DEMO
    assert "demo" not in REVIEW_POLICY_NORMAL
    _assert_none_of(GENERATION_POLICY_DEMO + REVIEW_POLICY_DEMO, FORBIDDEN_IN_DEMO, "demo generation/review blocks")


def test_placeholders_match_the_prompt_contract() -> None:
    assert SELECTION_POLICY_PLACEHOLDER == "{selection_policy}"
    assert GENERATION_POLICY_PLACEHOLDER == "{generation_policy}"
    assert REVIEW_POLICY_PLACEHOLDER == "{review_policy}"


# ---------------------------------------------------------------- assemblies


def _classify_assembly(selection: str) -> str:
    template = CLASSIFY_PROMPT.read_text(encoding="utf-8")
    assert template.count(SELECTION_POLICY_PLACEHOLDER) == 1
    return template.replace(SELECTION_POLICY_PLACEHOLDER, selection_policy_block(selection))


@pytest.mark.parametrize("selection", SELECTIONS)
def test_every_classify_assembly_has_exactly_one_policy_heading(selection: str) -> None:
    assembled = _classify_assembly(selection)
    lines = assembled.splitlines()

    assert sum(1 for line in lines if line.startswith("# Selection policy:")) == 1
    assert f"# Selection policy: {selection}" in assembled
    assert SELECTION_POLICY_PLACEHOLDER not in assembled
    assert len(lines) <= MAX_ASSEMBLY_LINES
    # The other blocks' headings never leak in.
    for other in SELECTIONS:
        if other != selection:
            assert f"# Selection policy: {other}" not in assembled


def test_demo_assembly_has_no_novelty_requirement() -> None:
    assembled = _classify_assembly("demo_workflow")

    _assert_none_of(assembled, FORBIDDEN_IN_DEMO, "demo classify assembly")
    assert "trivial" in assembled
    # The fixed classify text carries no selection criteria of its own: the strict
    # requirement appears only through the strict block.
    fixed = CLASSIFY_PROMPT.read_text(encoding="utf-8")
    _assert_none_of(fixed, FORBIDDEN_IN_DEMO, "fixed classify prompt")
    assert "non-obvious" in _classify_assembly("strict_knowledge")


def test_packaged_generation_and_review_fixed_texts_carry_no_novelty_requirement() -> None:
    generation = GENERATION_PROMPT.read_text(encoding="utf-8")
    review = REVIEW_PROMPT.read_text(encoding="utf-8")

    _assert_none_of(generation, FORBIDDEN_IN_DEMO, "fixed generation prompt")
    _assert_none_of(review, FORBIDDEN_IN_DEMO, "fixed review prompt")
    assert generation.count(GENERATION_POLICY_PLACEHOLDER) == 1
    assert review.count(REVIEW_POLICY_PLACEHOLDER) == 1
    demo_generation = generation.replace(GENERATION_POLICY_PLACEHOLDER, generation_policy_block(True))
    demo_review = review.replace(REVIEW_POLICY_PLACEHOLDER, review_policy_block(True))
    _assert_none_of(demo_generation, FORBIDDEN_IN_DEMO, "demo generation assembly")
    _assert_none_of(demo_review, FORBIDDEN_IN_DEMO, "demo review assembly")
    assert demo_generation.count(GENERATION_POLICY_DEMO) == 1 and GENERATION_POLICY_NORMAL not in demo_generation
    assert demo_review.count(REVIEW_POLICY_DEMO) == 1 and REVIEW_POLICY_NORMAL not in demo_review
