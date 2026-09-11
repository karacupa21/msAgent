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

"""Tests for skill_evolver.prompts: contract header, resolution, legacy rule, hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import msagent.cli.handlers  # noqa: F401
from msagent.skill_evolver import prompts as module
from msagent.skill_evolver.config import STAGES, DirectSkillGenerationConfig
from msagent.skill_evolver.prompts import (
    CONTRACT_HEADER_RE,
    REQUIRED_PLACEHOLDERS,
    PromptContractError,
    PromptText,
    StagePrompts,
    declared_contract,
    packaged_prompts_root,
    prompt_sha256,
    resolve_stage_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_PACKAGED = REPO_ROOT / "resources" / "configs" / "default" / "skill-evolver" / "prompts"

V1_TEXT = "## description: old\n\nold {skill_library} {evidence_bundle}\n"


def _v2(stage: str, *, description: str = "test prompt") -> str:
    body = "\n".join(REQUIRED_PLACEHOLDERS[stage])
    return f"## description: {description}\n## contract_version: 2\n\n# Role\n{body}\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake packaged prompts root: prompt_v2.md for every stage, prompt_v1.md for classify and generate."""
    root = tmp_path / "packaged" / "prompts"
    for stage in STAGES:
        _write(root / stage / "prompt_v2.md", _v2(stage, description=f"packaged {stage}"))
    for stage in ("classify", "generate"):
        _write(root / stage / "prompt_v1.md", V1_TEXT)
    monkeypatch.setattr(module, "packaged_prompts_root", lambda: root)
    return root


@pytest.fixture
def user_root(tmp_path: Path) -> Path:
    return tmp_path / "user" / "prompts"


LEGACY_CFG = DirectSkillGenerationConfig(legacy_format=True, prompt_file="prompt_v1.md")


# ------------------------------------------------------------------- helpers


def test_prompt_sha256_and_declared_contract() -> None:
    text = "## description: x\n## contract_version: 2\nbody {a}\n"

    assert prompt_sha256(text) == "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert declared_contract(text) == 2
    assert declared_contract("## description: x\n\nno header\n") == 1
    assert declared_contract("## contract_version: 7\n") == 7
    assert declared_contract("\n\n\n\n\n## contract_version: 2\n") == 1  # only the first five lines count
    assert CONTRACT_HEADER_RE.match("## contract_version:2  ")
    assert not CONTRACT_HEADER_RE.match("## contract_version: two")


def test_stage_prompts_accessors() -> None:
    texts = {
        stage: PromptText(stage, f"{stage} text", f"/p/{stage}.md", prompt_sha256(f"{stage} text"), 2)
        for stage in STAGES
    }
    prompts = StagePrompts(**texts)

    assert prompts.variants() == {"classify": "/p/classify.md", "generate": "/p/generate.md", "review": "/p/review.md"}
    assert prompts.hashes() == {stage: prompt_sha256(f"{stage} text") for stage in STAGES}
    assert prompts.get("review") is texts["review"]
    with pytest.raises(ValueError, match="Unknown prompt stage 'get'"):
        prompts.get("get")


# ---------------------------------------------------------------- resolution


def test_user_file_wins_over_packaged_then_packaged(packaged: Path, user_root: Path) -> None:
    user = _write(user_root / "classify" / "prompt_v2.md", _v2("classify", description="user classify"))
    notes: list[str] = []

    mine = resolve_stage_prompt(user_root, DirectSkillGenerationConfig(), "classify", notes=notes)
    theirs = resolve_stage_prompt(user_root, DirectSkillGenerationConfig(), "generate", notes=notes)

    assert (mine.source, mine.stage, mine.contract_version) == (str(user), "classify", 2)
    assert mine.text == user.read_text(encoding="utf-8")
    assert mine.sha256 == prompt_sha256(mine.text)
    assert theirs.source == str(packaged / "generate" / "prompt_v2.md")
    assert notes == []


def test_missing_prompt_file_is_an_error_not_a_glob(packaged: Path, user_root: Path) -> None:
    _write(user_root / "classify" / "a.md", _v2("classify"))
    _write(user_root / "classify" / "b.md", _v2("classify"))
    cfg = DirectSkillGenerationConfig(classify_prompt="missing.md")

    with pytest.raises(ValueError, match="prompt file not found") as info:
        resolve_stage_prompt(user_root, cfg, "classify", notes=[])

    message = str(info.value)
    assert str(user_root / "classify" / "missing.md") in message
    assert str(packaged / "classify" / "missing.md") in message
    assert "prompts.classify" in message
    assert not isinstance(info.value, PromptContractError)


def test_unknown_stage_is_rejected(packaged: Path, user_root: Path) -> None:
    with pytest.raises(ValueError, match="Unknown prompt stage 'bogus'"):
        resolve_stage_prompt(user_root, DirectSkillGenerationConfig(), "bogus", notes=[])


def test_stage_prompt_requires_declared_contract(packaged: Path, user_root: Path) -> None:
    user = _write(
        user_root / "classify" / "prompt_v2.md",
        "## description: mine\n\n{skill_library} {evidence_bundle} {selection_policy}\n",
    )
    before = user.read_bytes()

    with pytest.raises(PromptContractError) as info:
        resolve_stage_prompt(user_root, DirectSkillGenerationConfig(), "classify", notes=[])

    message = str(info.value)
    assert str(user) in message
    assert "declares contract_version 1" in message
    assert "prompts.contract_version is 2" in message
    assert f"copy {packaged / 'classify' / 'prompt_v2.md'} to {user_root / 'classify'}/" in message
    assert "set prompts.classify: prompt_v2.md" in message
    assert "not modified" in message
    assert user.read_bytes() == before

    user.write_text(_v2("classify"), encoding="utf-8")
    good = resolve_stage_prompt(user_root, DirectSkillGenerationConfig(), "classify", notes=[])
    assert good.sha256 == "sha256:" + hashlib.sha256(_v2("classify").encode("utf-8")).hexdigest()


def test_v2_config_pointing_at_a_contract_1_file_gets_the_migration_text(packaged: Path, user_root: Path) -> None:
    cfg = DirectSkillGenerationConfig(generate_prompt="prompt_v1.md")

    with pytest.raises(PromptContractError, match="declares contract_version 1") as info:
        resolve_stage_prompt(user_root, cfg, "generate", notes=[])

    assert "set prompts.generate: prompt_v2.md" in str(info.value)


@pytest.mark.parametrize("stage", STAGES)
def test_required_placeholders_are_checked(packaged: Path, user_root: Path, stage: str) -> None:
    for missing in REQUIRED_PLACEHOLDERS[stage]:
        text = _v2(stage).replace(missing, "")
        _write(user_root / stage / "prompt_v2.md", text)

        with pytest.raises(PromptContractError, match=f"lacks the {missing} placeholder required by contract 2"):
            resolve_stage_prompt(user_root, DirectSkillGenerationConfig(), stage, notes=[])


# --------------------------------------------------------------- legacy rule


def test_legacy_prompt_file_rule(packaged: Path, user_root: Path) -> None:
    notes: list[str] = []

    # No user copy: packaged prompt_v2.md, one note.
    absent = resolve_stage_prompt(user_root, LEGACY_CFG, "classify", notes=notes)
    assert absent.source == str(packaged / "classify" / "prompt_v2.md")
    assert absent.contract_version == 2
    assert notes == [
        "prompt_file 'prompt_v1.md' is a legacy setting; using packaged prompts/classify/prompt_v2.md (contract 2)"
    ]

    # Byte-identical seeded copy: still packaged prompt_v2.md.
    _write(user_root / "generate" / "prompt_v1.md", V1_TEXT)
    identical = resolve_stage_prompt(user_root, LEGACY_CFG, "generate", notes=notes)
    assert identical.source == str(packaged / "generate" / "prompt_v2.md")
    assert len(notes) == 2

    # Review stage never had a prompt_v1.md: packaged prompt_v2.md as well.
    review = resolve_stage_prompt(user_root, LEGACY_CFG, "review", notes=notes)
    assert review.source == str(packaged / "review" / "prompt_v2.md")

    # Edited copy: refused with the migration text; the file is untouched.
    edited = _write(user_root / "classify" / "prompt_v1.md", V1_TEXT + "my edit\n")
    before = edited.read_bytes()
    with pytest.raises(PromptContractError) as info:
        resolve_stage_prompt(user_root, LEGACY_CFG, "classify", notes=notes)
    message = str(info.value)
    assert message.startswith(f"{edited} is a customised contract-1 prompt; this build needs contract 2.")
    assert f"copy {packaged / 'classify' / 'prompt_v2.md'} to {user_root / 'classify'}/" in message
    assert "set prompts.classify: prompt_v2.md in config.skill.evolver.yml (schema_version 2)" in message
    assert edited.read_bytes() == before
    assert len(notes) == 3


def test_legacy_rule_needs_both_legacy_format_and_the_packaged_name(packaged: Path, user_root: Path) -> None:
    # legacy_format with another prompt_file: plain resolution of that name.
    other = DirectSkillGenerationConfig(legacy_format=True, prompt_file="custom.md")
    _write(user_root / "classify" / "custom.md", _v2("classify"))
    assert resolve_stage_prompt(user_root, other, "classify", notes=[]).source == str(
        user_root / "classify" / "custom.md"
    )

    # v1 file without prompt_file: stage names, packaged v2.
    no_file = DirectSkillGenerationConfig(legacy_format=True)
    notes: list[str] = []
    assert resolve_stage_prompt(user_root, no_file, "review", notes=notes).source == str(
        packaged / "review" / "prompt_v2.md"
    )
    assert notes == []


# ---------------------------------------------------------- packaged files


def test_packaged_prompts_root_points_into_resources() -> None:
    root = packaged_prompts_root()

    assert root == REAL_PACKAGED
    assert (root / "classify" / "prompt_v1.md").is_file()
    assert (root / "generate" / "prompt_v1.md").is_file()


@pytest.mark.parametrize("stage", STAGES)
def test_packaged_v2_prompts_declare_contract_and_placeholders(stage: str) -> None:
    path = REAL_PACKAGED / stage / "prompt_v2.md"
    assert path.is_file(), f"packaged {path} missing"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    assert lines[0].startswith("## description:")
    assert lines[1] == "## contract_version: 2"
    assert declared_contract(text) == 2
    for placeholder in REQUIRED_PLACEHOLDERS[stage]:
        assert placeholder in text

    resolved = resolve_stage_prompt(REAL_PACKAGED.parent / "nowhere", DirectSkillGenerationConfig(), stage, notes=[])
    assert resolved.source == str(path)
    assert resolved.sha256 == prompt_sha256(text)
