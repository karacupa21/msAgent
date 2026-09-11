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

"""Stage prompts: contract check, resolution and hashes.

Rules of prompt contract 2:

* A stage prompt lives at ``skill-evolver/prompts/<stage>/<name>``; the user
  copy wins over the packaged one, ``name`` is ``prompts.<stage>`` of the
  config. There is no glob: a missing file is an error, never a concatenation.
* The file declares its contract in one of its first five lines
  (``## contract_version: 2``; the description line comes first). A file
  without the header is contract 1. A declared contract other than
  ``prompts.contract_version`` stops the command with the migration text.
* Every placeholder of :data:`REQUIRED_PLACEHOLDERS` must be present; the
  pipeline substitutes them with ``str.replace``/one-pass regex, never
  ``str.format``, so prompt text may contain braces elsewhere.
* Legacy rule (v1 config with ``prompt_file: prompt_v1.md``): a user copy
  that is absent or byte-identical to the packaged ``prompt_v1.md`` means the
  packaged ``prompt_v2.md`` is used and noted; an edited copy is a
  :class:`PromptContractError` with the migration instruction. Files are
  never modified.
* ``sha256`` is taken over the template text before any substitution and
  recorded as ``sha256:<hex>`` in provenance and decision reports.

Stdlib only (plus :mod:`msagent.skill_evolver.config` for the names).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from msagent.core.constants import SKILL_EVOLVER_CONFIG_FOLDER_NAME
from msagent.skill_evolver.config import (
    DEFAULT_PROMPT_FILE,
    LEGACY_PACKAGED_PROMPT_FILE,
    DirectSkillGenerationConfig,
)

CONTRACT_HEADER_RE = re.compile(r"^## contract_version:\s*(\d+)\s*$")
# Lines of a prompt file searched for the contract header.
HEADER_LINES = 5
# Contract of a prompt file without a header.
LEGACY_CONTRACT = 1
REQUIRED_PLACEHOLDERS: dict[str, tuple[str, ...]] = {
    "classify": ("{skill_library}", "{evidence_bundle}", "{selection_policy}"),
    "generate": ("{candidates}", "{existing_skill}", "{generation_policy}"),
    "review": ("{skill_md}", "{candidates}", "{evidence}", "{existing_skill}", "{review_policy}"),
}


class PromptContractError(ValueError):
    """A prompt file does not satisfy the contract of this build; the message tells how to migrate."""


@dataclass(frozen=True, slots=True)
class PromptText:
    stage: str
    text: str
    source: str
    # "sha256:<hex>" of ``text``.
    sha256: str
    contract_version: int


@dataclass(frozen=True, slots=True)
class StagePrompts:
    classify: PromptText
    generate: PromptText
    review: PromptText

    def get(self, stage: str) -> PromptText:
        prompt = getattr(self, stage, None)
        if not isinstance(prompt, PromptText):
            raise ValueError(f"Unknown prompt stage '{stage}'; expected {tuple(REQUIRED_PLACEHOLDERS)}")
        return prompt

    def variants(self) -> dict[str, str]:
        return {stage: self.get(stage).source for stage in REQUIRED_PLACEHOLDERS}

    def hashes(self) -> dict[str, str]:
        return {stage: self.get(stage).sha256 for stage in REQUIRED_PLACEHOLDERS}


def prompt_sha256(text: str) -> str:
    """Hash of the template text before substitution."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def declared_contract(text: str) -> int:
    """Contract declared in the header; LEGACY_CONTRACT when no header line matches."""
    for line in text.splitlines()[:HEADER_LINES]:
        match = CONTRACT_HEADER_RE.match(line)
        if match:
            return int(match.group(1))
    return LEGACY_CONTRACT


def packaged_prompts_root() -> Path:
    """``resources/configs/default/skill-evolver/prompts`` of this build."""
    return Path(str(files("resources") / "configs" / "default")) / SKILL_EVOLVER_CONFIG_FOLDER_NAME / "prompts"


def _migration(user_root: Path, packaged_root: Path, stage: str) -> str:
    return (
        f"Migrate: copy {packaged_root / stage / DEFAULT_PROMPT_FILE} to {user_root / stage}/ and port your edits, "
        f"then set prompts.{stage}: {DEFAULT_PROMPT_FILE} in config.skill.evolver.yml (schema_version 2). "
        "The file was not modified."
    )


def _same_bytes(user: Path, packaged: Path) -> bool:
    return packaged.is_file() and user.read_bytes() == packaged.read_bytes()


def resolve_stage_prompt(
    user_root: Path,
    cfg: DirectSkillGenerationConfig,
    stage: str,
    *,
    notes: list[str],
) -> PromptText:
    """Locate, read and check the prompt of ``stage``; legacy notes are appended to ``notes``."""
    name = cfg.prompt_for(stage)
    packaged_root = packaged_prompts_root()
    user = user_root / stage / name
    packaged = packaged_root / stage / name
    if cfg.legacy_format and cfg.prompt_file == LEGACY_PACKAGED_PROMPT_FILE:
        if user.is_file() and not _same_bytes(user, packaged):
            raise PromptContractError(
                f"{user} is a customised contract-{LEGACY_CONTRACT} prompt; this build needs contract "
                f"{cfg.contract_version}. {_migration(user_root, packaged_root, stage)}"
            )
        notes.append(
            f"prompt_file '{LEGACY_PACKAGED_PROMPT_FILE}' is a legacy setting; using packaged "
            f"prompts/{stage}/{DEFAULT_PROMPT_FILE} (contract {cfg.contract_version})"
        )
        user = packaged = packaged_root / stage / DEFAULT_PROMPT_FILE

    path = user if user.is_file() else packaged
    if not path.is_file():
        raise ValueError(
            f"prompt file not found: {user} (nor packaged {packaged}); "
            f"check prompts.{stage} in config.skill.evolver.yml"
        )
    text = path.read_text(encoding="utf-8")
    declared = declared_contract(text)
    if declared != cfg.contract_version:
        raise PromptContractError(
            f"{path} declares contract_version {declared} but prompts.contract_version is {cfg.contract_version}; "
            f"{_migration(user_root, packaged_root, stage)}"
        )
    for placeholder in REQUIRED_PLACEHOLDERS[stage]:
        if placeholder not in text:
            raise PromptContractError(
                f"{path} lacks the {placeholder} placeholder required by contract {cfg.contract_version}"
            )
    return PromptText(stage, text, str(path), prompt_sha256(text), declared)
