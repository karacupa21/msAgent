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

"""Skill Evolver configuration: ``config.skill.evolver.yml`` (schema_version 2).

Rules of SCHEMA_VERSION 2:

* One file, read once per command run: the user copy under the msAgent
  config dir, else the packaged default, else the dataclass defaults. The
  file is never written by the pipeline.
* A file with ``schema_version`` is a v2 document: nested sections, every
  key known (``extra="forbid"``), every value of the declared type (strict:
  ``"false"`` is not a boolean, ``True`` is not a whole number; the YAML
  integer ``1`` is accepted for a score). Legacy flat keys in a v2 file are
  errors that name the v2 key to use.
* A file without ``schema_version`` is a flat v1 document and is normalised
  in memory: ``min_evidence_score`` -> ``gate``, ``max_plans``/``category``/
  ``output_dir`` -> ``generation``; ``active``/``prompt_file`` are kept for
  the legacy prompt rule (:mod:`msagent.skill_evolver.prompts`) and noted.
  Unknown keys and invalid values are errors whose path is the v1 key.
* Every problem is reported at once as :class:`SkillEvolverConfigError`
  with ``(dotted key, message)`` pairs; nothing falls back silently.
* Precedence: model defaults -> file -> CLI overrides
  (:func:`apply_overrides`) -> :func:`effective_rules`. ``demo_mode`` only
  switches the selection to ``demo_workflow``; it changes no limit.

Imports only the stdlib, ``yaml``, ``pydantic`` and ``msagent.core`` path
constants, so :mod:`msagent.exgraph.skills` may import it lazily.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError, field_validator

from msagent.core.constants import CONFIG_SKILL_EVOLVER_FILE_NAME
from msagent.core.paths import AppPaths

SCHEMA_VERSION = 2
# Prompt contract every stage prompt of this build must declare (prompts.py).
CONTRACT_VERSION = 2
POLICIES = ("strict_knowledge", "reusable_workflow")
# ``classification.policy`` plus the overlay ``demo_mode`` selects.
SELECTIONS = ("strict_knowledge", "reusable_workflow", "demo_workflow")
ON_NOTHING_ACTIONS = ("stop", "expand_context_once")
# Prompt stages of the evidence pipeline: folders under skill-evolver/prompts/.
STAGES = ("classify", "generate", "review")
DEFAULT_PROMPT_FILE = "prompt_v2.md"
# The contract-1 prompt file still shipped for the legacy prompt_file rule.
LEGACY_PACKAGED_PROMPT_FILE = "prompt_v1.md"
DEFAULT_CATEGORY = "default"
DEFAULT_VARIANT = "default"
# Same value as features.DEFAULT_MIN_EVIDENCE_SCORE (this module must not import features).
DEFAULT_MIN_EVIDENCE_SCORE = 1.0
DEFAULT_MAX_PLANS = 3
DEFAULT_MAX_LLM_CALLS = 16
# Newest trajectories of the agent mined for procedures shared across sessions.
CROSS_SESSION_LIMIT = 20
# Keys of a flat v1 file and where the v2 schema keeps them.
V1_KEYS = ("active", "prompt_file", "category", "output_dir", "min_evidence_score", "max_plans")
V1_TO_V2 = {
    "min_evidence_score": "gate.min_evidence_score",
    "max_plans": "generation.max_plans_per_thread",
    "category": "generation.category",
    "output_dir": "generation.output_dir",
}
_V2_TO_V1 = {v2: v1 for v1, v2 in V1_TO_V2.items()}
# Safe folder and file names inside the prompts root and the skill library (no path separators).
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_LITERAL_CHOICES = {
    ("schema_version",): (SCHEMA_VERSION,),
    ("classification", "policy"): POLICIES,
    ("on_nothing", "action"): ON_NOTHING_ACTIONS,
}
_FILE_KEY = "<file>"


class SkillEvolverConfigError(ValueError):
    """Config problems; every entry names the exact dotted key."""

    def __init__(self, problems: Sequence[tuple[str, str]], *, file: Path | None) -> None:
        self.problems = sorted(problems)
        self.file = file
        super().__init__("; ".join(self.lines()))

    def lines(self) -> list[str]:
        where = self.file.name if self.file is not None else "config"
        return [f"{where}: {path}: {message}" for path, message in self.problems]


# ------------------------------------------------------------------ validators


def _whole_number(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"must be a whole number, got {value!r}")
    return value


def _positive_int(value: object) -> int:
    number = _whole_number(value)
    if number <= 0:
        raise ValueError(f"must be a positive whole number, got {number}")
    return number


def _non_negative_int(value: object) -> int:
    number = _whole_number(value)
    if number < 0:
        raise ValueError(f"must be a whole number >= 0, got {number}")
    return number


def _score(value: object) -> float:
    message = f"must be a finite non-negative number, got {value!r}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(message)
    return number


def _directory_name(value: str) -> str:
    if not _SAFE_SEGMENT.fullmatch(value):
        raise ValueError(f"must be one directory name without separators, got {value!r}")
    return value


def _file_name(value: str) -> str:
    if not _SAFE_SEGMENT.fullmatch(value):
        raise ValueError(f"must be a file name without path separators or '..', got {value!r}")
    return value


def _contract(value: int) -> int:
    if value != CONTRACT_VERSION:
        raise ValueError(f"must equal {CONTRACT_VERSION} (prompt contract of this build)")
    return value


def _output_dir(value: str | None) -> str | None:
    if value is not None and not value.strip():
        raise ValueError(f"must be a non-empty path, got {value!r}")
    return value


# ---------------------------------------------------------------------- schema


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ClassificationSection(_Section):
    policy: Literal["strict_knowledge", "reusable_workflow"] = "strict_knowledge"


class GateSection(_Section):
    min_evidence_score: float = DEFAULT_MIN_EVIDENCE_SCORE

    _score = field_validator("min_evidence_score", mode="before")(_score)


class EvidenceSection(_Section):
    excerpt_max_chars: StrictInt = 1000
    bundle_max_chars: StrictInt = 60000
    surrounding_events: StrictInt = 2
    cross_session_limit: StrictInt = CROSS_SESSION_LIMIT

    _positive = field_validator("excerpt_max_chars", "bundle_max_chars", "cross_session_limit", mode="before")(
        _positive_int
    )
    _non_negative = field_validator("surrounding_events", mode="before")(_non_negative_int)


class OnNothingSection(_Section):
    action: Literal["stop", "expand_context_once"] = "expand_context_once"


class GenerationSection(_Section):
    max_plans_per_thread: StrictInt = DEFAULT_MAX_PLANS
    max_llm_calls_per_thread: StrictInt = DEFAULT_MAX_LLM_CALLS
    category: StrictStr = DEFAULT_CATEGORY
    # Stored as written; expanded and resolved against working_dir by resolve_output_dir().
    output_dir: StrictStr | None = None

    _positive = field_validator("max_plans_per_thread", "max_llm_calls_per_thread", mode="before")(_positive_int)
    _category = field_validator("category")(_directory_name)
    _output_dir = field_validator("output_dir")(_output_dir)


class PromptsSection(_Section):
    contract_version: StrictInt = CONTRACT_VERSION
    classify: StrictStr = DEFAULT_PROMPT_FILE
    generate: StrictStr = DEFAULT_PROMPT_FILE
    review: StrictStr = DEFAULT_PROMPT_FILE

    _whole = field_validator("contract_version", mode="before")(_whole_number)
    _contract = field_validator("contract_version")(_contract)
    _names = field_validator("classify", "generate", "review")(_file_name)


class DiagnosticsSection(_Section):
    save_decision_report: StrictBool = True
    save_evidence_text: StrictBool = False
    save_rejected_drafts: StrictBool = True


class SkillEvolverConfig(_Section):
    """The v2 document; ``model_validate`` accepts exactly what the packaged YAML documents."""

    schema_version: Literal[2]
    demo_mode: StrictBool = False
    classification: ClassificationSection = Field(default_factory=ClassificationSection)
    gate: GateSection = Field(default_factory=GateSection)
    evidence: EvidenceSection = Field(default_factory=EvidenceSection)
    on_nothing: OnNothingSection = Field(default_factory=OnNothingSection)
    generation: GenerationSection = Field(default_factory=GenerationSection)
    prompts: PromptsSection = Field(default_factory=PromptsSection)
    diagnostics: DiagnosticsSection = Field(default_factory=DiagnosticsSection)


def _message(error: dict[str, Any]) -> str:
    kind = error["type"]
    value = error.get("input")
    if kind == "extra_forbidden":
        return "unknown key"
    if kind == "missing":
        return "required"
    if kind == "literal_error":
        choices = _LITERAL_CHOICES.get(tuple(error["loc"]))
        expected = ", ".join(str(c) for c in choices) if choices else error["ctx"]["expected"]
        return f"must be one of {expected}, got {value!r}"
    if kind in ("bool_type", "bool_parsing"):
        return f"must be a boolean (true/false), got {value!r}"
    if kind in ("int_type", "int_parsing", "int_from_float"):
        return f"must be a whole number, got {value!r}"
    if kind in ("float_type", "float_parsing"):
        return f"must be a finite non-negative number, got {value!r}"
    if kind == "string_type":
        return f"must be text, got {value!r}"
    if kind in ("model_type", "dict_type", "model_attributes_type"):
        return "must be a mapping"
    if kind == "value_error":
        return error["msg"].removeprefix("Value error, ")
    return error["msg"]


def _translate(exc: ValidationError, *, v1_source: bool) -> list[tuple[str, str]]:
    problems: list[tuple[str, str]] = []
    for error in exc.errors():
        path = ".".join(str(part) for part in error["loc"])
        if v1_source:
            path = _V2_TO_V1.get(path, path)
        problems.append((path, _message(error)))
    return problems


# ------------------------------------------------------------ requested config


@dataclass(frozen=True, slots=True)
class DirectSkillGenerationConfig:
    """Settings as requested by the file and the CLI (flat view of the v2 document)."""

    schema_version: int = SCHEMA_VERSION
    demo_mode: bool = False
    policy: str = "strict_knowledge"
    min_evidence_score: float = DEFAULT_MIN_EVIDENCE_SCORE
    excerpt_max_chars: int = 1000
    bundle_max_chars: int = 60000
    surrounding_events: int = 2
    cross_session_limit: int = CROSS_SESSION_LIMIT
    on_nothing: str = "expand_context_once"
    max_plans: int = DEFAULT_MAX_PLANS
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    category: str = DEFAULT_CATEGORY
    # As written; relative paths are resolved by resolve_output_dir()/output_root().
    output_dir: Path | None = None
    contract_version: int = CONTRACT_VERSION
    classify_prompt: str = DEFAULT_PROMPT_FILE
    generate_prompt: str = DEFAULT_PROMPT_FILE
    review_prompt: str = DEFAULT_PROMPT_FILE
    save_decision_report: bool = True
    save_evidence_text: bool = False
    save_rejected_drafts: bool = True
    # Legacy replay variant folder (prompts/<active>/); never used for policy.
    active: str = DEFAULT_VARIANT
    # Legacy v1 key; drives the legacy prompt rule in prompts.py.
    prompt_file: str | None = None
    # True when read from a flat v1 file.
    legacy_format: bool = False
    # "defaults" | "packaged" | str(user file)
    source: str = "defaults"
    # Diagnostics, e.g. "active: legacy replay field, ignored".
    notes: tuple[str, ...] = ()
    # CLI overrides applied, e.g. ("--policy reusable_workflow", "--demo").
    overrides: tuple[str, ...] = ()

    def prompt_for(self, stage: str) -> str:
        """Prompt file name configured for ``stage``."""
        if stage not in STAGES:
            raise ValueError(f"Unknown prompt stage '{stage}'; expected {STAGES}")
        if self.legacy_format and self.prompt_file:
            return self.prompt_file
        return str(getattr(self, f"{stage}_prompt"))

    def as_record(self) -> dict[str, Any]:
        """The v2 nested shape plus provenance of the values (JSON-able)."""
        return {
            "schema_version": self.schema_version,
            "demo_mode": self.demo_mode,
            "classification": {"policy": self.policy},
            "gate": {"min_evidence_score": self.min_evidence_score},
            "evidence": {
                "excerpt_max_chars": self.excerpt_max_chars,
                "bundle_max_chars": self.bundle_max_chars,
                "surrounding_events": self.surrounding_events,
                "cross_session_limit": self.cross_session_limit,
            },
            "on_nothing": {"action": self.on_nothing},
            "generation": {
                "max_plans_per_thread": self.max_plans,
                "max_llm_calls_per_thread": self.max_llm_calls,
                "category": self.category,
                "output_dir": None if self.output_dir is None else str(self.output_dir),
            },
            "prompts": {"contract_version": self.contract_version, **{s: self.prompt_for(s) for s in STAGES}},
            "diagnostics": {
                "save_decision_report": self.save_decision_report,
                "save_evidence_text": self.save_evidence_text,
                "save_rejected_drafts": self.save_rejected_drafts,
            },
            "source": self.source,
            "legacy_format": self.legacy_format,
            "overrides": list(self.overrides),
            "notes": list(self.notes),
        }


def _from_model(
    model: SkillEvolverConfig,
    *,
    source: str,
    legacy_format: bool = False,
    active: str = DEFAULT_VARIANT,
    prompt_file: str | None = None,
    notes: tuple[str, ...] = (),
) -> DirectSkillGenerationConfig:
    generation = model.generation
    return DirectSkillGenerationConfig(
        schema_version=model.schema_version,
        demo_mode=model.demo_mode,
        policy=model.classification.policy,
        min_evidence_score=model.gate.min_evidence_score,
        excerpt_max_chars=model.evidence.excerpt_max_chars,
        bundle_max_chars=model.evidence.bundle_max_chars,
        surrounding_events=model.evidence.surrounding_events,
        cross_session_limit=model.evidence.cross_session_limit,
        on_nothing=model.on_nothing.action,
        max_plans=generation.max_plans_per_thread,
        max_llm_calls=generation.max_llm_calls_per_thread,
        category=generation.category,
        output_dir=None if generation.output_dir is None else Path(generation.output_dir),
        contract_version=model.prompts.contract_version,
        classify_prompt=model.prompts.classify,
        generate_prompt=model.prompts.generate,
        review_prompt=model.prompts.review,
        save_decision_report=model.diagnostics.save_decision_report,
        save_evidence_text=model.diagnostics.save_evidence_text,
        save_rejected_drafts=model.diagnostics.save_rejected_drafts,
        active=active,
        prompt_file=prompt_file,
        legacy_format=legacy_format,
        source=source,
        notes=notes,
    )


# ---------------------------------------------------------------------- loader


def packaged_config_file() -> Path:
    """The packaged ``config.skill.evolver.yml`` (seeded into the user config dir on startup)."""
    return Path(str(files("resources") / "configs" / "default")) / CONFIG_SKILL_EVOLVER_FILE_NAME.name


def load_skill_evolver_config(config_dir: Path | None = None) -> DirectSkillGenerationConfig:
    """Read config.skill.evolver.yml: user file, then packaged default, then dataclass defaults.

    Raises :class:`SkillEvolverConfigError` for every problem of the file that
    was found; nothing is written.
    """
    base = config_dir if config_dir is not None else AppPaths.resolve().config_dir
    user = base / CONFIG_SKILL_EVOLVER_FILE_NAME.name
    if user.is_file():
        return _parse_file(user, source=str(user))
    packaged = packaged_config_file()
    if packaged.is_file():
        return _parse_file(packaged, source="packaged")
    return DirectSkillGenerationConfig()


def _parse_file(path: Path, *, source: str) -> DirectSkillGenerationConfig:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SkillEvolverConfigError([(_FILE_KEY, f"invalid YAML: {exc}")], file=path) from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SkillEvolverConfigError([(_FILE_KEY, "must be a mapping")], file=path)
    if "schema_version" in data:
        return _parse_v2(data, path, source=source)
    return _parse_v1(data, path, source=source)


def _parse_v2(data: dict[str, Any], path: Path, *, source: str) -> DirectSkillGenerationConfig:
    problems: list[tuple[str, str]] = []
    for key in V1_KEYS:
        if key not in data:
            continue
        if key in V1_TO_V2:
            hint = f"use {V1_TO_V2[key]}"
        else:
            hint = "remove it (prompts are configured under prompts:)"
        problems.append((key, f"legacy v1 key in a schema_version {SCHEMA_VERSION} file; {hint}"))
    version = data["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        problems.append(("schema_version", f"must be a whole number, got {version!r}"))
    elif version != SCHEMA_VERSION:
        message = f"unsupported schema_version {version}; this build supports {SCHEMA_VERSION}"
        problems.append(("schema_version", message))
    if problems:
        raise SkillEvolverConfigError(problems, file=path)
    try:
        model = SkillEvolverConfig.model_validate(data)
    except ValidationError as exc:
        raise SkillEvolverConfigError(_translate(exc, v1_source=False), file=path) from exc
    return _from_model(model, source=source)


def _parse_v1(data: dict[str, Any], path: Path, *, source: str) -> DirectSkillGenerationConfig:
    problems: list[tuple[str, str]] = [(str(key), "unknown key") for key in data if key not in V1_KEYS]
    notes: list[str] = []
    active = DEFAULT_VARIANT
    prompt_file: str | None = None
    if data.get("active") is not None:
        try:
            active = _directory_name(_text(data["active"]))
        except ValueError as exc:
            problems.append(("active", str(exc)))
        notes.append("active: legacy replay field, ignored")
    if data.get("prompt_file") is not None:
        try:
            prompt_file = _file_name(_text(data["prompt_file"]))
        except ValueError as exc:
            problems.append(("prompt_file", str(exc)))
        notes.append("prompt_file: legacy setting; prompt files are resolved by contract_version (see prompts)")

    normalised: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "gate": {}, "generation": {}}
    for v1_key, v2_key in V1_TO_V2.items():
        if v1_key in data:
            section, field_name = v2_key.split(".")
            normalised[section][field_name] = data[v1_key]
    try:
        model = SkillEvolverConfig.model_validate(normalised)
    except ValidationError as exc:
        problems.extend(_translate(exc, v1_source=True))
        raise SkillEvolverConfigError(problems, file=path) from exc
    if problems:
        raise SkillEvolverConfigError(problems, file=path)
    return _from_model(
        model,
        source=source,
        legacy_format=True,
        active=active,
        prompt_file=prompt_file,
        notes=tuple(notes),
    )


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"must be text, got {value!r}")
    return value


# ------------------------------------------------------------------- overrides


def apply_overrides(
    cfg: DirectSkillGenerationConfig,
    *,
    policy: str | None = None,
    demo: bool | None = None,
) -> DirectSkillGenerationConfig:
    """CLI flags on top of the file for this run only; the file is not touched."""
    if policy is not None and policy not in POLICIES:
        raise ValueError(f"--policy: expected one of {', '.join(POLICIES)}, got {policy!r}")
    changes: dict[str, Any] = {}
    overrides = list(cfg.overrides)
    if policy is not None:
        changes["policy"] = policy
        overrides.append(f"--policy {policy}")
    if demo is not None:
        changes["demo_mode"] = demo
        overrides.append("--demo" if demo else "--no-demo")
    return replace(cfg, overrides=tuple(overrides), **changes)


def resolve_output_dir(cfg: DirectSkillGenerationConfig, working_dir: Path) -> Path | None:
    """``generation.output_dir`` with ``~`` expanded and relative paths under ``working_dir``; None when unset."""
    if cfg.output_dir is None:
        return None
    path = cfg.output_dir.expanduser()
    return path if path.is_absolute() else working_dir / path


def output_root(cfg: DirectSkillGenerationConfig, working_dir: Path) -> Path:
    """Root that receives ``.proposals/``: the configured output_dir or ``<working_dir>/skills``."""
    resolved = resolve_output_dir(cfg, working_dir)
    return resolved if resolved is not None else working_dir / "skills"


# ------------------------------------------------------------- effective rules


@dataclass(frozen=True, slots=True)
class EffectiveRules:
    """What one run applies after file and CLI: the selection and the copied limits."""

    policy: str
    demo: bool
    # "demo_workflow" when demo, else the policy.
    selection: str
    min_evidence_score: float
    max_plans: int
    max_llm_calls: int
    on_nothing: str
    excerpt_max_chars: int
    bundle_max_chars: int
    surrounding_events: int
    cross_session_limit: int
    category: str
    output_root: Path
    save_decision_report: bool
    save_evidence_text: bool
    save_rejected_drafts: bool

    def as_record(self) -> dict[str, Any]:
        record = {f.name: getattr(self, f.name) for f in fields(self)}
        record["output_root"] = str(self.output_root)
        return record


def effective_rules(cfg: DirectSkillGenerationConfig, *, working_dir: Path) -> EffectiveRules:
    """Rules of a run; demo changes the selection only, never a limit or budget."""
    return EffectiveRules(
        policy=cfg.policy,
        demo=cfg.demo_mode,
        selection="demo_workflow" if cfg.demo_mode else cfg.policy,
        min_evidence_score=cfg.min_evidence_score,
        max_plans=cfg.max_plans,
        max_llm_calls=cfg.max_llm_calls,
        on_nothing=cfg.on_nothing,
        excerpt_max_chars=cfg.excerpt_max_chars,
        bundle_max_chars=cfg.bundle_max_chars,
        surrounding_events=cfg.surrounding_events,
        cross_session_limit=cfg.cross_session_limit,
        category=cfg.category,
        output_root=output_root(cfg, working_dir),
        save_decision_report=cfg.save_decision_report,
        save_evidence_text=cfg.save_evidence_text,
        save_rejected_drafts=cfg.save_rejected_drafts,
    )
