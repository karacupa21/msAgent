#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Configuration of the background miner (config.skill.daemon.yml).

A component file of its own: ``config.skill.evolver.yml`` is schema_version 2 with
``extra="forbid"``, so a new section there would force a schema bump and an in-memory
migration on every existing home. Unlike the experience-graph loader this one **fails
loudly** — a broken file stops the tick instead of degrading to defaults.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from msagent.core.logging import get_logger

logger = get_logger(__name__)

CONFIG_FILE_NAME = "config.skill.daemon.yml"
ENV_CONFIG_PATH = "MSAGENT_SKILL_DAEMON_CONFIG"
ENV_DISABLED = "MSAGENT_SKILL_DAEMON_DISABLED"
ENV_ENABLED = "MSAGENT_SKILL_DAEMON_ENABLED"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# The daemon state lives beside the projects, not inside any single one of them.
STATE_FOLDER_NAME = "skill-daemon"

REMINE_REASONS = ("content_change", "methodology_change")
_FILE_KEY = "<file>"


class SkillDaemonConfigError(ValueError):
    """Config problems; every entry names the exact dotted key."""

    def __init__(self, problems: Sequence[tuple[str, str]], *, file: Path | None) -> None:
        self.problems = sorted(problems)
        self.file = file
        super().__init__("; ".join(self.lines()))

    def lines(self) -> list[str]:
        where = self.file.name if self.file is not None else "config"
        return [f"{where}: {path}: {message}" for path, message in self.problems]


class ScheduleSection(BaseModel):
    """When a tick may run and how much it may do."""

    model_config = ConfigDict(extra="forbid")

    quiet_period_seconds: int = Field(
        default=900,
        ge=0,
        description="A trajectory must be idle this long before it counts as finished",
    )
    min_interval_seconds: int = Field(
        default=1800,
        ge=0,
        description="Refuse to start a tick sooner than this after the previous one",
    )
    max_threads_per_tick: int = Field(default=5, ge=1)
    max_llm_calls_per_tick: int = Field(default=60, ge=1)


class ScopeSection(BaseModel):
    """Which projects and agents the daemon may touch."""

    model_config = ConfigDict(extra="forbid")

    projects: Literal["all", "listed"] = "all"
    project_dirs: list[str] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)


class MiningSection(BaseModel):
    """The policy a background run applies; mirrors the /skill-mine flags."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(
        default=None,
        description="null -> the model the CLI would resolve; else an alias from config.llms.yml",
    )
    policy: Literal["strict_knowledge", "reusable_workflow"] = "strict_knowledge"
    demo: bool = False
    remine_on: list[Literal["content_change", "methodology_change"]] = Field(
        default_factory=lambda: list(REMINE_REASONS),
    )
    max_attempts: int = Field(default=3, ge=1)


class NotifySection(BaseModel):
    """How a finished tick reaches the user."""

    model_config = ConfigDict(extra="forbid")

    inbox: bool = True


class SkillDaemonConfig(BaseModel):
    """config.skill.daemon.yml, schema_version 1.

    ``schema_version`` rather than ``version``, matching the sibling
    ``config.skill.evolver.yml``: these component files version their own schema and
    are not tied to the application version the config framework stamps elsewhere.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    enabled: bool = Field(
        default=False,
        description="Off by default: running the pipeline unattended spends LLM calls",
    )
    schedule: ScheduleSection = Field(default_factory=ScheduleSection)
    scope: ScopeSection = Field(default_factory=ScopeSection)
    mining: MiningSection = Field(default_factory=MiningSection)
    notify: NotifySection = Field(default_factory=NotifySection)

    @property
    def is_active(self) -> bool:
        """The YAML switch and the environment agree that the daemon may run."""
        return self.enabled and is_daemon_enabled()

    def remines_on(self, reason: str) -> bool:
        """Whether ``reason`` re-mines a thread the ledger already knows."""
        return reason in self.mining.remine_on


_cache_lock = threading.Lock()
_cached_config: SkillDaemonConfig | None = None
_cached_source: Path | None = None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def is_daemon_enabled(*, force_reload: bool = False) -> bool:
    """Live switch. DISABLED wins; ENABLED opts in; else the YAML value (default off)."""
    if _env_flag(ENV_DISABLED):
        return False
    if _env_flag(ENV_ENABLED):
        return True
    return load_daemon_config(force_reload=force_reload).enabled


def _candidate_paths() -> list[Path]:
    """Env override, then the user file, then the packaged default."""
    candidates: list[Path] = []
    env_path = os.environ.get(ENV_CONFIG_PATH, "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    from msagent.core.paths import AppPaths

    candidates.append(AppPaths.resolve().config_dir / CONFIG_FILE_NAME)
    candidates.append(Path(str(files("resources") / "configs" / "default")) / CONFIG_FILE_NAME)
    return candidates


def _translate(exc: ValidationError) -> list[tuple[str, str]]:
    """Pydantic errors as (dotted key, message) pairs."""
    return [(".".join(str(part) for part in error["loc"]) or "<root>", error["msg"]) for error in exc.errors()]


def _load_from_disk() -> tuple[SkillDaemonConfig, Path | None]:
    """The first existing candidate, parsed strictly. A broken file raises."""
    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise SkillDaemonConfigError([(_FILE_KEY, f"not valid YAML: {exc}")], file=path) from exc
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise SkillDaemonConfigError([(_FILE_KEY, "must contain a mapping")], file=path)
        try:
            return SkillDaemonConfig.model_validate(payload), path
        except ValidationError as exc:
            raise SkillDaemonConfigError(_translate(exc), file=path) from exc
    return SkillDaemonConfig(), None


def load_daemon_config(*, force_reload: bool = False) -> SkillDaemonConfig:
    """Load the daemon configuration (cached per process).

    Env flags are re-read on every call, so flipping
    ``MSAGENT_SKILL_DAEMON_DISABLED`` / ``_ENABLED`` needs no restart.
    """
    global _cached_config, _cached_source
    with _cache_lock:
        if _cached_config is None or force_reload:
            _cached_config, _cached_source = _load_from_disk()
        config = _cached_config
    if _env_flag(ENV_DISABLED):
        return config.model_copy(update={"enabled": False})
    if _env_flag(ENV_ENABLED):
        return config.model_copy(update={"enabled": True})
    return config


def config_source() -> Path | None:
    """The file the cached config came from; ``None`` when defaults are in force."""
    with _cache_lock:
        return _cached_source


def reset_config_cache() -> None:
    """Drop the cached config (tests, and the --watch loop between ticks)."""
    global _cached_config, _cached_source
    with _cache_lock:
        _cached_config = None
        _cached_source = None


def daemon_state_dir() -> Path:
    """``<MSAGENT_HOME>/state/skill-daemon`` — the ledger and the tick lock."""
    from msagent.core.paths import AppPaths

    return AppPaths.resolve().state_dir / STATE_FOLDER_NAME
