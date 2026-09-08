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

"""Configuration for the experience graph (config.exgraph.yml)."""

from __future__ import annotations

import logging
import os
import threading
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "config.exgraph.yml"
ENV_CONFIG_PATH = "MSAGENT_EXGRAPH_CONFIG"
ENV_DISABLED = "MSAGENT_EXGRAPH_DISABLED"
ENV_ENABLED = "MSAGENT_EXGRAPH_ENABLED"
_TRUTHY = {"1", "true", "yes", "on"}


class OutcomeConfig(BaseModel):
    policy: str = Field(default="v1", description="Outcome labeling policy name")


class OutputConfig(BaseModel):
    directory: str = Field(
        default="exgraph",
        description="Graph files directory; relative paths resolve against the project state dir",
    )


class SkillScanConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Attach SkillDoc nodes when a proposal or SKILL.md cites this thread",
    )


class SimilarConfig(BaseModel):
    min_tools: float = Field(default=0.5, description="Minimum tool-path Jaccard")
    min_tokens: float = Field(default=0.25, description="Minimum user-text token Jaccard")


class InsightConfig(BaseModel):
    min_support: int = Field(default=2, description="Minimum recipe support for an Insight")


class ExgraphConfig(BaseModel):
    version: str = Field(default="1.0")
    enabled: bool = Field(
        default=False,
        description="Off by default so a merge into skill-evolver does not change mining",
    )
    outcome: OutcomeConfig = Field(default_factory=OutcomeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    skills: SkillScanConfig = Field(default_factory=SkillScanConfig)
    similar: SimilarConfig = Field(default_factory=SimilarConfig)
    insight: InsightConfig = Field(default_factory=InsightConfig)

    @property
    def is_active(self) -> bool:
        return self.enabled and is_exgraph_enabled()


_cache_lock = threading.Lock()
_cached_config: ExgraphConfig | None = None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _env_disabled() -> bool:
    return _env_flag(ENV_DISABLED)


def _env_enabled() -> bool:
    return _env_flag(ENV_ENABLED)


def is_exgraph_enabled(*, force_reload: bool = False) -> bool:
    """Live switch. Disabled wins; ENABLED opts in; else YAML (default off)."""
    if _env_disabled():
        return False
    if _env_enabled():
        return True
    return load_exgraph_config(force_reload=force_reload).enabled


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get(ENV_CONFIG_PATH, "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    try:
        from msagent.core.paths import AppPaths

        candidates.append(AppPaths.resolve().config_dir / CONFIG_FILE_NAME)
    except Exception:
        logger.debug("Cannot resolve msAgent config dir for exgraph config", exc_info=True)
    try:
        candidates.append(Path(str(files("resources") / "configs" / "default")) / CONFIG_FILE_NAME)
    except Exception:
        logger.debug("Cannot resolve packaged default exgraph config", exc_info=True)
    return candidates


def _load_from_disk() -> ExgraphConfig:
    for path in _candidate_paths():
        try:
            if not path.is_file():
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(payload, dict):
                logger.warning("Exgraph config %s must contain a mapping; using defaults", path)
                return ExgraphConfig()
            return ExgraphConfig.model_validate(payload)
        except Exception:
            logger.warning("Invalid exgraph config %s; using defaults", path, exc_info=True)
            return ExgraphConfig()
    return ExgraphConfig()


def load_exgraph_config(*, force_reload: bool = False) -> ExgraphConfig:
    """Load the experience-graph configuration (cached per process).

    Env flags are applied on every call so flipping
    ``MSAGENT_EXGRAPH_DISABLED`` / ``MSAGENT_EXGRAPH_ENABLED``
    does not require a process restart. Disabled always wins.
    """
    global _cached_config
    with _cache_lock:
        if _cached_config is None or force_reload:
            _cached_config = _load_from_disk()
        config = _cached_config
        if _env_disabled():
            return config.model_copy(update={"enabled": False})
        if _env_enabled():
            return config.model_copy(update={"enabled": True})
        return config


def reset_config_cache() -> None:
    """Drop the cached configuration (used by tests)."""
    global _cached_config
    with _cache_lock:
        _cached_config = None
