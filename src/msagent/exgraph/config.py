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


class ExgraphConfig(BaseModel):
    version: str = Field(default="1.0")
    enabled: bool = Field(default=True)
    outcome: OutcomeConfig = Field(default_factory=OutcomeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    skills: SkillScanConfig = Field(default_factory=SkillScanConfig)

    @property
    def is_active(self) -> bool:
        return self.enabled


_cache_lock = threading.Lock()
_cached_config: ExgraphConfig | None = None


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
    """Load the experience-graph configuration (cached per process)."""
    global _cached_config
    with _cache_lock:
        if _cached_config is None or force_reload:
            config = _load_from_disk()
            if os.environ.get(ENV_DISABLED, "").strip().lower() in _TRUTHY:
                config = config.model_copy(update={"enabled": False})
            _cached_config = config
        return _cached_config


def reset_config_cache() -> None:
    """Drop the cached configuration (used by tests)."""
    global _cached_config
    with _cache_lock:
        _cached_config = None
