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

"""Configuration for the trajectory recorder (config.trajectory.recorder.yml)."""

from __future__ import annotations

import logging
import os
import threading
from enum import Enum
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "config.trajectory.recorder.yml"
ENV_CONFIG_PATH = "MSAGENT_TRAJECTORY_CONFIG"
ENV_DISABLED = "MSAGENT_TRAJECTORY_DISABLED"

_TRUTHY = {"1", "true", "yes", "on"}


class CaptureLevel(str, Enum):
    OFF = "off"
    MESSAGES = "messages"
    LLM_IO = "llm_io"


class StoreScope(str, Enum):
    """Where trajectory files live: a store per workspace, or one for all of them."""

    WORKSPACE = "workspace"
    SHARED = "shared"


class CaptureConfig(BaseModel):
    level: CaptureLevel = Field(default=CaptureLevel.MESSAGES, description="What to capture")
    tool_starts: bool = Field(default=True, description="Record tool.start events with inputs")
    retries: bool = Field(default=True, description="Record LLM retry events")
    graph_metadata: bool = Field(default=True, description="Attach langgraph node/namespace metadata to events")


class OutputConfig(BaseModel):
    scope: StoreScope = Field(
        default=StoreScope.WORKSPACE,
        description="workspace: a store per project state dir; shared: one for all",
    )
    directory: str = Field(
        default="trajectories",
        description="Store directory; a relative path resolves against the scope root",
    )
    filename: str = Field(
        default="{agent}_{thread_id}.jsonl",
        description="File name template; available fields: {agent}, {thread_id}",
    )


class LimitsConfig(BaseModel):
    max_field_chars: int = Field(default=0, ge=0, description="Max chars per string field, 0 = unlimited")
    max_file_mb: int = Field(default=0, ge=0, description="Max trajectory file size in MB, 0 = unlimited")


class RedactionConfig(BaseModel):
    patterns: list[str] = Field(default_factory=list, description="Regexes replaced in every captured string")
    replacement: str = Field(default="[REDACTED]", description="Replacement text for matched patterns")


class TrajectoryRecorderConfig(BaseModel):
    """Schema of config.trajectory.recorder.yml."""

    version: str = Field(default="1.0", description="Config schema version")
    enabled: bool = Field(default=True, description="Master switch for the recorder")
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)

    @property
    def is_active(self) -> bool:
        return self.enabled and self.capture.level != CaptureLevel.OFF

    @property
    def is_shared(self) -> bool:
        return self.output.scope == StoreScope.SHARED


def store_dir(config: TrajectoryRecorderConfig, *, state_dir: Path | None) -> Path:
    """Directory holding the trajectory files under the configured scope.

    An absolute ``output.directory`` is used as-is in both scopes. A relative one
    resolves against ``<MSAGENT_HOME>/state`` in the shared scope and against the
    project ``state_dir`` in the workspace scope, which then must be given.
    """
    directory = Path(config.output.directory).expanduser()
    if directory.is_absolute():
        return directory
    if config.is_shared:
        from msagent.core.paths import AppPaths

        return AppPaths.resolve().state_dir / directory
    if state_dir is None:
        raise ValueError("the workspace trajectory scope needs a project state dir")
    return Path(state_dir) / directory


_cache_lock = threading.Lock()
_cached_config: TrajectoryRecorderConfig | None = None


def _candidate_paths() -> list[Path]:
    """Config lookup order: env override, user config dir, packaged default."""
    candidates: list[Path] = []

    env_path = os.environ.get(ENV_CONFIG_PATH, "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    try:
        from msagent.core.paths import AppPaths

        candidates.append(AppPaths.resolve().config_dir / CONFIG_FILE_NAME)
    except Exception:
        logger.debug("Cannot resolve msAgent config dir for trajectory config", exc_info=True)

    try:
        candidates.append(Path(str(files("resources") / "configs" / "default")) / CONFIG_FILE_NAME)
    except Exception:
        logger.debug("Cannot resolve packaged default trajectory config", exc_info=True)

    return candidates


def _load_from_disk() -> TrajectoryRecorderConfig:
    for path in _candidate_paths():
        try:
            if not path.is_file():
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(payload, dict):
                logger.warning("Trajectory config %s must contain a mapping; using defaults", path)
                return TrajectoryRecorderConfig()
            return TrajectoryRecorderConfig.model_validate(payload)
        except Exception:
            logger.warning("Invalid trajectory config %s; using defaults", path, exc_info=True)
            return TrajectoryRecorderConfig()
    return TrajectoryRecorderConfig()


def load_trajectory_config(*, force_reload: bool = False) -> TrajectoryRecorderConfig:
    """Load the recorder configuration (cached per process)."""
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