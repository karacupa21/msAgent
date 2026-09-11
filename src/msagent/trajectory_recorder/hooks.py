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

"""Integration surface of the trajectory recorder.

This is the only module the rest of the codebase should import. Every function
is exception-safe: a recording failure is logged and swallowed, the agent run
is never affected. The ``context`` argument is duck-typed (the CLI ``Context``
object) so this package stays decoupled from ``msagent.cli``.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from msagent.trajectory_recorder.callback import TrajectoryCallbackHandler
from msagent.trajectory_recorder.config import (
    TrajectoryRecorderConfig,
    load_trajectory_config,
    store_dir,
)
from msagent.trajectory_recorder.recorder import SCHEMA_VERSION, TrajectoryRecorder
from msagent.trajectory_recorder.serialize import json_safe

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

_DEFAULT_FILENAME = "{agent}_{thread_id}.jsonl"


def _sanitize_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^\w\-.]+", "-", str(value).strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-.")
    return cleaned or fallback


def _canonical_working_dir(context: Any) -> str:
    """The context's working dir as an absolute canonical path, "" when unset.

    The shared store attributes each file to its workspace by this value, so a
    relative ``-w`` must not reach the file: nothing could resolve it later.
    """
    working_dir = getattr(context, "working_dir", None)
    if not working_dir:
        return ""
    return str(Path(working_dir).expanduser().resolve())


def build_trajectory_path(
    *,
    config: TrajectoryRecorderConfig,
    state_dir: Path | None,
    agent: str,
    thread_id: str,
) -> Path:
    """Resolve the trajectory file for one conversation thread.

    ``state_dir`` anchors the workspace scope; the shared scope ignores it.
    """
    directory = store_dir(config, state_dir=state_dir)

    fields = {
        "agent": _sanitize_component(agent, "agent"),
        "thread_id": _sanitize_component(thread_id, "thread"),
    }
    try:
        filename = config.output.filename.format(**fields)
    except (KeyError, IndexError, ValueError):
        logger.warning("Invalid trajectory filename template %r; using default", config.output.filename)
        filename = _DEFAULT_FILENAME.format(**fields)
    return directory / filename


class _TrajectoryManager:
    """Process-wide registry of per-thread recorders and active turns."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recorders: dict[str, TrajectoryRecorder] = {}
        self._by_thread: dict[str, TrajectoryRecorder] = {}
        self._active_turns: dict[str, tuple[str, float]] = {}

    def reset(self) -> None:
        with self._lock:
            self._recorders.clear()
            self._by_thread.clear()
            self._active_turns.clear()

    # ------------------------------------------------------------- recorders

    def recorder_for(self, context: Any, *, create: bool = True) -> TrajectoryRecorder | None:
        config = load_trajectory_config()
        if not config.is_active:
            return None

        thread_id = str(getattr(context, "thread_id", "") or "")
        agent = str(getattr(context, "agent", "") or "agent")
        if not thread_id:
            return None

        if not create:
            with self._lock:
                return self._by_thread.get(thread_id)

        if config.is_shared:
            state_dir = None
        else:
            state_dir = self._resolve_state_dir(context)
            if state_dir is None:
                return None
        path = build_trajectory_path(config=config, state_dir=state_dir, agent=agent, thread_id=thread_id)
        key = str(path)

        with self._lock:
            recorder = self._recorders.get(key)
            if recorder is None:
                recorder = TrajectoryRecorder(
                    path=path,
                    config=config,
                    base_fields={"thread_id": thread_id, "agent": agent},
                )
                self._recorders[key] = recorder
                self._emit_attach(recorder, context, config)
            self._by_thread[thread_id] = recorder
            return recorder

    @staticmethod
    def _resolve_state_dir(context: Any) -> Path | None:
        state_dir = getattr(context, "state_dir", None)
        if state_dir:
            return Path(state_dir)
        working_dir = getattr(context, "working_dir", None)
        if not working_dir:
            return None
        try:
            from msagent.core.paths import AppPaths

            return AppPaths.resolve().for_project(Path(working_dir)).root
        except Exception:
            logger.debug("Cannot resolve project state dir for trajectory recording", exc_info=True)
            return None

    @staticmethod
    def _emit_attach(recorder: TrajectoryRecorder, context: Any, config: TrajectoryRecorderConfig) -> None:
        snapshot: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "capture_level": config.capture.level.value,
            "working_dir": _canonical_working_dir(context),
            "model": getattr(context, "model", None),
            "model_display": getattr(context, "model_display", None),
        }
        approval_mode = getattr(context, "approval_mode", None)
        if approval_mode is not None:
            snapshot["approval_mode"] = str(getattr(approval_mode, "value", approval_mode))
        try:
            from msagent.core.constants import APP_VERSION, OS_VERSION, PLATFORM

            snapshot.update({"app_version": APP_VERSION, "platform": PLATFORM, "os_version": OS_VERSION})
        except Exception:
            logger.debug("Cannot resolve app metadata for trajectory recording", exc_info=True)
        recorder.emit("recorder.attach", snapshot)

    # ------------------------------------------------------------ turn state

    def begin_turn(self, thread_id: str, run_id: str) -> None:
        with self._lock:
            self._active_turns[thread_id] = (run_id, time.perf_counter())

    def end_turn(self, thread_id: str) -> tuple[str | None, int | None]:
        with self._lock:
            entry = self._active_turns.pop(thread_id, None)
        if entry is None:
            return None, None
        run_id, started = entry
        return run_id, round((time.perf_counter() - started) * 1000)

    def active_run_id(self, thread_id: str) -> str | None:
        with self._lock:
            entry = self._active_turns.get(thread_id)
        return entry[0] if entry else None


_manager = _TrajectoryManager()


def reset() -> None:
    """Drop all cached recorders and turn state (used by tests)."""
    _manager.reset()


def instrument_config(
    graph_config: "RunnableConfig",
    *,
    context: Any,
    run_id: str,
    user_message: str | None = None,
    source: str = "dispatch",
) -> "RunnableConfig":
    """Attach trajectory recording to one graph invocation.

    Returns a config with the trajectory callback handler appended, and emits a
    ``turn.start`` event. On any failure (or when recording is disabled) the
    original config is returned unchanged.
    """
    try:
        recorder = _manager.recorder_for(context)
        if recorder is None:
            return graph_config

        thread_id = str(getattr(context, "thread_id", "") or "")
        _manager.begin_turn(thread_id, run_id)

        payload: dict[str, Any] = {
            "run_id": run_id,
            "source": source,
            "model": getattr(context, "model", None),
        }
        approval_mode = getattr(context, "approval_mode", None)
        if approval_mode is not None:
            payload["approval_mode"] = str(getattr(approval_mode, "value", approval_mode))
        if user_message is not None:
            payload["user_message"] = user_message
        recorder.emit("turn.start", payload)

        handler = TrajectoryCallbackHandler(
            recorder=recorder,
            capture=recorder.config.capture,
            turn_run_id=run_id,
        )
        existing = graph_config.get("callbacks") if isinstance(graph_config, dict) else None
        if existing is None:
            callbacks: Any = [handler]
        elif isinstance(existing, list):
            callbacks = [*existing, handler]
        else:
            try:
                existing.add_handler(handler, inherit=True)
            except Exception:
                logger.debug("Cannot attach trajectory handler to callback manager", exc_info=True)
            callbacks = existing

        new_config = dict(graph_config)
        new_config["callbacks"] = callbacks
        return new_config  # type: ignore[return-value]
    except Exception:
        logger.warning("Trajectory instrumentation failed; continuing without recording", exc_info=True)
        return graph_config


def finish_turn(
    *,
    context: Any,
    run_id: str | None = None,
    status: str = "completed",
    error: BaseException | None = None,
) -> None:
    """Emit a ``turn.end`` event for the active turn of this thread."""
    try:
        recorder = _manager.recorder_for(context, create=False)
        thread_id = str(getattr(context, "thread_id", "") or "")
        active_run_id, duration_ms = _manager.end_turn(thread_id)
        if recorder is None:
            return
        resolved_run_id = run_id or active_run_id
        if resolved_run_id is None:
            return

        payload: dict[str, Any] = {
            "run_id": resolved_run_id,
            "status": "error" if error is not None else status,
        }
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)
        recorder.emit("turn.end", payload)
    except Exception:
        logger.debug("Trajectory finish_turn failed", exc_info=True)


def record_approval(*, context: Any, interrupt: Any, resume_value: Any) -> None:
    """Record one human approval/interrupt decision."""
    try:
        recorder = _manager.recorder_for(context, create=False)
        if recorder is None:
            return
        thread_id = str(getattr(context, "thread_id", "") or "")
        recorder.emit(
            "approval.decision",
            {
                "run_id": _manager.active_run_id(thread_id),
                "interrupt_id": getattr(interrupt, "id", None),
                "request": json_safe(getattr(interrupt, "value", None)),
                "decision": json_safe(resume_value),
            },
        )
    except Exception:
        logger.debug("Trajectory record_approval failed", exc_info=True)


def record_compression(*, context: Any, **payload: Any) -> None:
    """Record a context compression / conversation offload event."""
    try:
        recorder = _manager.recorder_for(context)
        if recorder is None:
            return
        thread_id = str(getattr(context, "thread_id", "") or "")
        body: dict[str, Any] = {"run_id": _manager.active_run_id(thread_id)}
        body.update(json_safe(payload))
        recorder.emit("context.compression", body)
    except Exception:
        logger.debug("Trajectory record_compression failed", exc_info=True)


def record_event(*, context: Any, event: str, payload: dict[str, Any] | None = None) -> None:
    """Record an arbitrary custom event on the current thread's trajectory."""
    try:
        recorder = _manager.recorder_for(context)
        if recorder is None:
            return
        recorder.emit(event, dict(payload or {}))
    except Exception:
        logger.debug("Trajectory record_event failed", exc_info=True)