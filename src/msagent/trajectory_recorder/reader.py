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

"""Build the typed trajectory model from recorder JSONL files.

Stdlib only (no langchain): this is the entry point for trajectory analysis
that must run in tests and CI without an LLM. Files are read line by line via
:func:`iter_numbered_events`; only the resulting model is retained in memory,
and every event keeps its physical line number (``EvidenceRef``).

Assembly rules:

* ``tool.start`` and ``tool.result``/``tool.error`` are joined by ``span_id``.
  A start without a result becomes ``status="orphan"`` (interrupted session);
  a result without a start (``capture.tool_starts=false``) gets ``args={}``.
* A turn opens on ``turn.start`` and closes on ``turn.end`` with the same
  ``run_id``. A turn that never ends (Ctrl+C) stays ``status="truncated"``.
* Callback events are routed by ``run_id``, so events written after their
  ``turn.end`` still land in the right turn. Events without a ``run_id`` go to
  the most recently started turn, or to a synthetic prelude turn
  (:data:`PRELUDE_RUN_ID`) when no turn has started yet.
* Broken JSON lines are skipped silently and counted in
  ``Trajectory.malformed_lines``; anything else that violates schema version 1
  raises :class:`TrajectoryReadError`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from msagent.trajectory_recorder.model import (
    PRELUDE_RUN_ID,
    AiMessage,
    Approval,
    ToolCall,
    ToolStatus,
    Trajectory,
    Turn,
    TurnStatus,
)

_SUPPORTED_SCHEMA_VERSION = 1
_TURN_END_STATUSES = frozenset({"completed", "error"})
_SKILL_TOOL = "get_skill"


class TrajectoryReadError(ValueError):
    """Raised when a file cannot be interpreted as a schema-v1 trajectory."""


# ---------------------------------------------------------------- parsing


def iter_numbered_events(
    path: Path,
    *,
    malformed: list[int] | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line, event)`` pairs from a trajectory JSONL file.

    ``line`` is the 1-based physical line number, the ``EvidenceRef.line`` of
    the event. Blank lines are ignored and broken lines are skipped, but both
    keep their number, so a corrupted line never shifts the refs after it.
    Lines that are not valid JSON or do not decode to an object are appended
    to ``malformed`` when it is given.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if malformed is not None:
                    malformed.append(line_no)
                continue
            if isinstance(payload, dict):
                yield line_no, payload
            elif malformed is not None:
                malformed.append(line_no)


def iter_events(
    path: Path,
    *,
    malformed: list[int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield events from a trajectory JSONL file, skipping broken lines.

    :func:`iter_numbered_events` without the line numbers.
    """
    for _, event in iter_numbered_events(path, malformed=malformed):
        yield event


def extract_message_text(message: Any) -> str:
    """Extract readable text from a serialized langchain message.

    Accepts the ``message_to_dict`` shape (``{"type": ..., "data": {...}}``).
    String content is returned as is; content blocks are joined with newlines
    (``text`` blocks verbatim, ``thinking`` blocks prefixed with
    ``[thinking]``, any other block as ``[<type>]``).
    """
    data = message.get("data", {}) if isinstance(message, dict) else {}
    content = data.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("thinking"), str):
                    parts.append(f"[thinking] {block['thinking']}")
                else:
                    parts.append(f"[{block.get('type', 'block')}]")
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------- helpers


def _subagent_of(event: dict[str, Any]) -> str | None:
    """Return the subagent namespace of an event, ``None`` for the root agent.

    langgraph namespaces read ``model:<task>`` for root-graph nodes and
    ``tools:<task>|model:<task>`` inside a subagent graph; the prefix without
    the last segment is shared by every event of one ``task`` invocation.
    """
    graph = event.get("graph")
    if not isinstance(graph, dict):
        return None
    namespace = graph.get("checkpoint_ns") or graph.get("langgraph_checkpoint_ns")
    if not isinstance(namespace, str):
        return None
    prefix, separator, _ = namespace.rpartition("|")
    return prefix if separator else None


def _tool_args(value: Any) -> dict[str, Any]:
    """Normalize ``tool.start.input``; legacy non-dict inputs are wrapped."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"input": value}


def _tool_result_text(event: dict[str, Any]) -> str:
    """Flatten the three ``tool.result`` payload shapes into text."""
    if "message" in event:
        return extract_message_text(event["message"])
    if event.get("command"):
        messages = event.get("messages") or []
        return "\n".join(extract_message_text(message) for message in messages)
    output = event.get("output")
    if output is None:
        return ""
    return output if isinstance(output, str) else str(output)


def _tool_calls_of(message: Any) -> list[dict[str, Any]]:
    """Return the structured ``tool_calls`` of a serialized AI message."""
    data = message.get("data") if isinstance(message, dict) else None
    calls = data.get("tool_calls") if isinstance(data, dict) else None
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def _skill_name(tool_name: Any, args: Any) -> str | None:
    """Return the skill consulted by a ``get_skill`` call, if any."""
    if tool_name != _SKILL_TOOL or not isinstance(args, dict):
        return None
    name = args.get("name")
    return name if isinstance(name, str) and name else None


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _is_v1_event(event: dict[str, Any], path: Path) -> bool:
    """Tell a schema-v1 event from a foreign object; newer schemas raise."""
    if "v" in event and event["v"] != _SUPPORTED_SCHEMA_VERSION:
        raise TrajectoryReadError(
            f"{path}: unsupported schema version {event['v']!r}",
        )
    has_kind = isinstance(event.get("event"), str)
    has_seq = isinstance(event.get("seq"), int)
    return event.get("v") == _SUPPORTED_SCHEMA_VERSION and has_kind and has_seq


# --------------------------------------------------------------- assembly


@dataclass(slots=True)
class _ToolSlot:
    """Mutable accumulator for one tool span; ``build`` freezes it."""

    span_id: str
    parent_span_id: str | None
    name: str
    args: dict[str, Any]
    subagent: str | None
    seq_start: int
    line_start: int
    seq_end: int | None = None
    line_end: int | None = None
    status: ToolStatus = "orphan"
    output_text: str = ""
    error_type: str | None = None
    error: str | None = None
    duration_ms: int | None = None

    def build(self) -> ToolCall:
        return ToolCall(
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            args=self.args,
            status=self.status,
            output_text=self.output_text,
            error_type=self.error_type,
            error=self.error,
            duration_ms=self.duration_ms,
            seq_start=self.seq_start,
            seq_end=self.seq_end,
            line_start=self.line_start,
            line_end=self.line_end,
            subagent=self.subagent,
        )


@dataclass(slots=True)
class _TurnState:
    turn: Turn
    slots: list[_ToolSlot] = field(default_factory=list)
    closed: bool = False


class _TrajectoryBuilder:
    """Accumulate the events of one file; ``finish`` produces the model."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.first_event: dict[str, Any] | None = None
        self.attach: dict[str, Any] | None = None
        self.first_turn_start: dict[str, Any] | None = None
        self.states: list[_TurnState] = []
        self.by_run_id: dict[str, _TurnState] = {}
        # Keyed by span_id and shared by all turns: a result may be written
        # after its turn ended and must still join its start.
        self.open_tools: dict[str, _ToolSlot] = {}
        self.skills: list[str] = []
        self.truncated_by_limit = False
        self.invalid_dicts = 0
        # Physical line of the event being fed; handlers copy it into the model.
        self.line = 0

    # ---------------------------------------------------------------- public

    def feed(self, event: dict[str, Any], line: int) -> None:
        """Consume one valid schema-v1 event recorded at physical ``line``."""
        if self.first_event is None:
            self.first_event = event
        self.line = line
        handler = self._HANDLERS.get(event["event"])
        if handler is not None:
            handler(self, event)

    def finish(self, malformed_lines: int) -> Trajectory:
        """Freeze the accumulated state into a :class:`Trajectory`."""
        first = self.first_event
        if first is None:
            raise TrajectoryReadError(f"{self.path}: no events found")
        attach = self.attach or {}
        model = attach.get("model_display") or attach.get("model")
        if not model:
            model = (self.first_turn_start or {}).get("model")
        turns: list[Turn] = []
        for state in self.states:
            state.turn.tool_calls = [slot.build() for slot in state.slots]
            turns.append(state.turn)
        return Trajectory(
            path=self.path,
            thread_id=str(first.get("thread_id") or self.path.stem),
            agent=str(first.get("agent") or ""),
            model=str(model) if model else None,
            working_dir=str(attach.get("working_dir") or ""),
            started_at=str(first.get("ts") or ""),
            turns=turns,
            truncated_by_limit=self.truncated_by_limit,
            skills_consulted=list(self.skills),
            malformed_lines=malformed_lines,
        )

    # --------------------------------------------------------------- routing

    def _open_turn(
        self,
        run_id: str,
        seq: int,
        source: str,
        user_message: str | None,
        *,
        closed: bool = False,
        status: TurnStatus = "truncated",
    ) -> _TurnState:
        turn = Turn(
            run_id=run_id,
            seq_start=seq,
            line_start=self.line,
            user_message=user_message,
            source=source,
            status=status,
        )
        state = _TurnState(turn=turn, closed=closed)
        self.states.append(state)
        if run_id != PRELUDE_RUN_ID:
            self.by_run_id[run_id] = state
        return state

    def _route(self, event: dict[str, Any]) -> _TurnState:
        """Find the turn an event belongs to, creating one when needed."""
        run_id = event.get("run_id")
        if isinstance(run_id, str) and run_id:
            state = self.by_run_id.get(run_id)
            return state or self._open_turn(run_id, event["seq"], "unknown", None)
        if self.states:
            return self.states[-1]
        return self._open_turn(
            PRELUDE_RUN_ID,
            event["seq"],
            "prelude",
            None,
            closed=True,
            status="completed",
        )

    def _require_str(self, event: dict[str, Any], key: str) -> str:
        value = event.get(key)
        if not isinstance(value, str) or not value:
            raise TrajectoryReadError(
                f"{self.path}: {event['event']} at seq {event['seq']} has no {key!r}",
            )
        return value

    def _add_skill(self, name: str | None) -> None:
        if name is not None and name not in self.skills:
            self.skills.append(name)

    # -------------------------------------------------------------- handlers

    def _on_attach(self, event: dict[str, Any]) -> None:
        if self.attach is None:
            self.attach = event

    def _on_limit(self, _event: dict[str, Any]) -> None:
        self.truncated_by_limit = True

    def _on_turn_start(self, event: dict[str, Any]) -> None:
        for state in self.states:
            if not state.closed:
                state.turn.status = "truncated"
                state.closed = True
        if self.first_turn_start is None:
            self.first_turn_start = event
        user_message = event.get("user_message")
        self._open_turn(
            self._require_str(event, "run_id"),
            event["seq"],
            str(event.get("source") or "dispatch"),
            None if user_message is None else str(user_message),
        )

    def _on_turn_end(self, event: dict[str, Any]) -> None:
        run_id = self._require_str(event, "run_id")
        state = self.by_run_id.get(run_id)
        if state is None:
            state = self._open_turn(run_id, event["seq"], "unknown", None)
        status = event.get("status")
        if status not in _TURN_END_STATUSES:
            raise TrajectoryReadError(
                f"{self.path}: turn.end of {run_id!r} has unexpected status {status!r}",
            )
        state.turn.status = status
        state.turn.duration_ms = _optional_int(event.get("duration_ms"))
        state.turn.error_type = _optional_str(event.get("error_type"))
        state.closed = True

    def _on_message_ai(self, event: dict[str, Any]) -> None:
        state = self._route(event)
        message = event.get("message")
        calls = _tool_calls_of(message)
        names = [str(call["name"]) for call in calls if call.get("name") is not None]
        usage = event.get("usage")
        state.turn.ai_messages.append(
            AiMessage(
                seq=event["seq"],
                span_id=self._require_str(event, "span_id"),
                text=extract_message_text(message),
                tool_call_names=names,
                usage=usage if isinstance(usage, dict) else None,
                duration_ms=_optional_int(event.get("duration_ms")),
                subagent=_subagent_of(event),
                line=self.line,
            ),
        )
        for call in calls:
            self._add_skill(_skill_name(call.get("name"), call.get("args")))

    def _on_tool_start(self, event: dict[str, Any]) -> None:
        state = self._route(event)
        name = str(event.get("name") or "tool")
        args = _tool_args(event.get("input"))
        slot = _ToolSlot(
            span_id=self._require_str(event, "span_id"),
            parent_span_id=_optional_str(event.get("parent_span_id")),
            name=name,
            args=args,
            subagent=_subagent_of(event),
            seq_start=event["seq"],
            line_start=self.line,
        )
        state.slots.append(slot)
        self.open_tools[slot.span_id] = slot
        self._add_skill(_skill_name(name, args))

    def _slot_for_result(self, event: dict[str, Any]) -> _ToolSlot:
        """Join a result with its start, or open a start-less slot."""
        span_id = self._require_str(event, "span_id")
        slot = self.open_tools.pop(span_id, None)
        if slot is None:
            slot = _ToolSlot(
                span_id=span_id,
                parent_span_id=_optional_str(event.get("parent_span_id")),
                name=str(event.get("name") or "tool"),
                args={},
                subagent=_subagent_of(event),
                seq_start=event["seq"],
                line_start=self.line,
            )
            self._route(event).slots.append(slot)
        slot.seq_end = event["seq"]
        slot.line_end = self.line
        slot.duration_ms = _optional_int(event.get("duration_ms"))
        return slot

    def _on_tool_result(self, event: dict[str, Any]) -> None:
        slot = self._slot_for_result(event)
        slot.status = "error" if event.get("status") == "error" else "ok"
        slot.output_text = _tool_result_text(event)

    def _on_tool_error(self, event: dict[str, Any]) -> None:
        slot = self._slot_for_result(event)
        slot.status = "error"
        slot.error_type = _optional_str(event.get("error_type"))
        slot.error = _optional_str(event.get("error"))

    def _on_retry(self, event: dict[str, Any]) -> None:
        self._route(event).turn.retries += 1

    def _on_approval(self, event: dict[str, Any]) -> None:
        self._route(event).turn.approvals.append(
            Approval(
                seq=event["seq"],
                run_id=_optional_str(event.get("run_id")),
                interrupt_id=_optional_str(event.get("interrupt_id")),
                request=event.get("request"),
                decision=event.get("decision"),
                line=self.line,
            ),
        )

    def _on_compression(self, event: dict[str, Any]) -> None:
        self._route(event).turn.compressions += 1

    _HANDLERS: dict[str, Callable[..., None]] = {
        "recorder.attach": _on_attach,
        "recorder.limit": _on_limit,
        "turn.start": _on_turn_start,
        "turn.end": _on_turn_end,
        "message.ai": _on_message_ai,
        "tool.start": _on_tool_start,
        "tool.result": _on_tool_result,
        "tool.error": _on_tool_error,
        "llm.retry": _on_retry,
        "approval.decision": _on_approval,
        "context.compression": _on_compression,
    }


# ---------------------------------------------------------------- loaders


def load_trajectory(path: Path) -> Trajectory:
    """Read one trajectory file into a :class:`Trajectory`.

    Raises :class:`TrajectoryReadError` when the file holds no usable event
    or violates schema version 1 in a way that cannot be skipped.
    """
    path = Path(path)
    builder = _TrajectoryBuilder(path)
    malformed: list[int] = []
    for line, event in iter_numbered_events(path, malformed=malformed):
        if _is_v1_event(event, path):
            builder.feed(event, line)
        else:
            builder.invalid_dicts += 1
    return builder.finish(len(malformed) + builder.invalid_dicts)


def load_trajectories(
    directory: Path,
    *,
    agent: str | None = None,
    thread_ids: Sequence[str] | None = None,
    workspace: Path | None = None,
    limit: int | None = None,
) -> list[Trajectory]:
    """Load the trajectories of a directory, newest (by mtime) first.

    ``agent`` and ``thread_ids`` filter on the envelope of each file's first
    event, ``workspace`` (an absolute directory) on the ``working_dir`` that
    event records (see :func:`workspace_key`); ``limit`` keeps only the newest
    files after filtering. A missing directory yields an empty list.
    """
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    workspace_id = None
    if workspace is not None:
        workspace_id = workspace_key(workspace)
        if workspace_id is None:
            raise ValueError(f"workspace must be an absolute path, got {workspace}")
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = sorted(
        directory.glob("*.jsonl"),
        key=lambda file: (file.stat().st_mtime, file.name),
        reverse=True,
    )
    wanted = None if thread_ids is None else set(thread_ids)
    if agent is not None or wanted is not None or workspace_id is not None:
        files = [f for f in files if _header_matches(f, agent, wanted, workspace_id)]
    if limit is not None:
        files = files[:limit]
    return [load_trajectory(file) for file in files]


def workspace_key(path: str | Path) -> str | None:
    """Identity of a workspace directory; ``None`` when it cannot be known.

    Canonical like ``ProjectPaths.resolve`` (resolved, case-normalized), so a
    recorded ``working_dir`` and a live one compare equal whenever they name
    the same directory. A relative path yields ``None``: resolving it would
    anchor it at the reading process's cwd, not at the recording one's.
    """
    if not str(path).strip():
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return None
    return os.path.normcase(str(candidate.resolve()))


def file_workspace(path: Path) -> str | None:
    """:func:`workspace_key` of the ``working_dir`` a file's first event records.

    Reads one line. A file with no event, or whose first event records no
    absolute ``working_dir``, belongs to no workspace.
    """
    events = iter_events(path)
    try:
        header = next(events, None)
    finally:
        events.close()
    return None if header is None else _event_workspace(header)


def _event_workspace(event: dict[str, Any]) -> str | None:
    value = event.get("working_dir")
    return workspace_key(value) if isinstance(value, str) else None


def _header_matches(
    path: Path,
    agent: str | None,
    thread_ids: set[str] | None,
    workspace: str | None,
) -> bool:
    header = _peek_header(path)
    if agent is not None and header.get("agent") != agent:
        return False
    if thread_ids is not None and header.get("thread_id") not in thread_ids:
        return False
    return workspace is None or _event_workspace(header) == workspace


def _peek_header(path: Path) -> dict[str, Any]:
    """Return the first event of a file without reading the rest of it."""
    events = iter_events(path)
    try:
        return next(events)
    except StopIteration:
        raise TrajectoryReadError(f"{path}: no events found") from None
    finally:
        events.close()
