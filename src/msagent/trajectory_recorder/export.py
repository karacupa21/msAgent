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

"""Read and export recorded trajectories.

Standalone by design (stdlib + msagent.core.paths only, no langchain), usable
both as a library and as a CLI:

    python -m msagent.trajectory_recorder.export list [-w DIR] [--all-workspaces]
    python -m msagent.trajectory_recorder.export show --thread <id> [--max-chars N]
    python -m msagent.trajectory_recorder.export export --thread <id> --format json|jsonl|md [--output FILE]

In the shared store (``output.scope: shared``) every command sees the threads
recorded in ``--working-dir`` (default: cwd); ``--all-workspaces`` lifts that.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from msagent.trajectory_recorder.config import load_trajectory_config, store_dir
from msagent.trajectory_recorder.reader import (
    extract_message_text,
    file_workspace,
    iter_events,
    workspace_key,
)


def resolve_trajectories_dir(*, working_dir: Path | None = None, state_dir: Path | None = None) -> Path:
    """Locate the trajectories directory for a project.

    The shared scope has one directory for every workspace and ignores both
    arguments; per-workspace readers narrow it with :func:`workspace_filter`.
    """
    config = load_trajectory_config()
    if state_dir is None and not config.is_shared:
        from msagent.core.paths import AppPaths

        state_dir = AppPaths.resolve().for_project(working_dir or Path.cwd()).root
    return store_dir(config, state_dir=state_dir)


def workspace_filter(working_dir: Path | None = None) -> Path | None:
    """The workspace a per-workspace reader narrows the store to.

    ``None`` in the workspace scope, whose directory already is the workspace;
    the resolved ``working_dir`` (default: cwd) in the shared scope.
    """
    if not load_trajectory_config().is_shared:
        return None
    return Path(working_dir or Path.cwd()).expanduser().resolve()


def _workspace_id(workspace: Path | None) -> str | None:
    """:func:`workspace_key` of a filter argument, which must be an absolute path."""
    if workspace is None:
        return None
    key = workspace_key(workspace)
    if key is None:
        raise ValueError(f"workspace filter must be an absolute path, got {workspace}")
    return key


def _in_workspace(paths: list[Path], workspace: str | None) -> list[Path]:
    """The files recorded in ``workspace`` (a workspace key); all of them when None."""
    if workspace is None:
        return paths
    return [path for path in paths if file_workspace(path) == workspace]


def find_trajectory_file(
    trajectories_dir: Path,
    thread_id: str,
    *,
    workspace: Path | None = None,
) -> Path | None:
    """Find the trajectory file for a thread id (or an unambiguous prefix).

    With ``workspace`` only the files recorded in that workspace take part, so a
    prefix has to be unique within it.
    """
    if not trajectories_dir.is_dir():
        return None
    key = _workspace_id(workspace)

    def matches(pattern: str) -> list[Path]:
        return _in_workspace(sorted(trajectories_dir.glob(pattern)), key)

    exact = matches(f"*_{thread_id}.jsonl") or matches(f"{thread_id}.jsonl")
    if exact:
        return exact[0]
    prefixed = matches(f"*_{thread_id}*.jsonl")
    return prefixed[0] if len(prefixed) == 1 else None


@dataclass(slots=True)
class TrajectorySummary:
    path: Path
    thread_id: str
    agent: str
    events: int
    turns: int
    first_user_message: str
    size_bytes: int
    # working_dir of the first recorder.attach; "" when the file has none.
    working_dir: str = ""


def summarize_file(path: Path) -> TrajectorySummary:
    thread_id = ""
    agent = ""
    working_dir = ""
    events = 0
    turns = 0
    first_user_message = ""
    for event in iter_events(path):
        events += 1
        thread_id = thread_id or str(event.get("thread_id", ""))
        agent = agent or str(event.get("agent", ""))
        if event.get("event") == "recorder.attach" and not working_dir:
            working_dir = str(event.get("working_dir") or "")
        if event.get("event") == "turn.start":
            turns += 1
            if not first_user_message:
                first_user_message = str(event.get("user_message", "") or "")
    return TrajectorySummary(
        path=path,
        thread_id=thread_id or path.stem,
        agent=agent,
        events=events,
        turns=turns,
        first_user_message=first_user_message.replace("\n", " ")[:80],
        size_bytes=path.stat().st_size,
        working_dir=working_dir,
    )


def list_trajectories(
    trajectories_dir: Path,
    *,
    workspace: Path | None = None,
) -> list[TrajectorySummary]:
    """The directory's trajectories, newest first; ``workspace`` narrows them."""
    if not trajectories_dir.is_dir():
        return []
    files = sorted(trajectories_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    own = _in_workspace(files, _workspace_id(workspace))
    return [summarize_file(path) for path in own]


# --------------------------------------------------------------- rendering


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [{len(text) - limit} chars clipped]"


def render_markdown(events: Iterator[dict[str, Any]], *, max_chars: int = 2000) -> str:
    """Render a trajectory as human-readable markdown."""
    lines: list[str] = []
    for event in events:
        kind = event.get("event")
        ts = event.get("ts", "")
        if kind == "recorder.attach":
            lines.append(
                f"# Trajectory: {event.get('agent', '')} / {event.get('thread_id', '')}\n\n"
                f"- model: {event.get('model_display') or event.get('model') or 'unknown'}\n"
                f"- working_dir: {event.get('working_dir', '')}\n"
                f"- capture: {event.get('capture_level', '')}, recorder attached {ts}\n"
            )
        elif kind == "turn.start":
            lines.append(f"\n## Turn `{event.get('run_id', '')}` ({ts}, {event.get('source', 'dispatch')})\n")
            user_message = event.get("user_message")
            if user_message:
                lines.append(f"**User:**\n\n{_clip(str(user_message), max_chars)}\n")
        elif kind == "message.ai":
            text = extract_message_text(event.get("message", {}))
            usage = event.get("usage") or {}
            meta: list[str] = []
            if event.get("duration_ms") is not None:
                meta.append(f"{event['duration_ms']} ms")
            if usage:
                meta.append(f"tokens {usage.get('input_tokens', '?')}/{usage.get('output_tokens', '?')}")
            namespace = (event.get("graph") or {}).get("checkpoint_ns", "")
            title = "Assistant (subagent)" if namespace else "Assistant"
            lines.append(f"**{title}** ({', '.join(meta) or ts}):\n\n{_clip(text, max_chars)}\n")
            tool_calls = (event.get("message", {}).get("data", {}) or {}).get("tool_calls") or []
            for tool_call in tool_calls:
                args = json.dumps(tool_call.get("args", {}), ensure_ascii=False)
                lines.append(f"- tool call `{tool_call.get('name')}` → {_clip(args, max_chars)}")
        elif kind == "tool.result":
            if event.get("message"):
                text = extract_message_text(event["message"])
            else:
                text = str(event.get("output", ""))
            duration = f", {event['duration_ms']} ms" if event.get("duration_ms") is not None else ""
            lines.append(
                f"**Tool `{event.get('name')}`** ({event.get('status', 'ok')}{duration}):\n\n"
                f"```\n{_clip(text, max_chars)}\n```\n"
            )
        elif kind == "tool.error":
            lines.append(f"**Tool `{event.get('name')}` FAILED:** {event.get('error_type')}: {event.get('error')}\n")
        elif kind == "approval.decision":
            decision = json.dumps(event.get("decision"), ensure_ascii=False)
            lines.append(f"**Approval:** interrupt `{event.get('interrupt_id')}` → {_clip(decision, max_chars)}\n")
        elif kind == "context.compression":
            lines.append(f"**Context compressed** ({ts}): {json.dumps(event.get('run_id'))}\n")
        elif kind == "llm.retry":
            lines.append(f"*LLM retry attempt {event.get('attempt')}: {event.get('error_type')}*\n")
        elif kind == "turn.end":
            duration = f" in {event['duration_ms']} ms" if event.get("duration_ms") is not None else ""
            suffix = f" — {event.get('error_type')}: {event.get('error')}" if event.get("error") else ""
            lines.append(f"*Turn finished: {event.get('status')}{duration}{suffix}*\n")
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI


def _write_output(text: str, output: str | None) -> None:
    if output and output != "-":
        Path(output).write_text(text, encoding="utf-8")
        print(f"Written to {output}")
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="msagent.trajectory_recorder.export", description="Inspect recorded trajectories")
    parser.add_argument("command", choices=["list", "show", "export"])
    parser.add_argument("-w", "--working-dir", default=None, help="Project working directory (default: cwd)")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Project state dir (overrides --working-dir; unused by the shared store)",
    )
    parser.add_argument(
        "--all-workspaces",
        action="store_true",
        help="Shared store: read every workspace, not only --working-dir",
    )
    parser.add_argument("-t", "--thread", default=None, help="Thread id (or unique prefix) for show/export")
    parser.add_argument("-f", "--format", choices=["md", "json", "jsonl"], default="md")
    parser.add_argument("-o", "--output", default=None, help="Output file ('-' or omitted = stdout)")
    parser.add_argument("--max-chars", type=int, default=2000, help="Per-field clip in md output, 0 = no clipping")
    args = parser.parse_args(argv)

    working_dir = Path(args.working_dir) if args.working_dir else None
    trajectories_dir = resolve_trajectories_dir(
        working_dir=working_dir,
        state_dir=Path(args.state_dir) if args.state_dir else None,
    )
    workspace = None if args.all_workspaces else workspace_filter(working_dir)

    if args.command == "list":
        summaries = list_trajectories(trajectories_dir, workspace=workspace)
        if not summaries:
            print(f"No trajectories found in {trajectories_dir}")
            return 0
        for summary in summaries:
            line = (
                f"{summary.thread_id}  agent={summary.agent}  turns={summary.turns}  "
                f"events={summary.events}  size={summary.size_bytes / 1024:.1f}KB"
            )
            if args.all_workspaces:
                line = f"{line}  workspace={summary.working_dir or '?'}"
            print(f"{line}  | {summary.first_user_message}")
        return 0

    if not args.thread:
        parser.error(f"--thread is required for '{args.command}'")
    path = find_trajectory_file(trajectories_dir, args.thread, workspace=workspace)
    if path is None:
        print(f"No trajectory found for thread '{args.thread}' in {trajectories_dir}", file=sys.stderr)
        return 1

    if args.command == "show" or args.format == "md":
        _write_output(render_markdown(iter_events(path), max_chars=args.max_chars), args.output)
        return 0

    if args.format == "jsonl":
        _write_output(path.read_text(encoding="utf-8"), args.output)
        return 0

    document = {"file": str(path), "events": list(iter_events(path))}
    _write_output(json.dumps(document, ensure_ascii=False, indent=2), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
