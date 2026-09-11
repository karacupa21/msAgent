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

"""Tests for the typed trajectory reader (hand-written JSONL fixtures, no LLM)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from msagent.trajectory_recorder import export, reader
from msagent.trajectory_recorder.model import PRELUDE_RUN_ID, EvidenceRef, ToolCall, Trajectory, Turn
from msagent.trajectory_recorder.reader import (
    TrajectoryReadError,
    extract_message_text,
    iter_events,
    iter_numbered_events,
    load_trajectories,
    load_trajectory,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "trajectories"


def _load(name: str) -> Trajectory:
    return load_trajectory(FIXTURES / name)


def _tool(turn: Turn, span_id: str) -> ToolCall:
    matches = [call for call in turn.tool_calls if call.span_id == span_id]
    assert len(matches) == 1, f"expected one tool call {span_id!r}, got {matches!r}"
    return matches[0]


def _line(seq: int, event: str, **payload: Any) -> str:
    envelope = {
        "v": 1,
        "event": event,
        "ts": f"2026-09-01T10:00:{seq:02d}.000+00:00",
        "seq": seq,
        "rec": "rec-t",
        "thread_id": "thread-t",
        "agent": "Tester",
    }
    return json.dumps({**envelope, **payload})


def _write(path: Path, *lines: str) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _seq_at(path: Path, line: int) -> int:
    """The ``seq`` written on physical ``line`` of the file (1-based).

    Reads with ``open()`` + ``enumerate`` like the reader does, never with
    ``splitlines()``: the recorder writes ``ensure_ascii=False``, so a
    ``\u2028`` inside a string would split differently.
    """
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            if number == line:
                return json.loads(raw)["seq"]
    raise AssertionError(f"{path} has no line {line}")


# ------------------------------------------------------- normal_subagent


def test_normal_session_header() -> None:
    trajectory = _load("normal_subagent.jsonl")
    assert trajectory.path == FIXTURES / "normal_subagent.jsonl"
    assert trajectory.thread_id == "thread-normal"
    assert trajectory.agent == "Profiler"
    assert trajectory.model == "deepseek-v4-pro (openai)"
    assert trajectory.working_dir == "/work/proj"
    assert trajectory.started_at == "2026-09-01T10:00:01.000+00:00"
    assert trajectory.truncated_by_limit is False
    assert trajectory.malformed_lines == 0
    assert trajectory.skills_consulted == ["cluster-analysis"]
    assert [turn.run_id for turn in trajectory.turns] == ["run-1", "run-2"]


def test_normal_session_turns() -> None:
    first, second = _load("normal_subagent.jsonl").turns

    assert (first.status, first.duration_ms, first.error_type) == (
        "completed",
        1500,
        None,
    )
    assert (first.source, first.user_message, first.seq_start) == (
        "dispatch",
        "Analyse the profile",
        2,
    )
    assert (first.retries, first.compressions) == (0, 0)
    assert [approval.interrupt_id for approval in first.approvals] == ["int-1"]
    assert (first.approvals[0].seq, first.approvals[0].run_id) == (16, None)
    assert first.approvals[0].decision == {"action": "approve"}
    assert first.approvals[0].request == {
        "tool": "bash",
        "args": {"cmd": "rm -rf build"},
    }

    messages = first.ai_messages
    assert [message.span_id for message in messages] == ["s1", "s3", "s5", "s7", "s8"]
    assert [message.subagent for message in messages] == [
        None,
        None,
        "tools:b2",
        "tools:b2",
        None,
    ]
    assert messages[0].tool_call_names == ["get_skill"]
    assert messages[0].usage == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert (messages[0].seq, messages[0].duration_ms) == (4, 300)
    assert messages[2].usage is None
    assert messages[4].text == "[thinking] plan\nFinal answer"

    assert (second.status, second.error_type, second.duration_ms) == (
        "error",
        "RuntimeError",
        800,
    )
    assert (second.source, second.user_message, second.seq_start) == (
        "resume",
        None,
        17,
    )
    assert (second.retries, second.compressions, second.approvals) == (1, 1, [])
    assert [message.text for message in second.ai_messages] == [
        "Retrying the tools.",
        "Done.",
        "late",
    ]


def test_normal_session_tool_calls() -> None:
    first, second = _load("normal_subagent.jsonl").turns

    assert [call.span_id for call in first.tool_calls] == ["s2", "s4", "s6"]
    skill = _tool(first, "s2")
    assert (skill.name, skill.status) == ("get_skill", "ok")
    assert skill.args == {"name": "cluster-analysis", "category": "profiler"}
    assert skill.output_text.startswith("# cluster-analysis")
    assert (skill.seq_start, skill.seq_end, skill.duration_ms) == (5, 6, 12)
    assert (skill.parent_span_id, skill.subagent) == ("chain-2", None)
    assert (skill.error_type, skill.error) == (None, None)
    task = _tool(first, "s4")
    assert (task.status, task.output_text) == ("ok", "Delegation done: 2 files")
    assert (task.seq_start, task.seq_end, task.duration_ms) == (8, 13, 900)
    listing = _tool(first, "s6")
    assert (listing.status, listing.output_text) == ("ok", "a.txt\nb.txt")
    assert listing.subagent == "tools:b2"

    assert [call.span_id for call in second.tool_calls] == ["s10", "s11", "s12"]
    bash = _tool(second, "s10")
    assert (bash.status, bash.error_type, bash.error) == (
        "error",
        "ToolException",
        "boom",
    )
    assert (bash.args, bash.output_text, bash.duration_ms) == (
        {"input": "ls -la"},
        "",
        7,
    )
    grep = _tool(second, "s11")
    assert (grep.status, grep.error_type, grep.error) == ("error", None, None)
    assert grep.output_text == "grep: no matches"
    assert _tool(second, "s12").status == "ok"


# ------------------------------------------------------- other fixtures


def test_orphan_tool_start() -> None:
    trajectory = _load("orphan_tool_start.jsonl")
    assert [turn.status for turn in trajectory.turns] == ["completed"] * 4
    first, second, third, fourth = trajectory.turns

    assert [(call.span_id, call.status) for call in first.tool_calls] == [
        ("s2", "orphan"),
        ("s3", "ok"),
        ("s4", "error"),
        ("s6", "ok"),
    ]
    orphan = _tool(first, "s2")
    assert (orphan.seq_start, orphan.seq_end, orphan.duration_ms) == (4, None, None)
    assert (orphan.args, orphan.output_text) == ({"path": "a.py"}, "")
    failed = _tool(first, "s4")
    assert (failed.args, failed.error_type) == ({}, "PermissionError")

    late = _tool(second, "s8")
    assert (late.status, late.seq_start, late.seq_end) == ("ok", 15, 17)
    assert (late.output_text, late.duration_ms) == ("built", 90)
    assert [(c.span_id, c.status) for c in third.tool_calls] == [("s10", "orphan")]
    assert [call.status for call in fourth.tool_calls] == ["ok"] * 3


def test_missing_turn_end() -> None:
    trajectory = _load("missing_turn_end.jsonl")
    assert [turn.status for turn in trajectory.turns] == [
        "truncated",
        "completed",
        "truncated",
    ]
    first, second, third = trajectory.turns

    assert (first.duration_ms, first.error_type) == (None, None)
    assert [call.status for call in first.tool_calls] == ["ok"] * 4
    assert _tool(first, "s2").args == {"input": [1, 2]}
    assert second.duration_ms == 200
    assert (third.seq_start, third.user_message) == (2, "Third")
    assert [(call.span_id, call.status) for call in third.tool_calls] == [
        ("s14", "ok"),
        ("s16", "ok"),
        ("s18", "orphan"),
    ]
    # The second recorder.attach (process restart) must not override the header.
    assert trajectory.model == "deepseek-v4-pro (openai)"
    assert trajectory.working_dir == "/work/proj"
    assert trajectory.started_at == "2026-09-01T10:00:01.000+00:00"


def test_malformed_lines_and_prelude() -> None:
    trajectory = _load("malformed_lines.jsonl")
    assert trajectory.malformed_lines == 9
    assert [turn.run_id for turn in trajectory.turns] == [
        PRELUDE_RUN_ID,
        "run-1",
        "run-2",
        "run-3",
    ]

    prelude = trajectory.turns[0]
    assert (prelude.status, prelude.source, prelude.seq_start) == (
        "completed",
        "prelude",
        2,
    )
    assert prelude.user_message is None
    assert (prelude.compressions, len(prelude.approvals)) == (1, 1)
    approval = prelude.approvals[0]
    assert (approval.interrupt_id, approval.seq, approval.run_id) == ("int-0", 3, None)
    assert (prelude.ai_messages, prelude.tool_calls) == ([], [])

    _, run_1, run_2, run_3 = trajectory.turns
    assert [turn.status for turn in (run_1, run_2, run_3)] == ["completed"] * 3
    assert [(c.span_id, c.status) for c in run_1.tool_calls] == [("s2", "ok")]
    assert [(c.span_id, c.status) for c in run_2.tool_calls] == [("s5", "ok")]
    assert [(c.span_id, c.status) for c in run_3.tool_calls] == [("s9", "ok")]


def test_lines_point_at_physical_lines_across_corrupted_ones() -> None:
    # Lines 4-8, 12, 20-21, 28 and 31 of the fixture are blank, broken or not
    # schema-v1 events; every model object must still carry the physical line
    # it was read from, and that line must hold the same seq.
    trajectory = _load("malformed_lines.jsonl")
    assert trajectory.source == "malformed_lines.jsonl"
    prelude, run_1, _, run_3 = trajectory.turns
    assert (prelude.seq_start, prelude.line_start) == (2, 2)
    approval = prelude.approvals[0]
    assert (approval.seq, approval.line) == (3, 3)
    assert (run_1.seq_start, run_1.line_start) == (9, 9)
    assert (run_1.ai_messages[0].seq, run_1.ai_messages[0].line) == (10, 10)
    s2 = _tool(run_1, "s2")
    # Line 12 is a truncated copy of the result; the real one is line 13.
    assert (s2.line_start, s2.seq_end, s2.line_end) == (11, 13, 13)
    s9 = _tool(run_3, "s9")
    assert (s9.line_start, s9.line_end) == (26, 27)
    for line, seq in ((2, 2), (3, 3), (9, 9), (10, 10), (11, 11), (13, 13), (26, 26), (27, 27)):
        assert _seq_at(trajectory.path, line) == seq
    numbered = [number for number, _ in iter_numbered_events(trajectory.path)]
    assert numbered == [1, 2, 3, 4, 9, 10, 11, *range(13, 31)]


def test_refs_differ_after_a_recorder_restart() -> None:
    # missing_turn_end.jsonl: writer rec-b restarts seq at 1 on line 22, so
    # seq 4 names two different events; their refs must not collide.
    trajectory = _load("missing_turn_end.jsonl")
    first, _, third = trajectory.turns
    assert (first.seq_start, first.line_start) == (2, 2)
    assert (third.seq_start, third.line_start) == (2, 23)
    s2, s14 = _tool(first, "s2"), _tool(third, "s14")
    assert (s2.seq_start, s2.line_start, s2.line_end) == (4, 4, 5)
    assert (s14.seq_start, s14.line_start, s14.line_end) == (4, 25, 26)
    assert _tool(third, "s18").line_end is None
    early = trajectory.event_ref(seq=s2.seq_start, line=s2.line_start)
    late = trajectory.event_ref(seq=s14.seq_start, line=s14.line_start)
    assert early == EvidenceRef(source="missing_turn_end.jsonl", line=4, seq=4)
    assert early != late and early.seq == late.seq
    for call in (s2, s14):
        assert _seq_at(trajectory.path, call.line_start) == call.seq_start


def test_iter_events_reports_line_numbers() -> None:
    malformed: list[int] = []
    events = list(iter_events(FIXTURES / "malformed_lines.jsonl", malformed=malformed))
    assert malformed == [5, 7, 8, 12, 31]
    assert len(events) == 25
    assert list(iter_events(FIXTURES / "malformed_lines.jsonl")) == events


def test_recorder_limit() -> None:
    trajectory = _load("recorder_limit.jsonl")
    assert trajectory.truncated_by_limit is True
    first, second = trajectory.turns
    assert (first.status, first.duration_ms) == ("completed", 2000)
    assert [call.status for call in first.tool_calls] == ["ok"] * 4
    assert (second.status, second.duration_ms) == ("truncated", None)
    assert [(call.span_id, call.status) for call in second.tool_calls] == [
        ("s11", "ok"),
        ("s13", "ok"),
        ("s15", "ok"),
        ("s17", "ok"),
        ("s19", "orphan"),
    ]
    assert len(second.ai_messages) == 5


def test_tool_result_without_start() -> None:
    trajectory = _load("result_without_start.jsonl")
    first, second = trajectory.turns
    for call in first.tool_calls + second.tool_calls:
        assert call.args == {}
        assert call.seq_start == call.seq_end
        assert call.line_start == call.line_end

    assert [(call.span_id, call.status) for call in first.tool_calls] == [
        ("s2", "ok"),
        ("s4", "error"),
        ("s6", "ok"),
        ("s7", "ok"),
        ("s8", "error"),
    ]
    assert _tool(first, "s2").output_text.startswith("# dit-quant")
    bash = _tool(first, "s4")
    assert (bash.error_type, bash.output_text) == (None, "exit 1")
    assert _tool(first, "s6").output_text == "sub done"
    listing = _tool(first, "s7")
    assert (listing.subagent, listing.output_text) == ("tools:b2", "a b")
    assert _tool(first, "s8").error_type == "PermissionError"

    assert len(second.tool_calls) == 9
    assert [c.span_id for c in second.tool_calls if c.status == "error"] == ["s22"]
    assert trajectory.skills_consulted == ["dit-quant"]


# ------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({"type": "ai", "data": {"content": "plain"}}, "plain"),
        (
            {
                "type": "ai",
                "data": {
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "thinking", "thinking": "b"},
                        {"type": "tool_use", "id": "x"},
                        "raw",
                    ],
                },
            },
            "a\n[thinking] b\n[tool_use]\nraw",
        ),
        ({"type": "ai", "data": {"content": [{"foo": 1}]}}, "[block]"),
        ({"type": "ai", "data": {"content": 42}}, "42"),
        ({"type": "ai", "data": {}}, ""),
        ({"type": "ai"}, ""),
        (None, ""),
        ("not a dict", ""),
    ],
)
def test_extract_message_text(message: Any, expected: str) -> None:
    assert extract_message_text(message) == expected


# ------------------------------------------------------- loud failures


def test_zero_events_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(TrajectoryReadError, match="no events"):
        load_trajectory(empty)

    garbage = _write(tmp_path / "garbage.jsonl", "not json", "[1]", '{"foo": 1}')
    with pytest.raises(TrajectoryReadError, match="no events"):
        load_trajectory(garbage)


def test_unsupported_schema_version_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "v2.jsonl",
        _line(1, "recorder.attach"),
        '{"v": 2, "event": "turn.start", "seq": 2, "run_id": "run-1"}',
    )
    with pytest.raises(TrajectoryReadError, match="schema version 2"):
        load_trajectory(path)


def test_bad_turn_end_status_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bad.jsonl",
        _line(1, "turn.start", run_id="run-1"),
        _line(2, "turn.end", run_id="run-1", status="cancelled"),
    )
    with pytest.raises(TrajectoryReadError, match="cancelled"):
        load_trajectory(path)


def test_missing_span_id_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "nospan.jsonl",
        _line(1, "turn.start", run_id="run-1"),
        _line(2, "tool.start", run_id="run-1", name="bash", input={}),
    )
    with pytest.raises(TrajectoryReadError, match="span_id"):
        load_trajectory(path)


def test_turn_end_without_start_creates_implicit_turn(tmp_path: Path) -> None:
    ai_message = {"type": "ai", "data": {"content": "hi"}}
    graph = {"checkpoint_ns": None, "langgraph_node": "model"}
    path = _write(
        tmp_path / "implicit.jsonl",
        _line(1, "recorder.attach", working_dir="/w"),
        _line(
            2,
            "message.ai",
            run_id="run-x",
            span_id="s1",
            message=ai_message,
            graph=graph,
        ),
        _line(3, "turn.end", run_id="run-x", status="completed", duration_ms=5),
        _line(4, "turn.start", run_id="run-y", source="dispatch", user_message="next"),
        _line(5, "llm.retry", run_id=None, attempt=1),
        _line(6, "tool.result", run_id="run-z", span_id="s2", name="bash", output=None),
        _line(7, "turn.end", run_id="run-w", status="error", error_type="Boom"),
    )
    trajectory = load_trajectory(path)

    assert trajectory.model is None
    assert trajectory.working_dir == "/w"
    assert [turn.run_id for turn in trajectory.turns] == [
        "run-x",
        "run-y",
        "run-z",
        "run-w",
    ]
    implicit, started, trailing, ended = trajectory.turns
    assert (implicit.source, implicit.user_message, implicit.seq_start) == (
        "unknown",
        None,
        2,
    )
    assert (implicit.status, implicit.duration_ms) == ("completed", 5)
    assert [message.text for message in implicit.ai_messages] == ["hi"]
    assert implicit.ai_messages[0].subagent is None
    # A turn.end for a run_id never seen before still produces a (closed) turn.
    assert (ended.source, ended.status) == ("unknown", "error")
    assert (ended.error_type, ended.seq_start, ended.duration_ms) == ("Boom", 7, None)
    # A null run_id after a turn exists goes to the most recently started turn.
    assert (started.status, started.retries) == ("truncated", 1)
    assert (trailing.source, trailing.status) == ("unknown", "truncated")
    assert [(c.span_id, c.status, c.output_text) for c in trailing.tool_calls] == [
        ("s2", "ok", ""),
    ]


# ------------------------------------------------------- directory load


def test_load_trajectories_order_filters_limit(tmp_path: Path) -> None:
    directory = tmp_path / "trajectories"
    directory.mkdir()
    (directory / "sub").mkdir()
    shutil.copy(FIXTURES / "normal_subagent.jsonl", directory / "sub" / "nested.jsonl")
    (directory / "notes.txt").write_text("ignored", encoding="utf-8")
    base = 1_700_000_000
    ages = {
        "orphan_tool_start.jsonl": base + 30,
        "normal_subagent.jsonl": base + 20,
        "recorder_limit.jsonl": base + 10,
    }
    for name, mtime in ages.items():
        target = directory / name
        shutil.copy(FIXTURES / name, target)
        os.utime(target, (mtime, mtime))

    loaded = load_trajectories(directory)
    assert [trajectory.path.name for trajectory in loaded] == [
        "orphan_tool_start.jsonl",
        "normal_subagent.jsonl",
        "recorder_limit.jsonl",
    ]
    by_agent = load_trajectories(directory, agent="Quantizer")
    assert [trajectory.thread_id for trajectory in by_agent] == [
        "thread-orphan",
        "thread-limit",
    ]
    by_thread = load_trajectories(directory, thread_ids=["thread-normal"])
    assert [trajectory.thread_id for trajectory in by_thread] == ["thread-normal"]
    newest = load_trajectories(directory, limit=2)
    assert [trajectory.path.name for trajectory in newest] == [
        "orphan_tool_start.jsonl",
        "normal_subagent.jsonl",
    ]
    assert load_trajectories(directory, limit=0) == []
    combined = load_trajectories(directory, agent="Quantizer", limit=1)
    assert [trajectory.thread_id for trajectory in combined] == ["thread-orphan"]
    none = load_trajectories(directory, agent="Quantizer", thread_ids=["thread-normal"])
    assert none == []


def test_load_trajectories_edge_cases(tmp_path: Path) -> None:
    assert load_trajectories(tmp_path / "missing") == []
    with pytest.raises(ValueError, match="limit"):
        load_trajectories(tmp_path, limit=-1)

    directory = tmp_path / "trajectories"
    directory.mkdir()
    (directory / "empty.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(TrajectoryReadError, match="no events"):
        load_trajectories(directory)
    with pytest.raises(TrajectoryReadError, match="no events"):
        load_trajectories(directory, agent="Profiler")


# ------------------------------------------------------- workspace filter


def _attached(directory: Path, thread_id: str, working_dir: str | None, *, mtime: int) -> Path:
    """A complete one-turn file whose recorder.attach records ``working_dir``."""
    attach = {} if working_dir is None else {"working_dir": working_dir}
    path = _write(
        directory / f"Tester_{thread_id}.jsonl",
        _line(1, "recorder.attach", thread_id=thread_id, **attach),
        _line(2, "turn.start", thread_id=thread_id, run_id="run-1", source="dispatch"),
        _line(3, "turn.end", thread_id=thread_id, run_id="run-1", status="completed"),
    )
    os.utime(path, (mtime, mtime))
    return path


def test_workspace_key_is_canonical_and_absolute_only(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    assert reader.workspace_key(work) == os.path.normcase(str(work.resolve()))
    assert reader.workspace_key(f"{work}/../work") == reader.workspace_key(str(work))
    assert reader.workspace_key("work") is None
    assert reader.workspace_key("") is None


def test_load_trajectories_filters_by_workspace(tmp_path: Path) -> None:
    mine, other = tmp_path / "mine", tmp_path / "other"
    directory = tmp_path / "store"
    directory.mkdir()
    base = 1_700_000_000
    _attached(directory, "t-new", str(mine), mtime=base + 40)
    _attached(directory, "t-other", str(other), mtime=base + 30)
    _attached(directory, "t-old", str(mine), mtime=base + 20)
    _attached(directory, "t-relative", "mine", mtime=base + 10)
    _attached(directory, "t-unknown", None, mtime=base)

    loaded = load_trajectories(directory, workspace=mine)
    assert [trajectory.thread_id for trajectory in loaded] == ["t-new", "t-old"]
    # limit keeps the newest of the workspace, not of the whole store.
    newest = load_trajectories(directory, workspace=mine, limit=1)
    assert [trajectory.thread_id for trajectory in newest] == ["t-new"]
    theirs = load_trajectories(directory, workspace=other, agent="Tester")
    assert [trajectory.thread_id for trajectory in theirs] == ["t-other"]
    assert len(load_trajectories(directory)) == 5
    with pytest.raises(ValueError, match="absolute"):
        load_trajectories(directory, workspace=Path("mine"))


def test_export_lookups_filter_by_workspace(tmp_path: Path) -> None:
    mine, other = tmp_path / "mine", tmp_path / "other"
    directory = tmp_path / "store"
    directory.mkdir()
    own = _attached(directory, "thread-mine", str(mine), mtime=1_700_000_020)
    _attached(directory, "thread-other", str(other), mtime=1_700_000_010)
    empty = directory / "Tester_thread-empty.jsonl"
    empty.write_text("", encoding="utf-8")

    assert reader.file_workspace(own) == reader.workspace_key(mine)
    assert reader.file_workspace(empty) is None
    assert export.summarize_file(own).working_dir == str(mine)
    # A file with no event belongs to no workspace instead of failing the listing.
    listed = export.list_trajectories(directory, workspace=mine)
    assert [summary.path for summary in listed] == [own]
    assert export.find_trajectory_file(directory, "thread-other", workspace=mine) is None
    # "thread-" is ambiguous in the store but unique within the workspace.
    assert export.find_trajectory_file(directory, "thread-") is None
    assert export.find_trajectory_file(directory, "thread-", workspace=mine) == own


# ------------------------------------------------------- export / isolation


def test_export_reuses_reader_helpers() -> None:
    assert export.extract_message_text is reader.extract_message_text
    assert export.iter_events is reader.iter_events
    assert not hasattr(export, "_message_text")

    markdown = export.render_markdown(iter_events(FIXTURES / "normal_subagent.jsonl"))
    assert "# Trajectory: Profiler / thread-normal" in markdown
    assert "Final answer" in markdown
    assert "**Assistant (subagent)**" in markdown
    assert "grep: no matches" in markdown


_ISOLATION_PROBE = """
import sys
import msagent.trajectory_recorder.model
import msagent.trajectory_recorder.reader as reader
assert reader.__file__.startswith(sys.argv[1]), reader.__file__
leaked = sorted(m for m in sys.modules if m.startswith(("langchain", "langgraph")))
assert not leaked, leaked
"""


def test_reader_imports_no_langchain() -> None:
    # The pytest process already has langchain loaded (conftest imports the CLI
    # initializer), so the check runs in a fresh interpreter. PYTHONPATH must
    # win over the wheel that may be installed in the virtualenv.
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", _ISOLATION_PROBE, str(SRC_DIR)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
