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

"""Tests for the code-only episode detectors (hand-built trajectories + fixtures)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest

from msagent.skill_evolver import features as features_mod
from msagent.skill_evolver.features import (
    DEFAULT_MIN_EVIDENCE_SCORE,
    DROP_CHAIN_LIMIT,
    DROP_EMPTY_RESULT,
    DROP_ERROR_IN_OUTPUT,
    DROP_MISSING_START,
    DROP_NO_COMPLETED_CALLS,
    DROP_NO_TASK,
    EPISODE_KINDS,
    EPISODE_WEIGHTS,
    ERROR_MARKERS,
    FEATURES_VERSION,
    GATE_DEMO_NOT_APPLICABLE,
    GATE_DEMO_OVERRIDE,
    GATE_NO_EPISODES,
    GATE_SCORE_BELOW,
    GATE_SCORE_REACHED,
    GATE_STRONG_CORRECTION,
    OBSERVED_MAX_CALLS,
    OBSERVED_MAX_CHAINS,
    REQUIRED_ROLES,
    WEAK_CORRECTION_WEIGHT,
    DetectorNote,
    Episode,
    EpisodeKind,
    EvidenceItem,
    classify_approval,
    evidence_score,
    expand_episode_context,
    extract_episodes,
    gate_decision,
    group_incidents,
    has_required_evidence,
    mine_cross_session,
)
from msagent.skill_evolver.retrieval import BM25Index, SkillDoc
from msagent.trajectory_recorder.model import (
    PRELUDE_RUN_ID,
    AiMessage,
    Approval,
    EvidenceRef,
    ToolCall,
    ToolStatus,
    Trajectory,
    Turn,
)
from msagent.trajectory_recorder.reader import iter_events, load_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "trajectories"
FIXTURE_FILES = sorted(FIXTURES.glob("*.jsonl"))
FIXTURE_IDS = [path.name for path in FIXTURE_FILES]

DOCS = [
    SkillDoc("cluster-analysis", "Run the clustering workflow over profiler data"),
    SkillDoc("dit-quant", "Quantize DiT diffusion models with int8 calibration"),
    SkillDoc("ep-parallel", "Adapt expert parallel models for msmodelslim"),
    SkillDoc(
        "profiling-bottleneck",
        "Profile a training run with msprof and locate the kernel bottleneck",
    ),
]


def _index() -> BM25Index:
    return BM25Index(DOCS)


# ------------------------------------------------------------------ builders


def _call(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    seq: int,
    status: ToolStatus = "ok",
    error_type: str | None = None,
    error: str | None = None,
    subagent: str | None = None,
    output: str = "",
    seq_end: int | None = -1,
) -> ToolCall:
    """A tool span starting at ``seq``; by default its result (if any) sits at ``seq + 1``.

    ``seq_end=-1`` (the default) derives the end from the status; pass an
    explicit value to model a start-less call (``seq_end == seq``). Physical
    lines equal seqs, as in every single-writer fixture.
    """
    if seq_end == -1:
        seq_end = None if status == "orphan" else seq + 1
    return ToolCall(
        span_id=f"s{seq}",
        parent_span_id=None,
        name=name,
        args={} if args is None else args,
        status=status,
        output_text=output,
        error_type=error_type,
        error=error,
        duration_ms=1,
        seq_start=seq,
        seq_end=seq_end,
        line_start=seq,
        line_end=seq_end,
        subagent=subagent,
    )


def _turn(
    run_id: str,
    seq: int,
    message: str | None,
    calls: Sequence[ToolCall] = (),
    *,
    ai: Sequence[AiMessage] = (),
    approvals: Sequence[Approval] = (),
    source: str = "dispatch",
) -> Turn:
    return Turn(
        run_id=run_id,
        seq_start=seq,
        line_start=seq,
        user_message=message,
        source=source,
        ai_messages=list(ai),
        tool_calls=list(calls),
        approvals=list(approvals),
        status="completed",
    )


def _ai(seq: int, text: str, tools: Sequence[str] = ()) -> AiMessage:
    return AiMessage(
        seq=seq,
        span_id=f"a{seq}",
        text=text,
        tool_call_names=list(tools),
        usage=None,
        duration_ms=None,
        subagent=None,
        line=seq,
    )


def _traj(
    *turns: Turn,
    thread_id: str = "thread-t",
    skills: Sequence[str] = (),
) -> Trajectory:
    return Trajectory(
        path=Path(f"{thread_id}.jsonl"),
        thread_id=thread_id,
        agent="Tester",
        model=None,
        working_dir="/w",
        started_at="2026-09-01T10:00:00.000+00:00",
        turns=list(turns),
        skills_consulted=list(skills),
    )


def _approval(seq: int, decision: Any, request: Any = None) -> Approval:
    return Approval(
        seq=seq,
        run_id=None,
        interrupt_id=f"int-{seq}",
        request=request,
        decision=decision,
        line=seq,
    )


def _kinds(episodes: list[Episode]) -> list[str]:
    return [episode.kind for episode in episodes]


def _ep(
    kind: str,
    seqs: Sequence[int],
    *,
    anchors: Sequence[str] = (),
    thread_id: str = "t",
    weight: float | None = None,
) -> Episode:
    """A hand-built episode citing ``seqs`` of ``<thread_id>.jsonl`` (line == seq)."""
    source = f"{thread_id}.jsonl"
    return Episode(
        kind=kind,  # type: ignore[arg-type]
        thread_id=thread_id,
        source=source,
        evidence=[EvidenceItem(EvidenceRef(source=source, line=seq, seq=seq), "event", True) for seq in seqs],
        tool_sequence=[],
        facts={},
        weight=EPISODE_WEIGHTS[kind] if weight is None else weight,
        anchors=list(anchors),
    )


def _roles(episode: Episode) -> list[tuple[str, int, bool]]:
    """``(role, seq, required)`` of every evidence item, in episode order."""
    return [(item.role, item.ref.seq, item.required) for item in episode.evidence]


def _seq_at(path: Path, line: int) -> int:
    """The ``seq`` written on physical ``line`` of ``path`` (read like the reader does)."""
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            if number == line:
                return json.loads(raw)["seq"]
    raise AssertionError(f"{path} has no line {line}")


def _msprof_chain(seq: int) -> list[ToolCall]:
    """Two failed msprof invocations fixed by a third: one incident, three episodes."""
    return [
        _call("bash", {"cmd": "msprof --collect train.py"}, seq=seq, status="error"),
        _call("bash", {"cmd": "msprof --application train.py"}, seq=seq + 2, status="error"),
        _call("bash", {"cmd": "msprof --application train.py --output ./prof"}, seq=seq + 4),
    ]


# ------------------------------------------------------------------- Episode


def _item(seq: int, *, source: str = "t.jsonl", required: bool = True) -> EvidenceItem:
    return EvidenceItem(EvidenceRef(source=source, line=seq, seq=seq), "event", required)


def test_episode_rejects_invalid_fields() -> None:
    def make(**overrides: Any) -> Episode:
        fields: dict[str, Any] = {
            "kind": "retry_loop",
            "thread_id": "t",
            "source": "t.jsonl",
            "evidence": [_item(1), _item(2, required=False)],
            "tool_sequence": ["bash"],
            "facts": {},
            "weight": 0.7,
        }
        fields.update(overrides)
        return Episode(**fields)

    assert make().weight == 0.7
    assert make().anchors == []
    assert make(anchors=["run-1#1"]).anchors == ["run-1#1"]
    for overrides, message in (
        ({"evidence": []}, "without evidence"),
        ({"evidence": [_item(3), _item(3)]}, "duplicate evidence"),
        ({"evidence": [_item(1, required=False)]}, "without required evidence"),
        ({"evidence": [_item(1, source="other.jsonl")]}, "cites no event of 't.jsonl'"),
        ({"evidence": [1, 2]}, "invalid evidence"),
        ({"source": ""}, "without source"),
        ({"weight": -0.1}, "out of range"),
        ({"weight": 1.1}, "out of range"),
        ({"kind": "bogus"}, "unknown episode kind"),
        ({"thread_id": ""}, "without thread_id"),
        ({"anchors": ["run-1#4", 4]}, "invalid anchors"),
        ({"anchors": [""]}, "invalid anchors"),
    ):
        with pytest.raises(ValueError, match=message):
            make(**overrides)


def test_evidence_seq_lists_own_source_seqs_sorted() -> None:
    episode = Episode(
        kind="repeated_procedure",
        thread_id="t",
        source="t.jsonl",
        evidence=[_item(8), _item(4), _item(4, source="other.jsonl"), _item(2, source="other.jsonl")],
        tool_sequence=["bash", "grep"],
        facts={},
        weight=1.0,
    )
    assert episode.evidence_seq == [4, 8]


# ------------------------------------------------------------ text windows


def test_clip_marks_the_cut() -> None:
    assert features_mod._clip("x" * 200) == "x" * 200
    clipped = features_mod._clip("x" * 300)
    assert (len(clipped), clipped[-1]) == (200, "…")
    assert features_mod._clip({"k": "v" * 400}).endswith("…")
    assert features_mod._clip(7) == 7 and features_mod._clip(None) is None


def test_clip_edges_keeps_head_and_tail() -> None:
    trace = "Traceback (most recent call last):\n" + "  frame\n" * 100 + "KeyError: 'device'"
    clipped = features_mod._clip_edges(trace)
    assert clipped.startswith("Traceback (most recent call last):")
    assert clipped.endswith("KeyError: 'device'")
    assert "…" in clipped and len(clipped) == 100 + 1 + 180
    assert features_mod._clip_edges("short") == "short"


def test_diff_window_keeps_changed_tail() -> None:
    old = "A" * 300 + "X" + "B" * 300
    new = "A" * 300 + "Y" + "B" * 300

    before, after = features_mod._diff_window(old, new)

    assert before == "…" + "A" * 60 + "X" + "B" * 60 + "…"
    assert after == "…" + "A" * 60 + "Y" + "B" * 60 + "…"
    assert features_mod._diff_window("a", "a --fix") == ("a", "a --fix")


def test_markers_report_positions_in_collapsed_text() -> None:
    found = features_mod._markers("x" * 10 + "\n\n  不，不对，  actually")
    assert found[0] == ("不对", 13)
    assert [marker for marker, _ in found] == ["不对", "actually"]
    assert features_mod._markers("不，不对")[:2] == [("不对", 2), ("不，", 0)]


def test_episode_weights_cover_all_kinds() -> None:
    assert set(EPISODE_WEIGHTS) == set(get_args(EpisodeKind))
    # Demo evidence never scores; every scoring kind weighs in (0, 1].
    assert EPISODE_WEIGHTS["observed_procedure"] == 0.0
    assert all(0.0 < weight <= 1.0 for kind, weight in EPISODE_WEIGHTS.items() if kind != "observed_procedure")
    assert 0.0 < WEAK_CORRECTION_WEIGHT < EPISODE_WEIGHTS["user_correction"]
    assert FEATURES_VERSION == 4


# ----------------------------------------------------------- error_recovery


def test_error_recovery_positive_reports_args_diff() -> None:
    failed = _call(
        "bash",
        {"cmd": "a", "shell": True},
        seq=4,
        status="error",
        error_type="ToolException",
        error="boom",
    )
    traj = _traj(
        _turn(
            "run-1",
            2,
            "go",
            [
                failed,
                _call("read_file", {"path": "x"}, seq=6),
                _call("bash", {"cmd": "a --fix", "timeout": 30}, seq=8),
            ],
        ),
    )

    (episode,) = extract_episodes(traj)

    assert episode.kind == "error_recovery"
    assert episode.weight == 0.6
    assert episode.thread_id == "thread-t"
    assert episode.evidence_seq == [2, 4, 5, 8, 9]
    assert episode.anchors == ["run-1#4", "run-1#8"]
    assert episode.tool_sequence == ["bash", "read_file", "bash"]
    assert episode.facts["tool"] == "bash"
    assert episode.facts["error_type"] == "ToolException"
    assert episode.facts["error"] == "boom"
    assert episode.facts["calls_between"] == 1
    assert episode.facts["subagent"] is None
    assert episode.facts["args_diff"] == {
        "added": {"timeout": 30},
        "removed": {"shell": True},
        "changed": {"cmd": {"old": "a", "new": "a --fix"}},
    }


def test_error_recovery_negative_cases() -> None:
    err = _call("bash", {"cmd": "a"}, seq=4, status="error")
    # No successful call of the same tool at all.
    assert extract_episodes(_traj(_turn("run-1", 2, "go", [err]))) == []
    # Identical arguments: a transient failure carries no knowledge.
    same = _traj(_turn("run-1", 2, "go", [err, _call("bash", {"cmd": "a"}, seq=6)]))
    assert extract_episodes(same) == []
    # A different tool succeeding is not a recovery.
    other = _traj(_turn("run-1", 2, "go", [err, _call("grep", {"cmd": "b"}, seq=6)]))
    assert extract_episodes(other) == []
    # An orphan span never recovers anything.
    orphan_call = _call("bash", {"cmd": "b"}, seq=6, status="orphan")
    assert extract_episodes(_traj(_turn("run-1", 2, "go", [err, orphan_call]))) == []


def test_error_recovery_window_is_five_calls() -> None:
    err = _call("bash", {"cmd": "a"}, seq=4, status="error")
    fillers = [_call(f"step{i}", {"n": i}, seq=10 + 2 * i) for i in range(5)]
    fixed = _call("bash", {"cmd": "b"}, seq=30)

    near = _traj(_turn("run-1", 2, "go", [err, *fillers[:4], fixed]))
    within = extract_episodes(near)
    assert _kinds(within) == ["error_recovery"]
    assert within[0].evidence_seq == [2, 4, 5, 30, 31]
    assert within[0].facts["calls_between"] == 4

    far = _traj(_turn("run-1", 2, "go", [err, *fillers, fixed]))
    assert extract_episodes(far) == []


def test_error_recovery_crosses_turns_and_pairs_every_failure() -> None:
    first = _call("bash", {"cmd": "a"}, seq=4, status="error")
    second = _call("bash", {"cmd": "b"}, seq=6, status="error")
    fixed = _call("bash", {"cmd": "c"}, seq=12)
    traj = _traj(
        _turn("run-1", 2, "go", [first, second]),
        _turn("run-2", 10, None, [fixed], source="resume"),
    )

    episodes = extract_episodes(traj)

    assert _kinds(episodes) == ["error_recovery", "error_recovery"]
    evidence = [episode.evidence_seq for episode in episodes]
    assert evidence == [[2, 4, 5, 12, 13], [2, 6, 7, 12, 13]]
    olds = [episode.facts["args_diff"]["changed"]["cmd"]["old"] for episode in episodes]
    assert olds == ["a", "b"]
    assert [episode.anchors for episode in episodes] == [
        ["run-1#4", "run-2#12"],
        ["run-1#6", "run-2#12"],
    ]


def test_error_recovery_clips_long_values() -> None:
    err = _call("bash", {"cmd": "x" * 500}, seq=4, status="error")
    fixed = _call("bash", {"cmd": "y" * 500}, seq=6)

    (episode,) = extract_episodes(_traj(_turn("run-1", 2, "go", [err, fixed])))

    changed = episode.facts["args_diff"]["changed"]["cmd"]
    assert (len(changed["old"]), len(changed["new"])) == (200, 200)
    assert changed["old"].endswith("…") and changed["new"].endswith("…")


def test_error_recovery_evidence_roles_and_error_from_output_text() -> None:
    # The error came back as a tool.result with status=error: its text lives
    # in output_text, and the required evidence is error + fixed call + result.
    trace = "Traceback\n" + "  at frame\n" * 40 + "FileNotFoundError: cfg.yml"
    failed = _call("read_file", {"path": "cfg.yml"}, seq=4, status="error", output=trace)
    fixed = _call("read_file", {"path": "conf/cfg.yml"}, seq=8, output="key: value")
    traj = _traj(_turn("run-1", 2, "read it", [failed, _call("ls", {}, seq=6), fixed]))

    (episode,) = extract_episodes(traj)

    assert _roles(episode) == [
        ("error", 5, True),
        ("result", 9, True),
        ("fixed_call", 8, True),
        ("failed_call", 4, False),
        ("task", 2, False),
    ]
    error_item = episode.evidence[0]
    assert error_item.ref == EvidenceRef(source="thread-t.jsonl", line=5, seq=5)
    assert error_item.snippet is not None and error_item.snippet.endswith("FileNotFoundError: cfg.yml")
    assert error_item.snippet.startswith("Traceback") and "…" in error_item.snippet
    assert episode.facts["error"] == error_item.snippet
    assert [item.snippet for item in episode.evidence[1:]] == [None, None, None, None]


def test_error_recovery_start_less_calls_cite_one_ref_each() -> None:
    failed = _call("bash", {}, seq=4, status="error", error="boom", seq_end=4)
    fixed = _call("bash", {"cmd": "make"}, seq=8, seq_end=8)

    (episode,) = extract_episodes(_traj(_turn("run-1", 2, "go", [failed, fixed])))

    assert _roles(episode) == [("error", 4, True), ("result", 8, True), ("task", 2, False)]
    assert episode.evidence_seq == [2, 4, 8]


def test_error_recovery_stays_inside_the_stream() -> None:
    err = _call("bash", {"cmd": "a"}, seq=4, status="error", subagent="tools:b1")
    # A success of another subagent, or of the root agent, fixes nothing.
    for other in ("tools:b2", None):
        fixed = _call("bash", {"cmd": "b"}, seq=6, subagent=other)
        assert extract_episodes(_traj(_turn("run-1", 2, "go", [err, fixed]))) == []
    # The same subagent succeeding is the recovery.
    fixed = _call("bash", {"cmd": "b"}, seq=6, subagent="tools:b1")
    (episode,) = extract_episodes(_traj(_turn("run-1", 2, "go", [err, fixed])))
    assert episode.kind == "error_recovery"
    assert episode.facts["subagent"] == "tools:b1"
    assert episode.anchors == ["run-1#4", "run-1#6"]

    # The window counts calls of the failing stream only: five subagent calls
    # in between do not push the root recovery out of it.
    root_err = _call("bash", {"cmd": "a"}, seq=4, status="error")
    fillers = [_call(f"step{i}", {"n": i}, seq=10 + 2 * i, subagent="tools:b1") for i in range(5)]
    root_fixed = _call("bash", {"cmd": "b"}, seq=30)
    (episode,) = extract_episodes(_traj(_turn("run-1", 2, "go", [root_err, *fillers, root_fixed])))
    assert episode.evidence_seq == [2, 4, 5, 30, 31]
    assert episode.tool_sequence == ["bash", "bash"]
    assert episode.facts["calls_between"] == 0


def test_error_recovery_continues_into_a_resume_turn_only() -> None:
    err = _call("bash", {"cmd": "a"}, seq=4, status="error")
    fixed = _call("bash", {"cmd": "b"}, seq=12)

    resumed = _traj(
        _turn("run-1", 2, "go", [err]),
        _turn("run-2", 10, None, [fixed], source="resume"),
    )
    (episode,) = extract_episodes(resumed)
    assert episode.kind == "error_recovery"
    assert episode.evidence_seq == [2, 4, 5, 12, 13]
    assert episode.anchors == ["run-1#4", "run-2#12"]

    # A new dispatch (or an unknown) turn is another execution context.
    for source, message in (("dispatch", "continue"), ("unknown", None)):
        apart = _traj(
            _turn("run-1", 2, "go", [err]),
            _turn("run-2", 10, message, [fixed], source=source),
        )
        assert extract_episodes(apart) == []


# ---------------------------------------------------------- user_correction


@pytest.mark.parametrize(
    ("message", "markers"),
    [
        ("不，不对 —— 首先看 summary", ["不对", "不，", "首先"]),
        ("No, you should have used grep", ["no, ", "should have"]),
        ("Actually, run pytest instead", ["instead", "actually"]),
        ("应该先检查日志", ["应该先"]),
    ],
)
def test_user_correction_positive(message: str, markers: list[str]) -> None:
    traj = _traj(
        _turn("run-1", 2, "do it", [_call("bash", {"cmd": "a"}, seq=4)]),
        _turn("run-2", 10, message, [_call("grep", {"pattern": "b"}, seq=12)]),
    )

    (episode,) = extract_episodes(traj)

    assert episode.kind == "user_correction"
    assert episode.weight == 0.9
    assert episode.evidence_seq == [2, 10]
    assert episode.anchors == ["run-2#10"]
    assert episode.tool_sequence == ["grep"]
    assert episode.facts == {
        "correction_text": message,
        "strength": "strong",
        "markers": markers,
        "tools_before": ["bash"],
        "tools_after": ["grep"],
        "changes": {"tools_added": ["grep"], "tools_removed": ["bash"], "args_changed": {}},
        "run_id_before": "run-1",
        "run_id_after": "run-2",
    }


def test_user_correction_negative_cases() -> None:
    before = _turn("run-1", 2, "do it", [_call("bash", {"cmd": "a"}, seq=4)])
    # Gratitude is not a correction even when the tools differ.
    thanks = _turn("run-2", 10, "谢谢，都正常", [_call("grep", {}, seq=12)])
    assert extract_episodes(_traj(before, thanks)) == []
    # A marker without a change of action is just grumbling.
    grumble = _turn("run-2", 10, "不，不对", [_call("bash", {"cmd": "a"}, seq=12)])
    assert extract_episodes(_traj(before, grumble)) == []
    # A resumed turn carries no user message and cannot correct anything.
    resume = _turn("run-2", 10, None, [_call("grep", {}, seq=12)], source="resume")
    assert extract_episodes(_traj(before, resume)) == []
    # The very first user turn has no predecessor; the prelude does not count.
    prelude = _turn(PRELUDE_RUN_ID, 1, None, [], source="prelude")
    first = _turn("run-1", 2, "不，不对", [_call("bash", {}, seq=4)])
    assert extract_episodes(_traj(first)) == []
    assert extract_episodes(_traj(prelude, first)) == []


def test_user_correction_markers_start_a_word_and_negations_open_the_message() -> None:
    before = _turn("run-1", 2, "看文件", [_call("bash", {"cmd": "ls"}, seq=4)])
    # "arduino," is not "no, ": in a space-delimited script a marker has to
    # start a word.
    inside = _turn(
        "run-2",
        10,
        "Check the arduino, then run the test",
        [_call("grep", {"pattern": "x"}, seq=12)],
    )
    assert extract_episodes(_traj(before, inside)) == []
    # Chinese has no such boundary, so a Han marker is found mid-sentence:
    # "这不对" does carry "不对".
    han = _turn("run-2", 10, "这不对，用 grep 搜索", [_call("grep", {"pattern": "x"}, seq=12)])
    (found,) = extract_episodes(_traj(before, han))
    assert found.facts["markers"] == ["不对"]
    # "如果不，就创建它" is an ordinary instruction: a negation corrects
    # only when it opens the message.
    middle = _turn(
        "run-2",
        10,
        "检查日志；如果不，就创建它",
        [_call("write_file", {"path": "log.txt"}, seq=12)],
    )
    assert extract_episodes(_traj(before, middle)) == []
    opening = _turn("run-2", 10, "不，看日志", [_call("grep", {"pattern": "x"}, seq=12)])

    (episode,) = extract_episodes(_traj(before, opening))

    assert episode.facts["markers"] == ["不，"]
    assert episode.facts["strength"] == "strong"


def test_user_correction_ignores_normalized_equal_arguments() -> None:
    # A path spelled differently plus a volatile key is the same action.
    first = _call("read_file", {"path": "./cfg/dev.yml", "limit": 10}, seq=4)
    second = _call("read_file", {"path": "cfg/dev.yml", "limit": 50}, seq=12)
    before = _turn("run-1", 2, "read it", [first])
    after = _turn("run-2", 10, "不，不对", [second])
    assert extract_episodes(_traj(before, after)) == []


def test_user_correction_clips_text_and_accepts_marker_anywhere() -> None:
    message = "x" * 600 + " no, do it instead"
    traj = _traj(
        _turn("run-1", 2, "do it", [_call("bash", {}, seq=4)]),
        _turn("run-2", 10, message, []),
    )

    (episode,) = extract_episodes(traj)

    # The phrase after the first 600 characters is what the window keeps.
    text = episode.facts["correction_text"]
    assert text.startswith("…") and text.endswith("no, do it instead")
    assert len(text) == 1 + 120 + len("instead")
    assert episode.facts["strength"] == "strong"
    assert episode.facts["tools_after"] == []
    assert episode.facts["changes"]["tools_removed"] == ["bash"]
    assert episode.tool_sequence == []
    correction = episode.evidence[0]
    assert (correction.role, correction.required, correction.snippet) == ("correction", True, text)
    assert _roles(episode) == [("correction", 10, True), ("corrected_turn", 2, False)]


def test_user_correction_cites_the_changed_calls_as_context() -> None:
    long_path = "a/" * 200 + "old.yml"
    before = _turn("run-1", 2, "read the config", [_call("read_file", {"path": long_path}, seq=4)])
    after = _turn("run-2", 10, "不，不对", [_call("read_file", {"path": long_path[:-7] + "new.yml"}, seq=12)])

    (episode,) = extract_episodes(_traj(before, after))

    assert _roles(episode) == [
        ("correction", 10, True),
        ("corrected_turn", 2, False),
        ("before_call", 4, False),
        ("after_call", 12, False),
    ]
    changed = episode.facts["changes"]["args_changed"]["read_file"]["changed"]["path"]
    assert changed == {"old": "…" + "a/" * 30 + "old.yml", "new": "…" + "a/" * 30 + "new.yml"}


def test_user_correction_records_a_path_change_of_the_same_tool() -> None:
    traj = _traj(
        _turn("run-1", 2, "read the config", [_call("read_file", {"path": "cfg/dev.yml"}, seq=4)]),
        _turn(
            "run-2",
            10,
            "不，不对：应该先读 cfg/prod.yml",
            [_call("read_file", {"path": "cfg/prod.yml"}, seq=12)],
        ),
    )

    (episode,) = extract_episodes(traj)

    assert episode.kind == "user_correction"
    assert episode.facts["strength"] == "strong"
    assert episode.facts["markers"] == ["不对", "不，", "应该先"]
    assert episode.facts["changes"] == {
        "tools_added": [],
        "tools_removed": [],
        "args_changed": {
            "read_file": {"changed": {"path": {"old": "cfg/dev.yml", "new": "cfg/prod.yml"}}},
        },
    }


def test_user_correction_weak_marker_is_never_strong() -> None:
    before = _turn("run-1", 2, "run it", [_call("bash", {"cmd": "a"}, seq=4)])
    message = "Actually, let's also run pytest"
    changed = _turn("run-2", 10, message, [_call("bash", {"cmd": "pytest"}, seq=12)])

    (episode,) = extract_episodes(_traj(before, changed))

    assert episode.kind == "user_correction"
    assert episode.facts["strength"] == "weak"
    assert episode.facts["markers"] == ["actually"]
    assert episode.weight == WEAK_CORRECTION_WEIGHT == 0.5
    assert episode.facts["changes"]["args_changed"] == {
        "bash": {"changed": {"cmd": {"old": "a", "new": "pytest"}}},
    }
    decision = gate_decision([episode], min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert (decision.passes, decision.reason) == (False, GATE_SCORE_BELOW)

    # A weak marker without a change of action is nothing at all.
    unchanged = _turn("run-2", 10, message, [_call("bash", {"cmd": "a"}, seq=12)])
    assert extract_episodes(_traj(before, unchanged)) == []


def test_user_correction_compares_turn_groups() -> None:
    traj = _traj(
        _turn("run-1", 2, "do it", [_call("bash", {"cmd": "a"}, seq=4)]),
        _turn("run-2", 10, None, [_call("read_file", {"path": "x"}, seq=12)], source="resume"),
        _turn(
            "run-3",
            20,
            "不，不对 —— 用 grep 搜索",
            [_call("grep", {"pattern": "b"}, seq=22)],
        ),
    )

    (episode,) = extract_episodes(traj)

    assert episode.kind == "user_correction"
    assert episode.evidence_seq == [2, 20]
    assert episode.anchors == ["run-3#20"]
    assert episode.facts["tools_before"] == ["bash", "read_file"]
    assert episode.facts["tools_after"] == ["grep"]
    assert episode.facts["changes"]["tools_removed"] == ["bash", "read_file"]
    assert episode.facts["run_id_before"] == "run-1"
    assert episode.facts["run_id_after"] == "run-3"


# ---------------------------------------------------------------- retry_loop


def test_retry_loop_positive() -> None:
    calls = [
        _call("grep", {"pattern": "hotspot", "path": "a.csv"}, seq=4),
        _call("read_file", {"path": "a.csv"}, seq=6),
        _call("grep", {"pattern": "hot_spot", "path": "a.csv"}, seq=8),
        _call("grep", {"pattern": "HotSpot", "path": "a.csv"}, seq=10),
    ]

    (episode,) = extract_episodes(_traj(_turn("run-1", 2, "find it", calls)))

    assert episode.kind == "retry_loop"
    assert episode.weight == 0.7
    assert episode.evidence_seq == [2, 4, 5, 8, 9, 10, 11]
    assert episode.anchors == ["run-1#4", "run-1#8", "run-1#10"]
    assert episode.tool_sequence == ["grep", "grep", "grep"]
    assert episode.facts["tool_name"] == "grep"
    assert episode.facts["work_object"] == "a.csv"
    assert episode.facts["reason"] == "search key varies"
    assert episode.facts["attempts"] == 3
    assert episode.facts["statuses"] == ["ok", "ok", "ok"]
    assert episode.facts["run_id"] == "run-1"
    variants = [variant["pattern"] for variant in episode.facts["args_variants"]]
    assert variants == ["hotspot", "hot_spot", "HotSpot"]
    # The first and last attempt and their results are required; the
    # attempt between them and the group's user message are context.
    assert _roles(episode) == [
        ("attempt", 4, True),
        ("result", 5, True),
        ("attempt", 8, False),
        ("result", 9, False),
        ("attempt", 10, True),
        ("result", 11, True),
        ("task", 2, False),
    ]
    assert episode.facts["outcome"] == ""  # the builder records no output


def test_retry_loop_negative_cases() -> None:
    # Two calls are not a loop.
    two = [_call("bash", {"cmd": "a"}, seq=4), _call("bash", {"cmd": "b"}, seq=6)]
    assert extract_episodes(_traj(_turn("run-1", 2, "go", two))) == []
    # Three identical calls differ in no value; so do calls without arguments.
    same = [_call("bash", {"cmd": "a"}, seq=seq) for seq in (4, 6, 8)]
    assert extract_episodes(_traj(_turn("run-1", 2, "go", same))) == []
    empty = [_call("bash", {}, seq=seq) for seq in (4, 6, 8)]
    assert extract_episodes(_traj(_turn("run-1", 2, "go", empty))) == []
    # Different work objects (commands a and c, path b) never chain.
    mixed = [
        _call("bash", {"cmd": "a"}, seq=4),
        _call("bash", {"path": "b"}, seq=6),
        _call("bash", {"cmd": "c"}, seq=8),
    ]
    assert extract_episodes(_traj(_turn("run-1", 2, "go", mixed))) == []
    # Attempts spread over two turns do not add up.
    split = _traj(
        _turn("run-1", 2, "go", two),
        _turn("run-2", 10, "more", [_call("bash", {"cmd": "c"}, seq=12)]),
    )
    assert extract_episodes(split) == []
    # A parameter sweep that never failed is not a loop.
    sweep = [
        _call("bash", {"cmd": "python train.py --lr 0.1"}, seq=4),
        _call("bash", {"cmd": "python train.py --lr 0.01"}, seq=6),
        _call("bash", {"cmd": "python train.py --lr 0.001"}, seq=8),
    ]
    assert extract_episodes(_traj(_turn("run-1", 2, "go", sweep))) == []
    # Paging through one file differs only in a volatile key: one variant.
    paging = [
        _call("read_file", {"path": "big.bin", "offset": 0}, seq=4),
        _call("read_file", {"path": "big.bin", "offset": 100}, seq=6),
        _call("read_file", {"path": "big.bin", "offset": 200}, seq=8),
    ]
    assert extract_episodes(_traj(_turn("run-1", 2, "go", paging))) == []
    # The third attempt made by another subagent belongs to another stream.
    apart = [
        _call("bash", {"cmd": "pip install torch"}, seq=4, status="error"),
        _call("bash", {"cmd": "pip install torch==2.1"}, seq=6, status="error"),
        _call("bash", {"cmd": "pip install torch==2.2"}, seq=8, subagent="tools:b1"),
    ]
    assert extract_episodes(_traj(_turn("run-1", 2, "go", apart))) == []
    # Three interrupted variants without a failure or a search key: no loop.
    orphans = [
        _call("bash", {"cmd": f"pip install torch=={v}"}, seq=4 + 2 * i, status="orphan")
        for i, v in enumerate(("2.0", "2.1", "2.2"))
    ]
    assert extract_episodes(_traj(_turn("run-1", 2, "go", orphans))) == []


def test_retry_loop_work_object_is_the_command_not_its_directory() -> None:
    # Three unrelated commands in one working directory are not one operation.
    unrelated = [
        _call("bash", {"cmd": "pytest tests/a", "cwd": "/repo"}, seq=4, status="error"),
        _call("bash", {"cmd": "make docs", "cwd": "/repo"}, seq=6),
        _call("bash", {"cmd": "git status", "cwd": "/repo"}, seq=8),
    ]
    assert _kinds(extract_episodes(_traj(_turn("run-1", 2, "go", unrelated)))) == ["error_recovery"]
    # The same command retried with options is one operation.
    attempts = [
        _call("bash", {"cmd": "pytest tests/a", "cwd": "/repo"}, seq=4, status="error"),
        _call("bash", {"cmd": "pytest tests/a -x", "cwd": "/repo"}, seq=6, status="error"),
        _call("bash", {"cmd": "pytest tests/a -x -k smoke", "cwd": "/repo"}, seq=8, status="error"),
    ]

    (episode,) = extract_episodes(_traj(_turn("run-1", 2, "go", attempts)))

    assert episode.kind == "retry_loop"
    assert episode.facts["work_object"] == "pytest tests/a"


def test_retry_loop_fanout_over_different_files_is_not_a_retry() -> None:
    # Reading three files is three work objects, whatever their key sets.
    calls = [
        _call("read_file", {"path": "c.py"}, seq=4),
        _call("bash", {"cmd": "ls"}, seq=6),
        _call("read_file", {"path": "d.py"}, seq=8),
        _call("read_file", {"path": "e.py"}, seq=10, status="orphan"),
    ]
    assert extract_episodes(_traj(_turn("run-1", 2, "read", calls))) == []


def test_retry_loop_counts_orphans_as_attempts() -> None:
    calls = [
        _call("bash", {"cmd": "pip install torch"}, seq=4, status="error"),
        _call("bash", {"cmd": "pip  install torch==2.1"}, seq=6, status="orphan"),
        _call("bash", {"cmd": "pip install torch --index-url https://x"}, seq=8, status="error"),
    ]

    (episode,) = extract_episodes(_traj(_turn("run-1", 2, "install", calls)))

    assert episode.kind == "retry_loop"
    # The orphan at 6 is an attempt without an end; the errors at 5 and 9 are
    # the first and last outcome.
    assert episode.evidence_seq == [2, 4, 5, 6, 8, 9]
    assert _roles(episode) == [
        ("attempt", 4, True),
        ("error", 5, True),
        ("attempt", 6, False),
        ("attempt", 8, True),
        ("error", 9, True),
        ("task", 2, False),
    ]
    assert episode.facts["outcome"] == ""
    assert episode.anchors == ["run-1#4", "run-1#6", "run-1#8"]
    assert episode.facts["work_object"] == "pip install"
    assert episode.facts["reason"] == "failed attempt"
    assert episode.facts["statuses"] == ["error", "orphan", "error"]
    assert episode.facts["args_variants"] == [
        {"cmd": "pip install torch"},
        {"cmd": "pip install torch==2.1"},
        {"cmd": "pip install torch --index-url https://x"},
    ]


def test_retry_loop_continues_into_a_resume_turn_only() -> None:
    first = [
        _call("grep", {"pattern": "a", "path": "f"}, seq=4),
        _call("grep", {"pattern": "b", "path": "f"}, seq=6),
    ]
    third = _call("grep", {"pattern": "c", "path": "f"}, seq=12)

    resumed = _traj(
        _turn("run-1", 2, "find", first),
        _turn("run-2", 10, None, [third], source="resume"),
    )
    (episode,) = extract_episodes(resumed)
    assert episode.kind == "retry_loop"
    assert episode.evidence_seq == [2, 4, 5, 6, 7, 12, 13]
    assert episode.anchors == ["run-1#4", "run-1#6", "run-2#12"]
    assert episode.facts["run_id"] == "run-1"
    assert episode.facts["reason"] == "search key varies"

    for source, message in (("dispatch", "keep looking"), ("unknown", None)):
        apart = _traj(
            _turn("run-1", 2, "find", first),
            _turn("run-2", 10, message, [third], source=source),
        )
        assert extract_episodes(apart) == []


# ----------------------------------------------------------- approval_denied

_TWO_ACTIONS = {
    "action_requests": [
        {"name": "bash", "args": {"cmd": "ls"}},
        {"name": "write_file", "args": {"content": "x" * 300}},
    ],
    "review_configs": [],
}
_ONE_ACTION = {"action_requests": [{"name": "bash", "args": {"cmd": "ls"}}]}
_FLAT_REQUEST = {"tool": "bash", "args": {"cmd": "rm"}}


@pytest.mark.parametrize(
    ("decision", "interrupt", "status", "denied_names"),
    [
        # Structured HITL decisions, matched by index against action_requests.
        ({"decisions": [{"type": "approve", "message": "no issues"}]}, _ONE_ACTION, "approved", []),
        (
            {
                "decisions": [
                    {"type": "approve"},
                    {"type": "reject", "message": "Rejected by policy."},
                ],
            },
            _TWO_ACTIONS,
            "denied",
            ["write_file"],
        ),
        (
            {"decisions": [{"type": "reject"}, {"type": "reject"}]},
            _TWO_ACTIONS,
            "denied",
            ["bash", "write_file"],
        ),
        ({"decisions": [{"type": "reject"}]}, _TWO_ACTIONS, "unknown", []),
        ({"decisions": [{"type": "reject"}, {"type": "reject"}]}, None, "unknown", []),
        ({"decisions": [{"type": "reject"}]}, None, "denied", [None]),
        ({"decisions": [{"type": "reject"}]}, _FLAT_REQUEST, "denied", ["bash"]),
        ({"decisions": []}, _ONE_ACTION, "unknown", []),
        ({"decisions": "reject"}, _ONE_ACTION, "unknown", []),
        ({"decisions": ["reject"]}, _ONE_ACTION, "unknown", []),
        ({"decisions": [{"type": "veto"}]}, _ONE_ACTION, "unknown", []),
        (
            {"decisions": [{"type": "edit", "edited_action": {"args": {"cmd": "ls -a"}}}]},
            _ONE_ACTION,
            "approved",
            [],
        ),
        ({"decisions": [{"type": "respond", "message": "later"}]}, _ONE_ACTION, "approved", []),
        # Flat decisions apply to the flat request.
        ({"action": "approve", "comment": "no issues"}, {"tool": "bash"}, "approved", []),
        ({"action": "reject"}, _FLAT_REQUEST, "denied", ["bash"]),
        ({"action": "Reject"}, {"question": "Proceed?"}, "denied", [None]),
        (
            {"type": "edit", "edited_action": {"args": {"cmd": "echo note"}}},
            {"tool": "bash"},
            "approved",
            [],
        ),
        ({"action": "veto"}, {"tool": "bash"}, "unknown", []),
        ({"comment": "no"}, {"tool": "bash"}, "unknown", []),
        # Legacy option answers must match exactly; free text is unknown.
        ("no", {"tool": "bash"}, "denied", ["bash"]),
        ("Denied", {"tool": "bash"}, "denied", ["bash"]),
        ("deny", {"tool": "bash"}, "denied", ["bash"]),
        ("rejected", None, "denied", [None]),
        ("yes", {"tool": "bash"}, "approved", []),
        (" OK ", {"tool": "bash"}, "approved", []),
        ("No, don't", {"tool": "bash"}, "unknown", []),
        ("Cancel", {"tool": "bash"}, "unknown", []),
        ("note", {"tool": "bash"}, "unknown", []),
        ("no_changes", {"tool": "bash"}, "unknown", []),
        # Anything else cannot be read.
        (None, {"tool": "bash"}, "unknown", []),
        (42, {"tool": "bash"}, "unknown", []),
        (["reject"], {"tool": "bash"}, "unknown", []),
    ],
)
def test_classify_approval(
    decision: Any,
    interrupt: Any,
    status: str,
    denied_names: list[str | None],
) -> None:
    verdict = classify_approval(interrupt, decision)

    assert verdict.status == status
    assert [action["name"] for action in verdict.denied] == denied_names
    assert all(isinstance(action["args"], dict) for action in verdict.denied)
    assert isinstance(verdict.reason, str) and verdict.reason


def test_classify_approval_reports_the_shape_problem() -> None:
    mismatch = classify_approval(_TWO_ACTIONS, {"decisions": [{"type": "reject"}]})
    assert mismatch.reason == "1 decisions for 2 action_requests"
    assert classify_approval(_ONE_ACTION, {"decisions": []}).reason == "no decision recorded"
    veto = classify_approval(_ONE_ACTION, {"decisions": [{"type": "veto"}]})
    assert veto.reason == "decision 0 has type 'veto'"
    # Denied arguments are clipped like every fact value.
    decision = {"decisions": [{"type": "approve"}, {"type": "reject"}]}
    verdict = classify_approval(_TWO_ACTIONS, decision)
    assert verdict.denied == [{"name": "write_file", "args": {"content": "x" * 199 + "…"}}]


def test_approval_denied_positive_with_real_hitl_shape() -> None:
    decision = {"decisions": [{"type": "approve"}, {"type": "reject"}]}
    following = [
        _call("bash", {"cmd": "cat x"}, seq=12),
        _call("ls", {"path": "."}, seq=14),
    ]
    traj = _traj(
        _turn(
            "run-1",
            2,
            "write it",
            [_call("grep", {"pattern": "a"}, seq=4)],
            approvals=[_approval(9, decision, _TWO_ACTIONS)],
        ),
        _turn("run-2", 10, None, following, source="resume"),
    )

    (episode,) = extract_episodes(traj)

    assert episode.kind == "approval_denied"
    assert episode.weight == 1.0
    assert episode.evidence_seq == [9, 12, 14]
    assert episode.anchors == ["run-1#9"]
    assert episode.tool_sequence == ["bash", "ls"]
    assert episode.facts["interrupt_id"] == "int-9"
    assert episode.facts["run_id"] == "run-1"
    assert episode.facts["tools"] == ["write_file"]
    assert episode.facts["denied_actions"] == [
        {"name": "write_file", "args": {"content": "x" * 199 + "…"}},
    ]
    assert _roles(episode) == [("approval", 9, True), ("next_call", 12, False), ("next_call", 14, False)]
    assert episode.facts["decision"] == decision
    assert len(episode.facts["request"]) == 200
    assert episode.facts["next_tools"] == [
        {"name": "bash", "args": {"cmd": "cat x"}, "status": "ok"},
        {"name": "ls", "args": {"path": "."}, "status": "ok"},
    ]


def test_approval_denied_negative_and_context_rules() -> None:
    # "no issues" in an approval message is not a denial.
    approve = _approval(
        9,
        {"decisions": [{"type": "approve", "message": "no issues"}]},
        {"tool": "bash"},
    )
    before = _call("grep", {}, seq=4)
    approved = _traj(_turn("run-1", 2, "go", [before], approvals=[approve]))
    assert extract_episodes(approved) == []

    denied = _approval(9, {"action": "reject"}, _FLAT_REQUEST)
    # Calls before the approval are excluded; later ones of the same turn
    # group (the resume turn) are included and capped at three.
    later = [_call(f"step{i}", {}, seq=20 + 2 * i) for i in range(4)]
    same_turn = _call("ls", {}, seq=11)
    traj = _traj(
        _turn("run-1", 2, "go", [before, same_turn], approvals=[denied]),
        _turn("run-2", 18, None, later, source="resume"),
    )

    (episode,) = extract_episodes(traj)

    assert episode.evidence_seq == [9, 11, 20, 22]
    assert episode.tool_sequence == ["ls", "step0", "step1"]
    assert episode.facts["tools"] == ["bash"]
    assert episode.facts["denied_actions"] == [{"name": "bash", "args": {"cmd": "rm"}}]

    # A following dispatch turn is not the reaction to the denial.
    dispatched = _traj(
        _turn("run-1", 2, "go", [before, same_turn], approvals=[denied]),
        _turn("run-2", 18, "next", later),
    )
    (episode,) = extract_episodes(dispatched)
    assert (episode.evidence_seq, episode.tool_sequence) == ([9, 11], ["ls"])

    # Nothing after the denial and an unknown request shape: the approval
    # alone is the evidence and no tool can be named.
    for request in (None, {"question": "Proceed?"}):
        alone = _approval(9, "no", request)
        turn = _turn("run-1", 2, "go", [], approvals=[alone])
        (episode,) = extract_episodes(_traj(turn))
        assert (episode.evidence_seq, episode.tool_sequence) == ([9], [])
        assert episode.facts["tools"] == []
        assert episode.facts["denied_actions"] == [{"name": None, "args": {}}]


def test_approval_unknown_formats_yield_no_episode() -> None:
    following = [_call("ls", {}, seq=11)]
    for decision, request in (
        ({"decisions": [{"type": "reject"}]}, _TWO_ACTIONS),
        ({"decisions": [{"type": "veto"}]}, _ONE_ACTION),
        ("Cancel", {"tool": "bash"}),
        ("No, don't", {"tool": "bash"}),
        (None, {"tool": "bash"}),
    ):
        turn = _turn("run-1", 2, "go", following, approvals=[_approval(9, decision, request)])
        assert extract_episodes(_traj(turn)) == []


# ----------------------------------------------------------------- skill_gap


def test_skill_gap_positive() -> None:
    calls = [_call("fetch_skills", {}, seq=4), _call("bash", {"cmd": "q"}, seq=6)]
    message = "Quantize the DiT model with int8 calibration"
    traj = _traj(_turn("run-1", 2, message, calls))

    (episode,) = extract_episodes(traj, skill_index=_index())

    assert episode.kind == "skill_gap"
    assert episode.weight == 0.4
    assert episode.evidence_seq == [2, 6]
    assert episode.anchors == []
    assert episode.tool_sequence == ["bash"]
    assert episode.facts["candidate_skill"] == "dit-quant"
    assert episode.facts["score"] >= 1.0
    assert episode.facts["matched_terms"] == ["calibration", "dit", "int8", "quantize"]
    assert episode.facts["domain_tools"] == ["bash"]
    assert _roles(episode) == [("first_call", 6, True), ("user_message", 2, True)]


def test_skill_gap_negative_cases() -> None:
    message = "Quantize the DiT model with int8 calibration"
    work = [_call("bash", {"cmd": "q"}, seq=4)]
    traj = _traj(_turn("run-1", 2, message, work))
    # No index or an empty index: the detector is off.
    assert extract_episodes(traj) == []
    assert extract_episodes(traj, skill_index=BM25Index([])) == []
    # A consulted skill means no gap.
    consulted = _traj(_turn("run-1", 2, message, work), skills=["dit-quant"])
    assert extract_episodes(consulted, skill_index=_index()) == []
    # Only catalog tools were used: no domain work happened.
    catalog = _traj(_turn("run-1", 2, message, [_call("fetch_skills", {}, seq=4)]))
    assert extract_episodes(catalog, skill_index=_index()) == []
    # Nothing in the library resembles the session, or only weakly.
    unrelated = _traj(_turn("run-1", 2, "Fix the flaky login test", work))
    assert extract_episodes(unrelated, skill_index=_index()) == []
    weak = _traj(_turn("run-1", 2, "Compare the models", work))
    assert extract_episodes(weak, skill_index=_index()) == []


# ------------------------------------------------------- mine_cross_session


def _procedure(thread_id: str, names: Sequence[str], *, seq: int = 4) -> Trajectory:
    calls = [_call(name, {"i": i}, seq=seq + 2 * i) for i, name in enumerate(names)]
    return _traj(_turn("run-1", 2, "go", calls), thread_id=thread_id)


def test_mine_cross_session_reports_closed_patterns() -> None:
    trajs = [
        _procedure("A", ["bash", "read_file", "grep"]),
        _procedure("B", ["bash", "read_file", "grep"], seq=20),
        _procedure("C", ["ls"]),
    ]

    (episode,) = mine_cross_session(trajs)

    assert episode.kind == "repeated_procedure"
    assert episode.weight == 1.0
    assert episode.thread_id == "A"
    assert episode.evidence_seq == [2, 4, 5, 6, 7, 8, 9]
    assert episode.anchors == ["run-1#4", "run-1#6", "run-1#8"]
    assert episode.tool_sequence == ["bash", "read_file", "grep"]
    assert episode.facts == {
        "ngram": ["bash", "read_file", "grep"],
        "support": 2,
        "thread_ids": ["A", "B"],
    }
    # Steps of both sessions are required evidence: the proof is the repetition.
    # The owner's results and the user message of its turn group are context.
    assert [(item.ref.source, item.role, item.ref.seq, item.required) for item in episode.evidence] == [
        ("A.jsonl", "step", 4, True),
        ("A.jsonl", "result", 5, False),
        ("A.jsonl", "step", 6, True),
        ("A.jsonl", "result", 7, False),
        ("A.jsonl", "step", 8, True),
        ("A.jsonl", "result", 9, False),
        ("B.jsonl", "step", 20, True),
        ("B.jsonl", "step", 22, True),
        ("B.jsonl", "step", 24, True),
        ("A.jsonl", "task", 2, False),
    ]
    assert episode.source == "A.jsonl"


def test_mine_cross_session_negative_cases() -> None:
    # Repetition inside one trajectory is not evidence.
    solo = _procedure("A", ["bash", "grep"] * 3)
    assert mine_cross_session([solo]) == []
    # The same thread recorded twice counts once.
    twin = _procedure("A", ["bash", "grep"], seq=30)
    assert mine_cross_session([solo, twin]) == []
    # Support below min_support, and min_support itself must be at least 2.
    trajs = [_procedure("A", ["bash", "grep"]), _procedure("B", ["bash", "grep"])]
    assert mine_cross_session(trajs, min_support=3) == []
    for bad in (1, 0):
        with pytest.raises(ValueError, match="min_support"):
            mine_cross_session(trajs, min_support=bad)


def test_mine_cross_session_subgram_with_larger_support_survives() -> None:
    trajs = [
        _procedure("A", ["bash", "read_file", "grep"]),
        _procedure("B", ["bash", "read_file", "grep"]),
        _procedure("C", ["bash", "read_file", "ls"]),
    ]

    episodes = mine_cross_session(trajs)

    assert [(e.tool_sequence, e.facts["support"]) for e in episodes] == [
        (["bash", "read_file"], 3),
        (["bash", "read_file", "grep"], 2),
    ]


def test_mine_cross_session_caps_ngrams_at_five() -> None:
    names = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
    trajs = [_procedure("A", names), _procedure("B", names)]

    episodes = mine_cross_session(trajs)

    assert [e.tool_sequence for e in episodes] == [names[0:5], names[1:6], names[2:7]]
    assert all(e.facts["support"] == 2 for e in episodes)


def test_mine_cross_session_ignores_catalog_calls() -> None:
    mixed = _procedure("A", ["get_skill", "bash", "fetch_tools", "grep", "run_tool"])
    plain = _procedure("B", ["bash", "grep"])

    (episode,) = mine_cross_session([mixed, plain])

    assert episode.thread_id == "A"
    assert episode.tool_sequence == ["bash", "grep"]
    assert episode.evidence_seq == [2, 6, 7, 10, 11]
    assert episode.anchors == ["run-1#6", "run-1#10"]
    assert episode.facts["thread_ids"] == ["A", "B"]
    # Catalog calls alone are not a procedure.
    catalog = [
        _procedure("A", ["get_skill", "fetch_tools"]),
        _procedure("B", ["get_skill", "fetch_tools"]),
    ]
    assert mine_cross_session(catalog) == []


def test_mine_cross_session_splits_at_a_failed_call() -> None:
    plain = _procedure("B", ["bash", "read_file", "grep"])
    for status in ("error", "orphan"):
        broken = _traj(
            _turn(
                "run-1",
                2,
                "go",
                [
                    _call("bash", {"cmd": "a"}, seq=4),
                    _call("read_file", {"path": "x"}, seq=6, status=status),
                    _call("grep", {"pattern": "p"}, seq=8),
                ],
            ),
            thread_id="A",
        )
        assert mine_cross_session([broken, plain]) == []
    whole = _traj(
        _turn(
            "run-1",
            2,
            "go",
            [
                _call("bash", {"cmd": "a"}, seq=4),
                _call("read_file", {"path": "x"}, seq=6),
                _call("grep", {"pattern": "p"}, seq=8),
            ],
        ),
        thread_id="A",
    )

    (episode,) = mine_cross_session([whole, plain])

    assert episode.tool_sequence == ["bash", "read_file", "grep"]
    assert episode.evidence_seq == [2, 4, 5, 6, 7, 8, 9]


def test_mine_cross_session_keeps_streams_apart() -> None:
    plain = _procedure("B", ["bash", "grep"])
    first = _call("bash", {"cmd": "a"}, seq=4)
    second = _call("grep", {"pattern": "p"}, seq=12)
    # A resume turn continues the sequence of the turn it resumes.
    resumed = _traj(
        _turn("run-1", 2, "go", [first]),
        _turn("run-2", 10, None, [second], source="resume"),
    )
    (episode,) = mine_cross_session([resumed, plain])
    assert episode.tool_sequence == ["bash", "grep"]
    assert episode.evidence_seq == [2, 4, 5, 12, 13]
    assert episode.anchors == ["run-1#4", "run-2#12"]
    # A dispatch turn starts another sequence.
    dispatched = _traj(_turn("run-1", 2, "go", [first]), _turn("run-2", 10, "more", [second]))
    assert mine_cross_session([dispatched, plain]) == []
    # A step made by another subagent is not a step of this stream ...
    elsewhere = _call("grep", {"pattern": "p"}, seq=6, subagent="tools:b1")
    apart = _traj(_turn("run-1", 2, "go", [first, elsewhere]))
    assert mine_cross_session([apart, plain]) == []
    # ... and does not interrupt it either.
    aside = _call("ls", {}, seq=6, subagent="tools:b1")
    around = _turn("run-1", 2, "go", [first, aside, _call("grep", {}, seq=8)])
    (episode,) = mine_cross_session([_traj(around), plain])
    assert episode.evidence_seq == [2, 4, 5, 8, 9]


# ------------------------------------------------------------------- scoring


def test_evidence_score_adds_independent_incidents() -> None:
    assert evidence_score([]) == 0.0
    signals = load_trajectory(FIXTURES / "skill_evolver_signals.jsonl")
    episodes = extract_episodes(signals)
    # error_recovery x2 + retry_loop describe one msprof chain: 0.7, not 1.9.
    assert len(group_incidents(episodes)) == 3
    assert evidence_score(episodes) == pytest.approx(2.6)


# ------------------------------------------------------------------ fixtures


def test_signals_fixture_end_to_end() -> None:
    traj = load_trajectory(FIXTURES / "skill_evolver_signals.jsonl")

    episodes = extract_episodes(traj, skill_index=_index())

    assert [(e.kind, e.evidence_seq) for e in episodes] == [
        ("error_recovery", [2, 4, 5, 10, 11]),
        ("error_recovery", [2, 7, 8, 10, 11]),
        ("user_correction", [2, 10, 17, 19]),
        ("retry_loop", [2, 4, 5, 7, 8, 10, 11]),
        ("approval_denied", [26, 29, 32]),
        ("skill_gap", [2, 4, 17]),
    ]
    recovery, second, correction, retry, denial, gap = episodes
    changed = recovery.facts["args_diff"]["changed"]["cmd"]
    assert changed["old"] == "msprof --collect train.py"
    assert changed["new"] == "msprof --application train.py --output ./prof"
    assert recovery.anchors == ["run-1#4", "run-1#10"]
    assert second.anchors == ["run-1#7", "run-1#10"]
    assert correction.facts["strength"] == "strong"
    assert correction.facts["markers"] == ["不对", "不，", "首先"]
    assert correction.facts["tools_before"] == ["bash", "bash", "bash", "read_file"]
    # The group of run-2 includes the resume turn run-3.
    assert correction.facts["tools_after"] == ["bash", "grep", "bash", "ls"]
    assert correction.facts["changes"]["tools_added"] == ["grep", "ls"]
    assert correction.facts["changes"]["tools_removed"] == ["read_file"]
    new_cmd = correction.facts["changes"]["args_changed"]["bash"]["changed"]["cmd"]["new"]
    assert new_cmd.endswith("--level kernel")
    assert correction.anchors == ["run-2#17"]
    assert retry.facts["attempts"] == 3
    assert retry.facts["work_object"] == "msprof train.py"
    assert retry.facts["reason"] == "failed attempt"
    assert retry.facts["statuses"] == ["error", "error", "ok"]
    assert retry.anchors == ["run-1#4", "run-1#7", "run-1#10"]
    assert denial.facts["tools"] == ["write_file"]
    assert denial.facts["denied_actions"] == [
        {"name": "write_file", "args": {"path": "report.md", "content": "# Report"}},
    ]
    assert denial.tool_sequence == ["bash", "ls"]
    assert denial.anchors == ["run-2#26"]
    assert gap.facts["candidate_skill"] == "profiling-bottleneck"
    assert gap.anchors == []
    assert [_kinds(incident) for incident in group_incidents(episodes)] == [
        ["error_recovery", "error_recovery", "retry_loop"],
        ["user_correction"],
        ["approval_denied"],
        ["skill_gap"],
    ]
    assert evidence_score(episodes) == pytest.approx(3.0)
    decision = gate_decision(episodes, min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert (decision.passes, decision.reason) == (True, GATE_SCORE_REACHED)


EXPECTED_KINDS = {
    "exgraph_accuracy.jsonl": {},
    "exgraph_reuse.jsonl": {"skill_gap": 1},
    "malformed_lines.jsonl": {"approval_denied": 1},
    "missing_turn_end.jsonl": {"skill_gap": 1},
    "normal_subagent.jsonl": {},
    "orphan_tool_start.jsonl": {"skill_gap": 1},
    "recorder_limit.jsonl": {"skill_gap": 1},
    "result_without_start.jsonl": {},
    "skill_evolver_demo_success.jsonl": {},
    "skill_evolver_signals.jsonl": {
        "error_recovery": 2,
        "user_correction": 1,
        "retry_loop": 1,
        "approval_denied": 1,
        "skill_gap": 1,
    },
}


def test_every_fixture_has_expected_kinds() -> None:
    assert sorted(EXPECTED_KINDS) == FIXTURE_IDS


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=FIXTURE_IDS)
def test_fixture_episode_kinds(path: Path) -> None:
    episodes = extract_episodes(load_trajectory(path), skill_index=_index())
    assert Counter(_kinds(episodes)) == Counter(EXPECTED_KINDS[path.name])


def test_malformed_fixture_denial_is_anchored_to_the_prelude() -> None:
    (episode,) = extract_episodes(load_trajectory(FIXTURES / "malformed_lines.jsonl"))
    assert episode.kind == "approval_denied"
    assert episode.evidence_seq == [3]
    assert episode.tool_sequence == []
    assert episode.facts["tools"] == ["bash"]
    assert episode.anchors == [f"{PRELUDE_RUN_ID}#3"]


def _recorded_seqs(path: Path) -> set[int]:
    """Seqs of the events the reader accepts (its own schema-v1 filter)."""
    seqs: set[int] = set()
    for event in iter_events(path):
        valid = event.get("v") == 1 and isinstance(event.get("event"), str)
        if valid and isinstance(event.get("seq"), int):
            seqs.add(event["seq"])
    return seqs


def _anchor_seqs(episode: Episode) -> set[int]:
    """The seq part of every anchor; anchors must read ``<run_id>#<seq>``."""
    assert all(isinstance(anchor, str) and "#" in anchor for anchor in episode.anchors)
    return {int(anchor.rsplit("#", 1)[1]) for anchor in episode.anchors}


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=FIXTURE_IDS)
def test_evidence_seqs_exist_in_source(path: Path) -> None:
    traj = load_trajectory(path)
    recorded = _recorded_seqs(path)

    for episode in extract_episodes(traj, skill_index=_index()):
        assert episode.kind in EPISODE_WEIGHTS
        assert episode.thread_id == traj.thread_id
        assert episode.source == path.name
        assert episode.evidence_seq == sorted(set(episode.evidence_seq))
        assert set(episode.evidence_seq) <= recorded
        assert _anchor_seqs(episode) <= set(episode.evidence_seq)
        assert 0.0 <= episode.weight <= 1.0
        assert any(item.required for item in episode.evidence)
        for item in episode.evidence:
            # Every ref points at the physical line that holds that very seq.
            assert item.ref.source == path.name
            assert _seq_at(path, item.ref.line) == item.ref.seq


def test_cross_session_evidence_exists_in_source() -> None:
    trajs = [load_trajectory(path) for path in FIXTURE_FILES]
    recorded = {traj.thread_id: _recorded_seqs(traj.path) for traj in trajs}
    paths = {traj.source: traj.path for traj in trajs}

    episodes = mine_cross_session(trajs)

    assert episodes
    for episode in episodes:
        assert episode.kind == "repeated_procedure"
        assert episode.facts["support"] == len(episode.facts["thread_ids"]) >= 2
        assert episode.thread_id in episode.facts["thread_ids"]
        assert episode.tool_sequence == episode.facts["ngram"]
        assert 2 <= len(episode.tool_sequence) <= 5
        assert set(episode.evidence_seq) <= recorded[episode.thread_id]
        assert _anchor_seqs(episode) <= set(episode.evidence_seq)
        # Layout: the owner's steps (each followed by its result unless the
        # call was recorded without tool.start, where start == end), the
        # second thread's steps, then the owner's task; steps required, the
        # rest context; every ref resolves to its physical line.
        n = len(episode.tool_sequence)
        own = [item for item in episode.evidence if item.ref.source == episode.source and item.role != "task"]
        other = [item for item in episode.evidence if item.ref.source != episode.source]
        task = [item for item in episode.evidence if item.role == "task"]
        own_results = [item for item in own if item.role == "result"]
        assert len(own_results) in (0, n) and len(own) == n + len(own_results)
        assert [item.role for item in own if item.role != "result"] == ["step"] * n
        assert [(item.role, item.required) for item in other] == [("step", True)] * n
        assert len({item.ref.source for item in other}) == 1
        assert [(item.role, item.required) for item in task] == [("task", False)]
        assert episode.evidence == [*own, *other, *task]
        assert all(item.required for item in own if item.role == "step")
        assert not any(item.required for item in own_results)
        for item in episode.evidence:
            assert _seq_at(paths[item.ref.source], item.ref.line) == item.ref.seq
    procedures = {tuple(e.tool_sequence): e for e in episodes}
    shared = procedures[("bash", "read_file", "grep", "bash")]
    assert shared.facts["thread_ids"] == ["thread-ctrlc", "thread-limit"]
    assert shared.thread_id == "thread-ctrlc"
    assert shared.evidence_seq == [2, 4, 5, 7, 8, 10, 11, 13, 14]
    assert shared.anchors == ["run-1#4", "run-1#7", "run-1#10", "run-1#13"]
    # The fifth step of the old five-gram failed in one thread: no longer a procedure.
    assert ("bash", "read_file", "grep", "bash", "bash") not in procedures
    # thread-reuse, thread-ctrlc, thread-limit, thread-signals
    assert procedures[("bash", "read_file")].facts["support"] == 4


# ------------------------------------------------------- incidents and gate


def test_group_incidents_links_episodes_transitively_by_anchor() -> None:
    def make(kind: EpisodeKind, seqs: list[int], anchors: list[str]) -> Episode:
        return _ep(kind, seqs, anchors=anchors)

    first = make("error_recovery", [4, 8], ["run-1#4", "run-1#8"])
    second = make("retry_loop", [8, 12], ["run-1#8", "run-1#12"])
    third = make("approval_denied", [20], ["run-1#20"])
    fourth = make("error_recovery", [12, 14], ["run-1#12", "run-1#14"])
    gap = make("skill_gap", [2, 4], [])
    twin = make("skill_gap", [2, 4], [])
    other_gap = make("skill_gap", [2, 6], [])

    incidents = group_incidents([first, second, third, fourth, gap, twin, other_gap])

    assert incidents == [[first, second, fourth], [third], [gap, twin], [other_gap]]
    assert group_incidents([]) == []


def test_one_incident_seen_by_several_detectors_counts_once() -> None:
    episodes = extract_episodes(_traj(_turn("run-1", 2, "profile", _msprof_chain(4))))

    assert _kinds(episodes) == ["error_recovery", "error_recovery", "retry_loop"]
    decision = gate_decision(episodes, min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert len(decision.incidents) == 1
    assert decision.score == pytest.approx(0.7)
    assert (decision.passes, decision.reason) == (False, GATE_SCORE_BELOW)
    assert evidence_score(episodes) == pytest.approx(0.7)


def test_independent_incidents_add_up() -> None:
    traj = _traj(
        _turn("run-1", 2, "profile", _msprof_chain(4)),
        _turn("run-2", 20, "again", _msprof_chain(22)),
    )

    episodes = extract_episodes(traj)

    assert len(episodes) == 6
    decision = gate_decision(episodes, min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert len(decision.incidents) == 2
    assert decision.score == pytest.approx(1.4)
    assert (decision.passes, decision.reason) == (True, GATE_SCORE_REACHED)


def test_shared_turn_does_not_merge_incidents() -> None:
    denied = _approval(14, {"action": "reject"}, {"tool": "write_file", "args": {"path": "r.md"}})
    traj = _traj(
        _turn("run-1", 2, "do it", [_call("bash", {"cmd": "a"}, seq=4)]),
        _turn(
            "run-2",
            10,
            "不，不对 —— 应该先用 grep 搜索",
            [_call("grep", {"pattern": "b"}, seq=12), _call("ls", {}, seq=16)],
            approvals=[denied],
        ),
    )

    episodes = extract_episodes(traj)

    assert _kinds(episodes) == ["user_correction", "approval_denied"]
    assert [e.anchors for e in episodes] == [["run-2#10"], ["run-2#14"]]
    decision = gate_decision(episodes, min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert len(decision.incidents) == 2
    assert decision.score == pytest.approx(1.9)
    assert (decision.passes, decision.reason) == (True, GATE_SCORE_REACHED)


def test_duplicate_episode_does_not_change_the_gate() -> None:
    signals = load_trajectory(FIXTURES / "skill_evolver_signals.jsonl")
    episodes = extract_episodes(signals, skill_index=_index())
    base = gate_decision(episodes, min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert len(episodes) == 6
    assert len(base.incidents) == 4

    for extra in episodes:
        again = gate_decision([*episodes, extra], min_score=DEFAULT_MIN_EVIDENCE_SCORE)
        assert (again.score, again.passes, again.reason) == (base.score, base.passes, base.reason)
        assert len(again.incidents) == len(base.incidents)


def test_gate_decision_compares_exact_decimal_sums() -> None:
    # 0.6 + 0.7 + 0.7 is 2.0 to the user who set the threshold, not
    # 1.9999999999999998.
    def episode(kind: str, seq: int) -> Episode:
        return _ep(kind, [seq], anchors=[f"run-1#{seq}"], thread_id="thread-t")

    episodes = [episode("error_recovery", 2), episode("retry_loop", 10), episode("retry_loop", 20)]

    decision = gate_decision(episodes, min_score=2.0)

    assert decision.score == 2.0
    assert (decision.passes, decision.reason) == (True, GATE_SCORE_REACHED)


def test_gate_decision_reasons() -> None:
    gap = _ep("skill_gap", [2, 6])
    denial = _ep("approval_denied", [9], anchors=["run-1#9"])

    empty = gate_decision([], min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert empty.incidents == []
    assert (empty.score, empty.passes, empty.reason) == (0.0, False, GATE_NO_EPISODES)
    # No episodes never passes, whatever the threshold.
    assert gate_decision([], min_score=0.0).reason == GATE_NO_EPISODES

    below = gate_decision([gap], min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert below.incidents == [[gap]]
    assert (below.score, below.passes, below.reason) == (0.4, False, GATE_SCORE_BELOW)

    reached = gate_decision([denial], min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert (reached.incidents, reached.score, reached.passes) == ([[denial]], 1.0, True)
    assert reached.reason == GATE_SCORE_REACHED
    assert gate_decision([gap], min_score=0.4).reason == GATE_SCORE_REACHED


def test_strong_correction_passes_the_default_gate() -> None:
    traj = _traj(
        _turn("run-1", 2, "do it", [_call("bash", {"cmd": "a"}, seq=4)]),
        _turn(
            "run-2",
            10,
            "不，不对 —— 应该先用 grep",
            [_call("grep", {"pattern": "b"}, seq=12)],
        ),
    )
    (episode,) = extract_episodes(traj)
    assert episode.facts["strength"] == "strong"

    decision = gate_decision([episode], min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert decision.score == pytest.approx(0.9)
    assert (decision.passes, decision.reason) == (True, GATE_STRONG_CORRECTION)
    # A stricter threshold opts out of the rule.
    strict = gate_decision([episode], min_score=1.5)
    assert (strict.passes, strict.reason) == (False, GATE_SCORE_BELOW)
    # Below the score the ordinary rule applies first.
    lenient = gate_decision([episode], min_score=0.5)
    assert (lenient.passes, lenient.reason) == (True, GATE_SCORE_REACHED)


# --------------------------------------------------------- context, primary_seq


def test_context_items_and_primary_seq() -> None:
    traj = load_trajectory(FIXTURES / "skill_evolver_signals.jsonl")

    episodes = extract_episodes(traj, skill_index=_index())

    assert [(e.kind, e.evidence_seq) for e in episodes] == [
        ("error_recovery", [2, 4, 5, 10, 11]),
        ("error_recovery", [2, 7, 8, 10, 11]),
        ("user_correction", [2, 10, 17, 19]),
        ("retry_loop", [2, 4, 5, 7, 8, 10, 11]),
        ("approval_denied", [26, 29, 32]),
        ("skill_gap", [2, 4, 17]),
    ]
    # The identity consumers key on is the first required own event, not the
    # shared task context both recoveries cite.
    assert [e.primary_seq for e in episodes] == [5, 8, 17, 4, 26, 2]
    tasks = [item for e in episodes for item in e.evidence if item.role == "task"]
    assert len(tasks) == 3 and not any(item.required for item in tasks)
    assert {item.ref.seq for item in tasks} == {2}
    # A group head without a message adds no task item.
    err = _call("bash", {"cmd": "a"}, seq=4, status="error")
    fixed = _call("bash", {"cmd": "b"}, seq=6)
    (episode,) = extract_episodes(_traj(_turn("run-1", 2, None, [err, fixed])))
    assert episode.evidence_seq == [4, 5, 6, 7]
    assert "task" not in {item.role for item in episode.evidence}
    # A hand-built episode without required own items falls back to its first seq.
    fallback = Episode(
        kind="skill_gap",
        thread_id="t",
        source="t.jsonl",
        evidence=[_item(6, required=False), _item(3, source="o.jsonl")],
        tool_sequence=[],
        facts={},
        weight=0.4,
    )
    assert fallback.primary_seq == 6


def test_has_required_evidence_per_kind() -> None:
    assert set(REQUIRED_ROLES) == EPISODE_KINDS
    signals = load_trajectory(FIXTURES / "skill_evolver_signals.jsonl")
    demo = load_trajectory(FIXTURES / "skill_evolver_demo_success.jsonl")
    episodes = [
        *extract_episodes(signals, skill_index=_index()),
        *extract_episodes(demo, demo=True),
        *mine_cross_session([signals, load_trajectory(FIXTURES / "exgraph_reuse.jsonl")]),
    ]
    assert {e.kind for e in episodes} == EPISODE_KINDS
    assert all(has_required_evidence(e) for e in episodes)
    # A start-less recovery collapses fixed_call into result: its arguments are unknown.
    failed = _call("bash", {}, seq=4, status="error", error="boom", seq_end=4)
    fixed = _call("bash", {"cmd": "make"}, seq=8, seq_end=8)
    (startless,) = extract_episodes(_traj(_turn("run-1", 2, "go", [failed, fixed])))
    assert {item.role for item in startless.evidence if item.required} == {"error", "result"}
    assert not has_required_evidence(startless)
    # The role must be carried by a *required* item.
    (observed,) = extract_episodes(demo, demo=True)
    demoted = [replace(item, required=False) if item.role == "task" else item for item in observed.evidence]
    assert not has_required_evidence(replace(observed, evidence=demoted))
    assert not has_required_evidence(_ep("observed_procedure", [2, 4], weight=0.0))


# -------------------------------------------------------- observed_procedure


def _ok(name: str, args: dict[str, Any], *, seq: int, output: str = "done", subagent: str | None = None) -> ToolCall:
    return _call(name, args, seq=seq, output=output, subagent=subagent)


def _observed(traj: Trajectory, notes: list[DetectorNote] | None = None) -> list[Episode]:
    return [e for e in extract_episodes(traj, demo=True, notes=notes) if e.kind == "observed_procedure"]


def test_observed_procedure_demo_fixture_single_chain() -> None:
    path = FIXTURES / "skill_evolver_demo_success.jsonl"
    traj = load_trajectory(path)
    assert (traj.thread_id, traj.agent, traj.working_dir) == (
        "thread-demo-synthetic",
        "SyntheticDemo",
        "/synthetic/demo",
    )
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["synthetic"] is True
    assert "data" not in (traj.turns[0].user_message or "")
    # Ordinary detectors see nothing; no LLM-free rule fires on a plain success.
    assert extract_episodes(traj, skill_index=_index()) == []

    notes: list[DetectorNote] = []
    (episode,) = extract_episodes(traj, skill_index=_index(), demo=True, notes=notes)

    assert notes == []
    assert episode.kind == "observed_procedure"
    assert episode.weight == 0.0
    assert _roles(episode) == [
        ("task", 2, True),
        ("step", 4, True),
        ("result", 5, True),
        ("step", 7, True),
        ("result", 8, True),
        ("step", 10, True),
        ("result", 11, True),
    ]
    assert episode.anchors == ["run-d1#4", "run-d1#7", "run-d1#10"]
    assert episode.tool_sequence == ["read_file", "bash", "bash"]
    assert episode.primary_seq == 2
    assert has_required_evidence(episode)
    assert episode.facts["task"] == traj.turns[0].user_message
    assert (episode.facts["run_id"], episode.facts["subagent"]) == ("run-d1", None)
    assert (episode.facts["chain_index"], episode.facts["calls"]) == (1, 3)
    assert episode.facts["outcome"] == "TOTAL_OK"
    assert episode.facts["verification"] == "observed output of bash: TOTAL_OK"
    steps = episode.facts["steps"]
    assert [step["tool"] for step in steps] == ["read_file", "bash", "bash"]
    assert steps[0]["args"] == {"path": "input/sales.csv"} and steps[0]["work_object"] == "input/sales.csv"
    assert steps[1]["args"]["cmd"].startswith("python3 -c") and steps[1]["result"] == "60.0"
    assert steps[2]["args"]["cmd"].endswith("…")  # clipped like every fact value
    assert [item.snippet for item in episode.evidence if item.role == "result"] == [
        "id,amount\n1,10\n2,20\n3,30\n",
        "60.0",
        "TOTAL_OK",
    ]
    for item in episode.evidence:
        assert _seq_at(path, item.ref.line) == item.ref.seq


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=FIXTURE_IDS)
def test_observed_procedure_not_extracted_without_demo_on_any_fixture(path: Path) -> None:
    traj = load_trajectory(path)
    # exgraph calls extract_episodes(traj) without keywords: same path, same result.
    assert "observed_procedure" not in _kinds(extract_episodes(traj))
    assert "observed_procedure" not in _kinds(extract_episodes(traj, skill_index=_index()))


def test_observed_procedure_chain_limits_and_windows() -> None:
    twelve = [_ok(f"t{i}", {"n": i}, seq=4 + 2 * i) for i in range(12)]
    notes: list[DetectorNote] = []

    chains = _observed(_traj(_turn("run-1", 2, "go", twelve)), notes)

    assert notes == []
    assert [len(e.tool_sequence) for e in chains] == [5, 5, 2]
    assert [e.facts["chain_index"] for e in chains] == [1, 2, 3]
    assert [e.anchors[0] for e in chains] == ["run-1#4", "run-1#14", "run-1#24"]
    refs = [item.ref for e in chains for item in e.evidence if item.role != "task"]
    assert len(refs) == len(set(refs))  # windows are disjoint
    assert all(len(e.tool_sequence) <= OBSERVED_MAX_CALLS for e in chains)
    # Four segments (ok calls separated by failures of another tool): three
    # chains and one note for the fourth.
    calls: list[ToolCall] = []
    for i in range(4):
        calls.append(_ok(f"step{i}", {"n": i}, seq=4 + 4 * i))
        calls.append(_call("fail", {"n": i}, seq=6 + 4 * i, status="error"))
    notes = []
    chains = _observed(_traj(_turn("run-1", 2, "go", calls)), notes)
    assert [e.tool_sequence for e in chains] == [["step0"], ["step1"], ["step2"]]
    assert len(chains) == OBSERVED_MAX_CHAINS
    assert [(n.detector, n.reason, n.seqs) for n in notes] == [("observed_procedure", DROP_CHAIN_LIMIT, [16])]
    assert notes[0].detail == "chain step3 skipped: 3 chains already selected"


def test_observed_procedure_chain_limit_is_the_last_disqualification() -> None:
    # Three selected one-call chains; the next window carries an error marker
    # and is dropped for that, the one after it is valid and hits the limit.
    calls: list[ToolCall] = []
    for i in range(3):
        calls.append(_ok(f"step{i}", {"n": i}, seq=4 + 4 * i))
        calls.append(_call("fail", {"n": i}, seq=6 + 4 * i, status="error"))
    calls.append(_ok("bash", {"cmd": "make"}, seq=16, output="process finished with exit code 2"))
    calls.append(_call("fail", {"n": 3}, seq=18, status="error"))
    calls.append(_ok("step4", {"n": 4}, seq=20))
    notes: list[DetectorNote] = []

    chains = _observed(_traj(_turn("run-1", 2, "go", calls)), notes)

    assert [e.tool_sequence for e in chains] == [["step0"], ["step1"], ["step2"]]
    assert [(n.reason, n.seqs) for n in notes] == [(DROP_ERROR_IN_OUTPUT, [16]), (DROP_CHAIN_LIMIT, [20])]
    assert notes[0].detail == "bash at seq 17 returned ok but its output contains 'exit code 2'"
    assert notes[1].detail == "chain step4 skipped: 3 chains already selected"


def test_observed_procedure_single_call_is_a_chain() -> None:
    traj = _traj(_turn("run-1", 2, "list it", [_ok("bash", {"cmd": "ls -la"}, seq=4, output="a.txt\nb.txt")]))

    (episode,) = _observed(traj)

    assert _roles(episode) == [("task", 2, True), ("step", 4, True), ("result", 5, True)]
    assert episode.anchors == ["run-1#4"]
    assert episode.facts["outcome"] == "a.txt\nb.txt"
    assert episode.facts["steps"] == [
        {"tool": "bash", "work_object": "ls", "args": {"cmd": "ls -la"}, "result": "a.txt\nb.txt"},
    ]
    # mine_cross_session keeps NGRAM_MIN: a single call is a chain only for the detector.
    assert len(features_mod._procedure_segments(traj, min_len=1)) == 1
    assert features_mod._procedure_segments(traj) == []


def test_observed_procedure_orphan_only_and_ai_claim_yield_no_episode() -> None:
    orphan = _call("bash", {"cmd": "python3 x.py"}, seq=4, status="orphan")
    notes: list[DetectorNote] = []
    assert extract_episodes(_traj(_turn("run-1", 2, "sum it", [orphan])), demo=True, notes=notes) == []
    assert [(n.detector, n.reason) for n in notes] == [("observed_procedure", DROP_NO_COMPLETED_CALLS)]
    assert notes[0].detail == "1 domain calls: 0 ok, 0 error, 1 orphan; no completed chain"
    # The assistant's text is never evidence: a claimed success without a
    # completed call is no chain at all, with or without the orphan.
    claim = _ai(3, "Done: the total is 60")
    for calls in ([], [orphan]):
        notes = []
        traj = _traj(_turn("run-1", 2, "sum it", calls, ai=[claim]))
        assert extract_episodes(traj, demo=True, notes=notes) == []
        assert [n.reason for n in notes] == [DROP_NO_COMPLETED_CALLS]
        assert notes[0].detail.endswith(f"{len(calls)} orphan; no completed chain")
    # Catalog calls are not domain work.
    notes = []
    catalog = _traj(_turn("run-1", 2, "sum it", [_ok("get_skill", {"name": "x"}, seq=4)]))
    assert extract_episodes(catalog, demo=True, notes=notes) == []
    assert notes[0].detail == "0 domain calls: 0 ok, 0 error, 0 orphan; no completed chain"


@pytest.mark.parametrize(
    ("output", "marker"),
    [
        ("Traceback (most recent call last):\n  File x\nKeyError: 'amount'", "Traceback (most recent call last)"),
        ("KeyError: 'amount'", "KeyError"),
        ("langchain_core.tools.ToolException: bad input", "ToolException"),
        ("Error: unknown column", "Error:"),
        ("fatal: not a git repository", "fatal:"),
        ("FAILED tests/test_x.py::test_y", "FAILED"),
        ("cat: input/sales.csv: No such file or directory", "No such file or directory"),
        ("bash: python4: command not found", "command not found"),
        ("rm: out.txt: Permission denied", "Permission denied"),
        ("process finished with exit code 2", "exit code 2"),
        ("Exit status 130", "Exit status 130"),
        ("non-zero exit from make", "non-zero exit"),
    ],
)
def test_observed_procedure_error_marker_in_ok_output_drops_chain(output: str, marker: str) -> None:
    assert features_mod._error_marker(output) == marker
    read = _ok("read_file", {"path": "s.csv"}, seq=4, output="id,amount")
    check = _ok("bash", {"cmd": "python3 check.py"}, seq=8, output="TOTAL_OK")
    # An erroneous output disqualifies the chain wherever it sits, last or between.
    for calls in (
        [read, _ok("bash", {"cmd": "python3 sum.py"}, seq=6, output=output)],
        [read, _ok("bash", {"cmd": "python3 sum.py"}, seq=6, output=output), check],
    ):
        notes: list[DetectorNote] = []
        assert _observed(_traj(_turn("run-1", 2, "sum it", calls)), notes) == []
        (note,) = notes
        assert (note.detector, note.reason) == ("observed_procedure", DROP_ERROR_IN_OUTPUT)
        assert note.detail == f"bash at seq 7 returned ok but its output contains {marker!r}"
        assert note.seqs == [call.seq_start for call in calls]


def test_error_markers_do_not_flag_ordinary_output() -> None:
    assert len(ERROR_MARKERS) == 10 and r"\berror\b" not in ERROR_MARKERS
    for text in ("3 passed", "TOTAL_OK", "0 errors", "error-free build", "exit code 0", "60.0", "id,amount\n1,10"):
        assert features_mod._error_marker(text) is None, text
        (episode,) = _observed(_traj(_turn("run-1", 2, "go", [_ok("bash", {"cmd": "make"}, seq=4, output=text)])))
        assert episode.facts["outcome"] == text


def test_observed_procedure_requires_task_start_and_last_output() -> None:
    read = _ok("read_file", {"path": "s.csv"}, seq=4, output="id,amount")
    total = _ok("bash", {"cmd": "python3 sum.py"}, seq=6, output="60.0")
    # No user message on the group head: an earlier group's message is never borrowed.
    notes: list[DetectorNote] = []
    for head in (_turn("run-2", 10, None, [read, total]), _turn("run-2", 10, "  ", [read, total])):
        notes = []
        traj = _traj(_turn("run-1", 2, "sum it", [_call("ls", {}, seq=3, output="s.csv", seq_end=3)]), head)
        assert _observed(traj, notes) == []
        assert [n.reason for n in notes] == [DROP_MISSING_START, DROP_NO_TASK]
        assert notes[1].detail == "turn run-2 (dispatch) has no user message"
        assert notes[1].seqs == [4, 6]
    prelude = _turn(PRELUDE_RUN_ID, 1, None, [_ok("bash", {"cmd": "ls"}, seq=3)], source="prelude")
    notes = []
    assert _observed(_traj(prelude), notes) == []
    assert (notes[0].reason, notes[0].detail) == (DROP_NO_TASK, f"turn {PRELUDE_RUN_ID} (prelude) has no user message")
    # A call recorded without tool.start has unknown arguments.
    notes = []
    startless = _call("bash", {}, seq=6, output="60.0", seq_end=6)
    assert _observed(_traj(_turn("run-1", 2, "sum it", [read, startless])), notes) == []
    assert (notes[0].reason, notes[0].detail) == (
        DROP_MISSING_START,
        "bash at seq 6 was recorded without tool.start; its arguments are unknown",
    )
    # The last call must show an observable outcome; an empty intermediate output is fine.
    notes = []
    silent = _ok("bash", {"cmd": "python3 sum.py"}, seq=6, output="  \n")
    assert _observed(_traj(_turn("run-1", 2, "sum it", [read, silent])), notes) == []
    assert (notes[0].reason, notes[0].detail) == (
        DROP_EMPTY_RESULT,
        "last call bash at seq 7 has no output; no observable outcome",
    )
    quiet_first = _ok("read_file", {"path": "s.csv"}, seq=4, output="")
    (episode,) = _observed(_traj(_turn("run-1", 2, "sum it", [quiet_first, total])))
    assert episode.facts["steps"][0]["result"] == "" and episode.facts["outcome"] == "60.0"
    # Every call of result_without_start.jsonl lacks its start.
    notes = []
    traj = load_trajectory(FIXTURES / "result_without_start.jsonl")
    assert extract_episodes(traj, skill_index=_index(), demo=True, notes=notes) == []
    assert notes and {n.reason for n in notes} == {DROP_MISSING_START}


def test_observed_procedure_keeps_streams_apart_and_uses_physical_identity() -> None:
    calls = [
        _ok("read_file", {"path": "s.csv"}, seq=4, output="id,amount"),
        _ok("bash", {"cmd": "python3 a.py"}, seq=6, output="1", subagent="tools:b2"),
        _ok("bash", {"cmd": "python3 b.py"}, seq=8, output="2"),
    ]

    root, aside = _observed(_traj(_turn("run-1", 2, "sum it", calls)))

    assert (root.tool_sequence, root.facts["subagent"], root.anchors) == (
        ["read_file", "bash"],
        None,
        ["run-1#4", "run-1#8"],
    )
    assert (aside.tool_sequence, aside.facts["subagent"], aside.anchors) == (["bash"], "tools:b2", ["run-1#6"])
    assert (root.facts["chain_index"], aside.facts["chain_index"]) == (1, 2)
    # Two recorder writers repeat seqs 2..8 on later physical lines: refs are
    # lines, so the third chain is distinct from the first.
    path = FIXTURES / "missing_turn_end.jsonl"
    first, second, third = _observed(load_trajectory(path))
    assert first.anchors == ["run-1#4", "run-1#7", "run-1#10", "run-1#13"]
    assert second.anchors == ["run-2#18"]
    assert third.anchors == ["run-3#4", "run-3#7"]
    assert [(item.ref.line, item.ref.seq) for item in third.evidence] == [(23, 2), (25, 4), (26, 5), (28, 7), (29, 8)]
    assert {item.ref for item in first.evidence}.isdisjoint({item.ref for item in third.evidence})
    for episode in (first, second, third):
        for item in episode.evidence:
            assert _seq_at(path, item.ref.line) == item.ref.seq


def test_observed_procedure_coexists_with_retry_loop_and_shares_incident() -> None:
    searches = [
        _ok("grep", {"pattern": "hotspot", "path": "a.csv"}, seq=4, output="-"),
        _ok("grep", {"pattern": "hot_spot", "path": "a.csv"}, seq=6, output="-"),
        _ok("grep", {"pattern": "HotSpot", "path": "a.csv"}, seq=8, output="HotSpot,3"),
    ]
    procedure = [
        _ok("read_file", {"path": "s.csv"}, seq=14, output="id,amount"),
        _ok("bash", {"cmd": "python3 sum.py"}, seq=16, output="60.0"),
    ]
    traj = _traj(_turn("run-1", 2, "find it", searches), _turn("run-2", 12, "sum it", procedure))
    assert _kinds(extract_episodes(traj)) == ["retry_loop"]

    episodes = extract_episodes(traj, demo=True)

    retry, greps, chain = episodes
    assert _kinds(episodes) == ["retry_loop", "observed_procedure", "observed_procedure"]
    assert greps.anchors == retry.anchors == ["run-1#4", "run-1#6", "run-1#8"]
    assert chain.anchors == ["run-2#14", "run-2#16"]
    # The grep chain describes the retry's events: one incident; the other chain
    # is its own zero-weight incident, so the score does not move.
    assert [_kinds(incident) for incident in group_incidents(episodes)] == [
        ["retry_loop", "observed_procedure"],
        ["observed_procedure"],
    ]
    assert evidence_score(episodes) == evidence_score([retry]) == pytest.approx(0.7)
    plain = gate_decision(episodes, min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert (plain.passes, plain.describe()) == (False, GATE_SCORE_BELOW)
    demo = gate_decision(episodes, min_score=DEFAULT_MIN_EVIDENCE_SCORE, demo=True)
    assert (demo.passes, demo.reason, demo.score) == (True, GATE_DEMO_OVERRIDE, pytest.approx(0.7))
    assert demo.detail == "observed_procedure, retry_loop have required evidence"


def test_zero_weight_chain_joins_an_incident_but_never_bridges_two() -> None:
    first = _ep("error_recovery", [4, 6], anchors=["run-1#4", "run-1#6"])
    second = _ep("retry_loop", [8, 10], anchors=["run-1#8", "run-1#10"])
    bridge = _ep("observed_procedure", [6, 8], anchors=["run-1#6", "run-1#8"])
    twin = _ep("observed_procedure", [6, 8], anchors=["run-1#6", "run-1#8"])
    assert group_incidents([first, second, bridge, twin]) == [[first, bridge, twin], [second]]
    assert group_incidents([bridge, first, second]) == [[bridge, first], [second]]

    # bash fails then succeeds with changed arguments (error_recovery), then
    # three ok greps with a varying pattern (retry_loop): two incidents. In demo
    # the ok run bash → grep × 3 is one observed chain anchored on both of them.
    calls = [
        _call("bash", {"cmd": "python3 sum.py"}, seq=4, status="error"),
        _ok("bash", {"cmd": "python3 sum.py --fix"}, seq=6, output="60.0"),
        _ok("grep", {"pattern": "hotspot", "path": "a.csv"}, seq=8, output="-"),
        _ok("grep", {"pattern": "hot_spot", "path": "a.csv"}, seq=10, output="-"),
        _ok("grep", {"pattern": "HotSpot", "path": "a.csv"}, seq=12, output="HotSpot,3"),
    ]
    traj = _traj(_turn("run-1", 2, "sum it, then find it", calls))
    plain = extract_episodes(traj)
    demo = extract_episodes(traj, demo=True)

    assert _kinds(plain) == ["error_recovery", "retry_loop"]
    assert _kinds(demo) == ["error_recovery", "retry_loop", "observed_procedure"]
    assert demo[2].anchors == ["run-1#6", "run-1#8", "run-1#10", "run-1#12"]
    assert [_kinds(incident) for incident in group_incidents(demo)] == [
        ["error_recovery", "observed_procedure"],
        ["retry_loop"],
    ]
    assert len(group_incidents(plain)) == len(group_incidents(demo)) == 2
    assert evidence_score(plain) == evidence_score(demo) == pytest.approx(1.3)
    without = gate_decision(plain, min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    with_demo = gate_decision(demo, min_score=DEFAULT_MIN_EVIDENCE_SCORE, demo=True)
    assert (
        (without.passes, without.describe()) == (with_demo.passes, with_demo.describe()) == (True, GATE_SCORE_REACHED)
    )


# Chains the observed_procedure detector selects per fixture (demo=True), and
# the drop notes it records; computed at implementation time, then pinned.
EXPECTED_DEMO_CHAINS = {
    "exgraph_accuracy.jsonl": 1,
    "exgraph_reuse.jsonl": 1,
    "malformed_lines.jsonl": 3,
    "missing_turn_end.jsonl": 3,
    "normal_subagent.jsonl": 2,
    "orphan_tool_start.jsonl": 3,
    "recorder_limit.jsonl": 2,
    "result_without_start.jsonl": 0,
    "skill_evolver_demo_success.jsonl": 1,
    "skill_evolver_signals.jsonl": 2,
}
EXPECTED_DEMO_NOTES = {
    "orphan_tool_start.jsonl": {DROP_CHAIN_LIMIT: 1},
    "result_without_start.jsonl": {DROP_MISSING_START: 4},
}


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=FIXTURE_IDS)
def test_demo_evidence_refs_resolve_on_every_fixture(path: Path) -> None:
    assert sorted(EXPECTED_DEMO_CHAINS) == FIXTURE_IDS
    traj = load_trajectory(path)
    notes: list[DetectorNote] = []

    episodes = extract_episodes(traj, skill_index=_index(), demo=True, notes=notes)

    ordinary = [e for e in episodes if e.kind != "observed_procedure"]
    observed = [e for e in episodes if e.kind == "observed_procedure"]
    assert Counter(_kinds(ordinary)) == Counter(EXPECTED_KINDS[path.name])
    assert len(observed) == EXPECTED_DEMO_CHAINS[path.name]
    assert Counter(note.reason for note in notes) == Counter(EXPECTED_DEMO_NOTES.get(path.name, {}))
    assert all(note.detector == "observed_procedure" for note in notes)
    assert [e.facts["chain_index"] for e in observed] == list(range(1, len(observed) + 1))
    for episode in observed:
        assert episode.source == path.name and episode.thread_id == traj.thread_id
        assert episode.weight == 0.0
        assert 1 <= len(episode.tool_sequence) <= OBSERVED_MAX_CALLS
        assert len(episode.anchors) == len(episode.tool_sequence) == episode.facts["calls"]
        assert has_required_evidence(episode)
        assert all(item.required for item in episode.evidence)
        assert _anchor_seqs(episode) <= set(episode.evidence_seq)
        assert episode.facts["outcome"].strip()
        for item in episode.evidence:
            assert item.ref.source == path.name
            assert _seq_at(path, item.ref.line) == item.ref.seq


# ----------------------------------------------------------------- demo gate


def test_gate_decision_demo_override_only_when_ordinary_rules_fail() -> None:
    empty = gate_decision([], min_score=DEFAULT_MIN_EVIDENCE_SCORE, demo=True)
    assert (empty.passes, empty.reason, empty.detail, empty.describe()) == (
        False,
        GATE_NO_EPISODES,
        "",
        GATE_NO_EPISODES,
    )

    (chain,) = extract_episodes(load_trajectory(FIXTURES / "skill_evolver_demo_success.jsonl"), demo=True)
    plain = gate_decision([chain], min_score=DEFAULT_MIN_EVIDENCE_SCORE)
    assert (plain.score, plain.passes, plain.reason, plain.detail) == (0.0, False, GATE_SCORE_BELOW, "")
    assert plain.describe() == GATE_SCORE_BELOW
    demo = gate_decision([chain], min_score=DEFAULT_MIN_EVIDENCE_SCORE, demo=True)
    assert (demo.score, demo.passes, demo.reason) == (0.0, True, GATE_DEMO_OVERRIDE)
    assert demo.detail == "observed_procedure has required evidence"
    assert demo.describe() == "demo_override; observed_procedure has required evidence"
    assert demo.incidents == [[chain]]

    # The ordinary rules come first and carry no detail.
    denial = _ep("approval_denied", [9], anchors=["run-1#9"])
    reached = gate_decision([denial, chain], min_score=DEFAULT_MIN_EVIDENCE_SCORE, demo=True)
    assert (reached.passes, reached.reason, reached.detail) == (True, GATE_SCORE_REACHED, "")
    correcting = _traj(
        _turn("run-1", 2, "do it", [_call("bash", {"cmd": "a"}, seq=4)]),
        _turn("run-2", 10, "不，不对 —— 应该先用 grep", [_call("grep", {"pattern": "b"}, seq=12)]),
    )
    (correction,) = extract_episodes(correcting)
    strong = gate_decision([correction], min_score=DEFAULT_MIN_EVIDENCE_SCORE, demo=True)
    assert (strong.passes, strong.reason, strong.detail) == (True, GATE_STRONG_CORRECTION, "")

    # Without a complete package demo mode changes nothing but the detail.
    stump = replace(chain, evidence=[item for item in chain.evidence if item.role != "result"])
    refused = gate_decision([stump, _ep("skill_gap", [2, 6])], min_score=DEFAULT_MIN_EVIDENCE_SCORE, demo=True)
    assert (refused.score, refused.passes, refused.reason) == (0.4, False, GATE_SCORE_BELOW)
    assert refused.detail == GATE_DEMO_NOT_APPLICABLE
    assert refused.describe() == f"{GATE_SCORE_BELOW}; {GATE_DEMO_NOT_APPLICABLE}"
    # demo=False is untouched by the new rule.
    assert (
        gate_decision([stump, _ep("skill_gap", [2, 6])], min_score=DEFAULT_MIN_EVIDENCE_SCORE).describe()
        == GATE_SCORE_BELOW
    )


# ------------------------------------------------------------ expand context


def test_expand_episode_context_adds_neighbours_results_and_task_only() -> None:
    calls = [
        _ok("ls", {}, seq=4, output="s.csv"),
        _call("bash", {"cmd": "python3 sum.py"}, seq=6, status="error", error="Traceback\nKeyError: 'x'"),
        _ok("bash", {"cmd": "python3 sum.py amount"}, seq=8, output="60.0"),
        _ok("bash", {"cmd": "python3 check.py"}, seq=10, output="TOTAL_OK"),
        _ok("grep", {"pattern": "x"}, seq=12, output="-"),
        _ok("ls", {"path": "."}, seq=14, output="a", subagent="tools:b1"),
    ]
    traj = _traj(_turn("run-1", 2, "sum it", calls))
    source = traj.source

    def cited(seq: int) -> Episode:
        return Episode(
            kind="retry_loop",
            thread_id=traj.thread_id,
            source=source,
            evidence=[EvidenceItem(EvidenceRef(source=source, line=seq, seq=seq), "attempt", True)],
            tool_sequence=["bash"],
            facts={"k": 1},
            weight=0.7,
            anchors=[f"run-1#{seq}"],
        )

    episode = cited(8)
    (expanded,) = expand_episode_context([episode], traj, surrounding_events=1)

    assert expanded is not episode
    assert _roles(expanded) == [
        ("attempt", 8, True),
        ("context_call", 6, False),
        ("context_error", 7, False),
        ("context_result", 9, False),
        ("context_call", 10, False),
        ("context_result", 11, False),
        ("task", 2, False),
    ]
    assert expanded.evidence[2].snippet == "Traceback\nKeyError: 'x'"
    assert (expanded.anchors, expanded.facts, expanded.weight, expanded.kind) == (
        ["run-1#8"],
        {"k": 1},
        0.7,
        "retry_loop",
    )
    # Zero neighbours: only the missing result and the task.
    (tight,) = expand_episode_context([episode], traj, surrounding_events=0)
    assert _roles(tight) == [("attempt", 8, True), ("context_result", 9, False), ("task", 2, False)]
    # The window never leaves the stream (the subagent call at 14 is not a neighbour of 12).
    (edge,) = expand_episode_context([cited(12)], traj, surrounding_events=3)
    assert [item.ref.seq for item in edge.evidence] == [12, 6, 7, 8, 9, 10, 11, 13, 2]
    # ``only`` restricts the expansion; an untouched episode is the same object.
    first, second = expand_episode_context([cited(8), cited(10)], traj, surrounding_events=1, only={1})
    assert first is not None and _roles(first) == [("attempt", 8, True)]
    assert first.evidence == cited(8).evidence and len(second.evidence) == 7
    # Nothing new to add: the same object comes back (a wider window would keep
    # growing through the neighbours' neighbours; the caller expands once).
    (same,) = expand_episode_context([tight], traj, surrounding_events=0)
    assert same is tight
    # Another trajectory's episode and a cross-thread item are never touched.
    other = replace(cited(8), thread_id="o", source="o.jsonl", evidence=[_item(8, source="o.jsonl")])
    assert expand_episode_context([other], traj, surrounding_events=2) == [other]
    shared = replace(cited(8), evidence=[*cited(8).evidence, _item(3, source="o.jsonl")])
    (grown,) = expand_episode_context([shared], traj, surrounding_events=0)
    assert [item.ref.source for item in grown.evidence] == [source, "o.jsonl", source, source]
    # A head without a message adds no task; a negative window is an error.
    quiet = _traj(_turn("run-1", 2, None, calls))
    (no_task,) = expand_episode_context([cited(8)], quiet, surrounding_events=0)
    assert _roles(no_task) == [("attempt", 8, True), ("context_result", 9, False)]
    with pytest.raises(ValueError, match="surrounding_events must be >= 0"):
        expand_episode_context([episode], traj, surrounding_events=-1)


# ----------------------------------------------------------------- isolation

_ISOLATION_PROBE = """
import sys
import msagent.skill_evolver.features as features
import msagent.skill_evolver.retrieval as retrieval
assert features.__file__.startswith(sys.argv[1]), features.__file__
assert retrieval.__file__.startswith(sys.argv[1]), retrieval.__file__
banned = ("langchain", "langgraph", "httpx", "requests", "urllib3", "aiohttp",
          "urllib.request", "http.client")
leaked = sorted(m for m in sys.modules if m.startswith(banned))
assert not leaked, leaked
"""


def test_features_import_no_langchain_or_network() -> None:
    # The pytest process already has langchain loaded (conftest imports the CLI
    # initializer), so the check runs in a fresh interpreter; PYTHONPATH must
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
