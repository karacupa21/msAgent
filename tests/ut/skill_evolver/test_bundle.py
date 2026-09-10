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

"""Tests for the evidence bundle rendered for the classify stage (no LLM)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from msagent.skill_evolver.bundle import (
    ELLIPSIS,
    EXCERPT_LIMIT,
    EXCLUDED_CODE,
    FACT_LIMIT,
    OBSERVED_LINE,
    EvidenceBundle,
    build_evidence_bundle,
)
from msagent.skill_evolver.features import (
    EPISODE_WEIGHTS,
    Episode,
    EvidenceItem,
    extract_episodes,
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
from msagent.trajectory_recorder.reader import load_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "trajectories"
FIXTURE_FILES = sorted(FIXTURES.glob("*.jsonl"))
FIXTURE_IDS = [path.name for path in FIXTURE_FILES]
LOGGER = "msagent.skill_evolver.bundle"
HEADER_RE = re.compile(r"^### Episode E(\d+) — (\w+) \(weight (\d\.\d\d), thread ([^)]{1,8})\)$")

DOCS = [
    SkillDoc("cluster-analysis", "Run the clustering workflow over profiler data"),
    SkillDoc("dit-quant", "Quantize DiT diffusion models with int8 calibration"),
    SkillDoc("ep-parallel", "Adapt expert parallel models for msmodelslim"),
    SkillDoc(
        "profiling-bottleneck",
        "Profile a training run with msprof and locate the kernel bottleneck",
    ),
]


# ------------------------------------------------------------------ builders


def _call(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    seq: int,
    status: ToolStatus = "ok",
    output: str = "",
    error_type: str | None = None,
    error: str | None = None,
    seq_end: int | None = -1,
) -> ToolCall:
    """A tool span starting at ``seq``; by default its result sits at ``seq + 1``.

    ``seq_end=-1`` (the default) derives the end from the status; pass an
    explicit value to model a start-less call (``seq_end == seq``).
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
        subagent=None,
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


def _traj(*turns: Turn, thread_id: str = "thread-t") -> Trajectory:
    return Trajectory(
        path=Path(f"{thread_id}.jsonl"),
        thread_id=thread_id,
        agent="Tester",
        model=None,
        working_dir="/w",
        started_at="2026-09-01T10:00:00.000+00:00",
        turns=list(turns),
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


def _approval(seq: int, decision: Any, request: Any = None) -> Approval:
    return Approval(
        seq=seq,
        run_id=None,
        interrupt_id=f"int-{seq}",
        request=request,
        decision=decision,
        line=seq,
    )


def _episode(
    kind: str,
    seqs: Sequence[int],
    *,
    thread_id: str = "thread-t",
    tools: Sequence[str] = ("bash",),
    facts: dict[str, Any] | None = None,
    optional: Sequence[int] = (),
    snippets: dict[int, str] | None = None,
) -> Episode:
    """An episode of ``<thread_id>.jsonl`` citing ``seqs`` (line == seq); ``optional`` seqs are context."""
    source = f"{thread_id}.jsonl"
    return Episode(
        kind=kind,  # type: ignore[arg-type]
        thread_id=thread_id,
        source=source,
        evidence=[
            EvidenceItem(
                EvidenceRef(source=source, line=seq, seq=seq),
                "event",
                seq not in optional,
                (snippets or {}).get(seq),
            )
            for seq in seqs
        ],
        tool_sequence=list(tools),
        facts={"tool": "bash"} if facts is None else facts,
        weight=EPISODE_WEIGHTS[kind],
    )


def _sample() -> Trajectory:
    """seq 2 user turn; 4/5 failed bash; 6 ai; 8/9 ok bash; 12 approval."""
    calls = [
        _call(
            "bash",
            {"cmd": "make"},
            seq=4,
            status="error",
            error_type="CalledProcessError",
            error="exit 2\n  missing dep",
        ),
        _call("bash", {"cmd": "make deps && make"}, seq=8, output="ok  built\n\n42 targets"),
    ]
    turn = _turn(
        "run-1",
        2,
        "please   build it",
        calls,
        ai=[_ai(6, "I will build", ["bash"])],
        approvals=[_approval(12, {"decisions": [{"type": "reject"}]}, {"tool": "rm"})],
    )
    return _traj(turn)


def _blocks(text: str) -> list[str]:
    return text.split("\n\n")


def _headers(text: str) -> list[re.Match[str]]:
    matches = [HEADER_RE.match(block.splitlines()[0]) for block in _blocks(text)]
    assert all(matches), text
    return matches  # type: ignore[return-value]


def _excerpts(block: str) -> list[str]:
    lines = block.splitlines()
    return lines[lines.index("Excerpts:") + 1 :]


def _shown_seqs(bundle: EvidenceBundle) -> set[int]:
    return {fragment.ref.seq for fragment in bundle.shown.values()}


def _statuses(bundle: EvidenceBundle) -> list[str]:
    return [item.status for item in bundle.episodes]


def _seq_at(path: Path, line: int) -> int:
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            if number == line:
                return json.loads(raw)["seq"]
    raise AssertionError(f"{path} has no line {line}")


# ------------------------------------------------------------------ ordering


def test_blocks_ordered_by_weight_and_numbered() -> None:
    episodes = [
        _episode("skill_gap", [2, 4]),  # 0.4
        _episode("approval_denied", [8, 12]),  # 1.0
        _episode("error_recovery", [4, 5, 8, 9]),  # 0.6
    ]

    bundle = build_evidence_bundle(episodes, [_sample()])

    headers = _headers(bundle.text)
    assert [m.group(1, 2, 3) for m in headers] == [
        ("1", "approval_denied", "1.00"),
        ("2", "error_recovery", "0.60"),
        ("3", "skill_gap", "0.40"),
    ]
    assert {m.group(4) for m in headers} == {"thread-t"}
    assert _shown_seqs(bundle) == {2, 4, 5, 8, 9, 12}
    assert list(bundle.shown) == [f"ev{n}" for n in range(1, 7)]
    assert _statuses(bundle) == ["shown"] * 3
    assert bundle.kept == [episodes[1], episodes[2], episodes[0]]


def test_equal_weights_keep_input_order() -> None:
    episodes = [_episode("retry_loop", [4, 8]), _episode("retry_loop", [2, 4])]

    bundle = build_evidence_bundle(episodes, [_sample()])

    first, second = _blocks(bundle.text)
    assert _excerpts(first)[0].startswith('- [ev1] tool.start bash: {"cmd": "make"}')
    assert _excerpts(second)[0] == '- [ev3] user: "please build it"'


def test_shared_event_has_one_id_across_blocks() -> None:
    episodes = [_episode("retry_loop", [4, 8]), _episode("retry_loop", [2, 4])]

    bundle = build_evidence_bundle(episodes, [_sample()])

    first, second = _blocks(bundle.text)
    assert _excerpts(first) == [
        '- [ev1] tool.start bash: {"cmd": "make"}',
        '- [ev2] tool.start bash: {"cmd": "make deps && make"}',
    ]
    assert _excerpts(second) == [
        '- [ev3] user: "please build it"',
        '- [ev1] tool.start bash: {"cmd": "make"}',
    ]
    assert len(bundle.shown) == 3
    assert bundle.shown["ev1"].ref == EvidenceRef(source="thread-t.jsonl", line=4, seq=4)


def test_thread_id_is_shortened_in_header() -> None:
    traj = _traj(_turn("run-1", 2, "hi"), thread_id="thread-0123456789")
    episode = _episode("skill_gap", [2], thread_id="thread-0123456789")

    bundle = build_evidence_bundle([episode], [traj])

    assert bundle.text.splitlines()[0] == "### Episode E1 — skill_gap (weight 0.40, thread thread-0)"


# -------------------------------------------------------------------- budget


def test_budget_trims_then_excludes() -> None:
    traj = _sample()
    light = _episode("skill_gap", [2, 4], optional=[4])
    heavy = _episode("approval_denied", [12, 8], optional=[8])
    full = build_evidence_bundle([light, heavy], [traj])
    first_block, second_block = _blocks(full.text)
    assert _statuses(full) == ["shown", "shown"]
    assert _shown_seqs(full) == {2, 4, 8, 12}

    exact = build_evidence_bundle([light, heavy], [traj], max_chars=len(full.text))
    assert (exact.text, set(exact.shown)) == (full.text, set(full.shown))

    trimmed = build_evidence_bundle([light, heavy], [traj], max_chars=len(full.text) - 1)
    assert _statuses(trimmed) == ["shown", "trimmed"]
    assert _blocks(trimmed.text)[0] == first_block
    assert _excerpts(_blocks(trimmed.text)[1]) == [
        '- [ev3] user: "please build it"',
        f"- {ELLIPSIS} 1 more events not shown",
    ]
    assert _shown_seqs(trimmed) == {2, 8, 12}  # seq 4 was not shown: not citable
    assert trimmed.kept == [heavy, light]

    excluded = build_evidence_bundle([light, heavy], [traj], max_chars=len(first_block))
    assert _statuses(excluded) == ["shown", "excluded"]
    assert excluded.text == first_block
    assert _shown_seqs(excluded) == {8, 12}
    assert excluded.kept == [heavy]
    assert second_block not in excluded.text


def test_excluded_heavy_episode_does_not_stop_scanning(caplog: pytest.LogCaptureFixture) -> None:
    heavy = _episode("approval_denied", [12], facts={"decision": "x" * 2000})
    light = _episode("skill_gap", [2], facts={})

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        bundle = build_evidence_bundle([light, heavy], [_sample()], max_chars=300)

    assert _statuses(bundle) == ["excluded", "shown"]
    assert bundle.text.startswith("### Episode E1 — skill_gap")
    assert list(bundle.shown) == ["ev1"]
    assert bundle.kept == [light]
    assert "excluded approval_denied episode of thread 'thread-t'" in caplog.text


def test_budget_too_small_excludes_everything() -> None:
    bundle = build_evidence_bundle([_episode("approval_denied", [12])], [_sample()], max_chars=10)

    assert (bundle.text, bundle.shown, _statuses(bundle)) == ("", {}, ["excluded"])
    assert bundle.kept == []


@pytest.mark.parametrize("max_chars", [0, -1])
def test_non_positive_budget_raises(max_chars: int) -> None:
    with pytest.raises(ValueError, match="max_chars must be positive"):
        build_evidence_bundle([], [_sample()], max_chars=max_chars)


def test_empty_episodes_give_empty_bundle() -> None:
    assert build_evidence_bundle([], [_sample()]) == EvidenceBundle("", {}, [])
    assert build_evidence_bundle([], []) == EvidenceBundle("", {}, [])


# ------------------------------------------------------------------ excerpts


def test_excerpts_come_from_the_cited_records() -> None:
    episode = _episode("error_recovery", [2, 4, 5, 6, 8, 9, 12], facts={"tool": "bash", "calls_between": 2})

    bundle = build_evidence_bundle([episode], [_sample()])

    lines = bundle.text.splitlines()
    assert "Evidence: seq" not in bundle.text
    assert lines[1] == "Tools: bash"
    assert lines[2:5] == ["Facts:", '- tool: "bash"', "- calls_between: 2"]
    assert lines[5] == "Excerpts:"
    assert lines[6:] == [
        '- [ev1] user: "please build it"',
        '- [ev2] tool.start bash: {"cmd": "make"}',
        "- [ev3] tool.error bash (error): CalledProcessError: exit 2 missing dep",
        '- [ev4] ai: "I will build" [tool calls: bash]',
        '- [ev5] tool.start bash: {"cmd": "make deps && make"}',
        "- [ev6] tool.result bash (ok): ok built 42 targets",
        '- [ev7] approval.decision: request={"tool": "rm"} decision={"decisions": [{"type": "reject"}]}',
    ]
    fragment = bundle.shown["ev3"]
    assert fragment.text == "tool.error bash (error): CalledProcessError: exit 2 missing dep"
    assert fragment.ref == EvidenceRef(source="thread-t.jsonl", line=5, seq=5)
    assert (fragment.role, fragment.required) == ("event", True)
    assert all(f"- [{fid}] {fragment.text}" in bundle.text for fid, fragment in bundle.shown.items())


def test_turn_without_message_orphan_and_empty_output() -> None:
    turn = _turn(
        "run-2",
        20,
        None,
        [_call("grep", {"q": "x"}, seq=22, status="orphan"), _call("ls", seq=24, output="")],
        source="resume",
    )
    episode = _episode("retry_loop", [20, 22, 24, 25], tools=[])

    bundle = build_evidence_bundle([episode], [_traj(turn)])

    assert "Tools:" not in bundle.text
    assert _excerpts(bundle.text) == [
        "- [ev1] turn.start: (source=resume)",
        '- [ev2] tool.start grep: {"q": "x"}',
        "- [ev3] tool.start ls: {}",
        "- [ev4] tool.result ls (ok): (no output)",
    ]


def test_start_less_call_has_only_a_result_record() -> None:
    turn = _turn("run-1", 2, "go", [_call("bash", seq=4, seq_end=4, output="done")])

    bundle = build_evidence_bundle([_episode("retry_loop", [4])], [_traj(turn)])

    assert bundle.text.splitlines()[-1] == "- [ev1] tool.result bash (ok): done"


def test_error_in_tool_result_output_is_shown() -> None:
    # The error came back as tool.result status=error: the text is in output_text.
    trace = "Traceback (most recent call last):\n  File x\nKeyError: 'device'"
    failed = _call("bash", {"cmd": "run"}, seq=4, status="error", output=trace)

    bundle = build_evidence_bundle([_episode("retry_loop", [5])], [_traj(_turn("run-1", 2, "go", [failed]))])

    assert bundle.text.splitlines()[-1] == (
        "- [ev1] tool.error bash (error): error: Traceback (most recent call last): File x KeyError: 'device'"
    )


def test_snippet_replaces_the_record_head() -> None:
    turn = _turn("run-1", 2, "x" * 400 + " no, do it instead")
    snippet = f"{ELLIPSIS}xxxx no, do it instead"
    episode = _episode("user_correction", [2], snippets={2: snippet})

    bundle = build_evidence_bundle([episode], [_traj(turn)])

    assert bundle.text.splitlines()[-1] == f"- [ev1] user: {snippet}"
    assert bundle.shown["ev1"].text == f"user: {snippet}"


def test_correction_snippet_reaches_the_bundle() -> None:
    # The correcting phrase sits after the first 600 characters of the message.
    message = "context " * 80 + "不对：首先运行性能分析器" + " tail" * 20
    traj = _traj(
        _turn("run-1", 2, "do it", [_call("bash", {}, seq=4)]),
        _turn("run-2", 10, message, []),
    )
    assert message.index("不对") > 600

    bundle = build_evidence_bundle(extract_episodes(traj), [traj])

    (correction,) = [f for f in bundle.shown.values() if f.role == "correction"]
    assert "不对：首先运行性能分析器" in correction.text
    assert correction.text.startswith(f"user: {ELLIPSIS}")
    assert correction.text in bundle.text


def test_args_diff_window_is_in_facts_and_excerpt() -> None:
    # The changed part of a long argument sits at its end.
    old = "python train.py " + "--flag=value " * 40 + "--device cpu"
    new = old[: -len("cpu")] + "npu"
    failed = _call("bash", {"cmd": old}, seq=4, status="error", error="no cpu")
    fixed = _call("bash", {"cmd": new}, seq=6, output="ok")
    traj = _traj(_turn("run-1", 2, "train", [failed, fixed]))

    bundle = build_evidence_bundle(extract_episodes(traj), [traj])

    (facts_line,) = [line for line in bundle.text.splitlines() if line.startswith("- args_diff:")]
    assert f'"new": "{ELLIPSIS}' in facts_line and "--device npu" in facts_line
    assert "--device cpu" in facts_line
    assert "python train.py" not in facts_line  # the unchanged head is not repeated


def test_excerpt_and_fact_clipping() -> None:
    turn = _turn("run-1", 2, "word " * 400)
    long_decision = {"comment": "x" * 5000}
    episode = _episode("approval_denied", [2], facts={"decision": long_decision})

    bundle = build_evidence_bundle([episode], [_traj(turn)])

    fact_line, excerpt_line = bundle.text.splitlines()[-3], bundle.text.splitlines()[-1]
    assert fact_line.startswith('- decision: {"comment": "xxx')
    assert fact_line.endswith(ELLIPSIS)
    assert len(fact_line) == len("- decision: ") + FACT_LIMIT
    assert excerpt_line.startswith('- [ev1] user: "word word word')
    assert excerpt_line.endswith(ELLIPSIS)
    assert len(excerpt_line) == len("- [ev1] user: ") + EXCERPT_LIMIT


def test_minimal_block_shows_required_only_and_counts_omitted() -> None:
    turns = [_turn(f"run-{i}", 2 * i, f"message {i}") for i in range(1, 21)]
    seqs = [turn.seq_start for turn in turns]
    episode = _episode("skill_gap", seqs, optional=seqs[1:])
    full = build_evidence_bundle([episode], [_traj(*turns)])
    assert len(full.shown) == 20

    bundle = build_evidence_bundle([episode], [_traj(*turns)], max_chars=len(full.text) - 1)

    assert _statuses(bundle) == ["trimmed"]
    assert _excerpts(bundle.text) == ['- [ev1] user: "message 1"', f"- {ELLIPSIS} 19 more events not shown"]
    assert list(bundle.shown) == ["ev1"]
    assert bundle.shown["ev1"].required is True


def test_excerpt_chars_is_configurable_and_guarded() -> None:
    turn = _turn("run-1", 2, "word " * 400)
    episode = _episode("approval_denied", [2])

    bundle = build_evidence_bundle([episode], [_traj(turn)], excerpt_chars=50)

    excerpt_line = bundle.text.splitlines()[-1]
    assert excerpt_line.startswith('- [ev1] user: "word word')
    assert excerpt_line.endswith(ELLIPSIS)
    assert len(excerpt_line) == len("- [ev1] user: ") + 50
    assert bundle.shown["ev1"].text == excerpt_line[len("- [ev1] ") :]
    # The default is the module constant (see test_excerpt_and_fact_clipping).
    default = build_evidence_bundle([episode], [_traj(turn)])
    assert len(default.text.splitlines()[-1]) == len("- [ev1] user: ") + EXCERPT_LIMIT
    assert build_evidence_bundle([episode], [_traj(turn)], excerpt_chars=EXCERPT_LIMIT) == default
    for bad in (len(ELLIPSIS), 0, -5):
        with pytest.raises(ValueError, match=f"excerpt_chars must be greater than {len(ELLIPSIS)}, got {bad}"):
            build_evidence_bundle([episode], [_traj(turn)], excerpt_chars=bad)


# ---------------------------------------------------------------------- demo


def _demo_traj() -> Trajectory:
    """One ok chain (read_file, bash, bash) and a denied approval in one turn."""
    calls = [
        _call("read_file", {"path": "input/sales.csv"}, seq=4, output="id,amount\n1,10\n2,20\n3,30\n"),
        _call("bash", {"cmd": "python3 sum.py input/sales.csv amount"}, seq=6, output="60.0"),
        _call("bash", {"cmd": "python3 check.py input/sales.csv amount 60"}, seq=8, output="TOTAL_OK"),
    ]
    approval = _approval(12, {"decisions": [{"type": "reject"}]}, {"tool": "rm"})
    return _traj(_turn("run-1", 2, "sum it and check", calls, approvals=[approval]))


def test_demo_ranks_observed_procedure_first_and_prints_its_line() -> None:
    traj = _demo_traj()
    episodes = extract_episodes(traj, demo=True)
    assert [e.kind for e in episodes] == ["approval_denied", "observed_procedure"]
    denial, observed = episodes

    demo = build_evidence_bundle(episodes, [traj], demo=True)

    first, second = _blocks(demo.text)
    assert [m.group(1, 2, 3) for m in _headers(demo.text)] == [
        ("1", "observed_procedure", "0.00"),
        ("2", "approval_denied", "1.00"),
    ]
    assert first.splitlines()[1] == OBSERVED_LINE.format(calls=3)
    assert first.splitlines()[1] == (
        "Observed procedure: 3 call(s) in one execution context (selected by code); "
        "ok status is not proof of task success"
    )
    assert "Observed procedure:" not in second
    assert _excerpts(first) == [
        '- [ev1] user: "sum it and check"',
        '- [ev2] tool.start read_file: {"path": "input/sales.csv"}',
        "- [ev3] tool.result read_file (ok): id,amount 1,10 2,20 3,30",
        '- [ev4] tool.start bash: {"cmd": "python3 sum.py input/sales.csv amount"}',
        "- [ev5] tool.result bash (ok): 60.0",
        '- [ev6] tool.start bash: {"cmd": "python3 check.py input/sales.csv amount 60"}',
        "- [ev7] tool.result bash (ok): TOTAL_OK",
    ]
    assert demo.kept == [observed, denial]
    assert [(item.status, item.episode_id) for item in demo.episodes] == [("shown", "E1"), ("shown", "E2")]
    assert demo.episode_ids == {"E1": observed, "E2": denial}
    assert list(demo.episode_ids) == ["E1", "E2"]

    # Without demo the weight order stands and the zero-weight chain comes last.
    plain = build_evidence_bundle(episodes, [traj])
    assert [m.group(2) for m in _headers(plain.text)] == ["approval_denied", "observed_procedure"]
    assert plain.episode_ids == {"E1": denial, "E2": observed}
    assert _blocks(plain.text)[1].splitlines()[1] == OBSERVED_LINE.format(calls=3)

    # An excluded episode has no id and is absent from episode_ids.
    excluded = build_evidence_bundle(episodes, [traj], demo=True, max_chars=len(first))
    assert [(item.status, item.episode_id) for item in excluded.episodes] == [("shown", "E1"), ("excluded", None)]
    assert excluded.episode_ids == {"E1": observed}
    assert EXCLUDED_CODE == "insufficient_context_budget"
    # Positional construction and the default of the new field keep working.
    assert EvidenceBundle("", {}, []).episode_ids == {}


def test_demo_fixture_bundle_resolves_and_is_complete() -> None:
    path = FIXTURES / "skill_evolver_demo_success.jsonl"
    traj = load_trajectory(path)
    (episode,) = extract_episodes(traj, skill_index=BM25Index(DOCS), demo=True)

    bundle = build_evidence_bundle([episode], [traj], demo=True)

    assert _statuses(bundle) == ["shown"] and len(_blocks(bundle.text)) == 1
    assert bundle.episode_ids == {"E1": episode}
    assert {fragment.role for fragment in bundle.shown.values()} == {"task", "step", "result"}
    assert all(fragment.required for fragment in bundle.shown.values())
    for fragment in bundle.shown.values():
        assert fragment.ref.source == path.name
        assert _seq_at(path, fragment.ref.line) == fragment.ref.seq
    excerpts = _excerpts(bundle.text)
    assert excerpts[0] == (
        '- [ev1] user: "Compute the sum of the amount column in input/sales.csv and check that the total is 60"'
    )
    results = [line for line in excerpts if "tool.result" in line]
    assert len(results) == 3
    assert results[1].endswith("tool.result bash (ok): 60.0")
    assert results[2].endswith("tool.result bash (ok): TOTAL_OK")
    assert "ai:" not in bundle.text  # the assistant's text is never shown as evidence


# ---------------------------------------------------------------- collisions


@pytest.mark.parametrize(("run_id", "source"), [("run-x", "unknown"), (PRELUDE_RUN_ID, "prelude")])
def test_synthetic_turn_never_shadows_the_real_record(run_id: str, source: str) -> None:
    # The reader opens such turns at the line of the first event it routes there.
    turn = _turn(run_id, 4, None, [_call("bash", {"cmd": "ls"}, seq=4)], source=source)

    bundle = build_evidence_bundle([_episode("retry_loop", [4, 5])], [_traj(turn)])

    assert bundle.text.splitlines()[-2:] == [
        '- [ev1] tool.start bash: {"cmd": "ls"}',
        "- [ev2] tool.result bash (ok): (no output)",
    ]


def test_seq_repeated_after_restart_gets_distinct_refs(caplog: pytest.LogCaptureFixture) -> None:
    # Two writers (recorder restart) both counted up to seq 4 in one thread;
    # the second writer's events sit on later physical lines.
    first = _turn("run-1", 2, "one", [_call("bash", {"n": 1}, seq=4)])
    later_call = replace(_call("grep", {"n": 2}, seq=4), line_start=24, line_end=25)
    second = replace(_turn("run-2", 2, "two", [later_call]), line_start=22)
    source = "thread-t.jsonl"
    episode = Episode(
        kind="retry_loop",
        thread_id="thread-t",
        source=source,
        evidence=[
            EvidenceItem(EvidenceRef(source=source, line=4, seq=4), "attempt", True),
            EvidenceItem(EvidenceRef(source=source, line=24, seq=4), "attempt", True),
        ],
        tool_sequence=["bash", "grep"],
        facts={},
        weight=0.7,
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        bundle = build_evidence_bundle([episode], [_traj(first, second)])

    assert _excerpts(bundle.text) == [
        '- [ev1] tool.start bash: {"n": 1}',
        '- [ev2] tool.start grep: {"n": 2}',
    ]
    assert [(f.ref.line, f.ref.seq) for f in bundle.shown.values()] == [(4, 4), (24, 4)]
    assert caplog.text == ""


def test_duplicate_source_keeps_the_first_copy() -> None:
    first = _traj(_turn("run-1", 2, "first copy"))
    second = _traj(_turn("run-1", 2, "second copy"))

    bundle = build_evidence_bundle([_episode("skill_gap", [2])], [first, second])

    assert bundle.text.splitlines()[-1] == '- [ev1] user: "first copy"'


# -------------------------------------------------------------------- errors


def test_unknown_source_raises() -> None:
    episode = _episode("skill_gap", [2], thread_id="thread-other")

    with pytest.raises(ValueError, match="cites source 'thread-other.jsonl', which is not among"):
        build_evidence_bundle([episode], [_sample()])


def test_unresolved_ref_raises() -> None:
    turns = [_turn(f"run-{i}", 2 * i, f"message {i}") for i in range(1, 21)]
    seqs = [turn.seq_start for turn in turns]
    seqs[10] = 21
    episode = _episode("skill_gap", seqs)

    with pytest.raises(ValueError, match=r"cites \['thread-t.jsonl:21'\], which is not in its trajectory"):
        build_evidence_bundle([episode], [_traj(*turns)])


def test_unresolved_ref_is_reported_regardless_of_budget() -> None:
    light = _episode("skill_gap", [2, 99])
    heavy = _episode("approval_denied", [12])

    with pytest.raises(ValueError, match=r"cites \['thread-t.jsonl:99'\]"):
        build_evidence_bundle([light, heavy], [_sample()], max_chars=10**6)


# ------------------------------------------------------------------ fixtures


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=FIXTURE_IDS)
def test_fixture_episodes_render(path: Path) -> None:
    traj = load_trajectory(path)
    episodes = extract_episodes(traj, skill_index=BM25Index(DOCS))

    bundle = build_evidence_bundle(episodes, [traj])

    assert _statuses(bundle) == ["shown"] * len(episodes)
    assert len(_blocks(bundle.text)) == (len(episodes) if episodes else 1)
    assert bundle.text.count("### Episode E") == len(episodes)
    assert "Evidence: seq" not in bundle.text
    cited = re.findall(r"\[(ev\d+)\]", bundle.text)
    assert set(cited) == set(bundle.shown)
    assert list(bundle.shown) == [f"ev{n}" for n in range(1, len(bundle.shown) + 1)]
    for fragment_id, fragment in bundle.shown.items():
        assert f"- [{fragment_id}] {fragment.text}" in bundle.text
    assert {item.ref for e in episodes for item in e.evidence} == {f.ref for f in bundle.shown.values()}


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=FIXTURE_IDS)
def test_shown_refs_resolve_to_physical_lines(path: Path) -> None:
    # Includes malformed_lines.jsonl: corrupted lines never shift a ref.
    traj = load_trajectory(path)
    bundle = build_evidence_bundle(extract_episodes(traj, skill_index=BM25Index(DOCS)), [traj])

    for fragment in bundle.shown.values():
        assert fragment.ref.source == path.name
        assert _seq_at(path, fragment.ref.line) == fragment.ref.seq


def test_cross_session_episodes_render() -> None:
    trajs = [load_trajectory(path) for path in FIXTURE_FILES]
    episodes = mine_cross_session(trajs)
    assert episodes  # the fixtures share procedures (see test_features)

    bundle = build_evidence_bundle(episodes, trajs)

    assert _statuses(bundle) == ["shown"] * len(episodes)
    blocks = _blocks(bundle.text)
    paths = {traj.source: traj.path for traj in trajs}
    for block, item in zip(blocks, bundle.episodes):
        episode = item.episode
        support = episode.facts["support"]
        assert block.splitlines()[1] == f"Support: {support} threads (counted by code); excerpts from 2 of them"
        excerpts = _excerpts(block)
        # Owner steps with their results (a start-less owner has start == end,
        # so its result refs collapse into the step refs), the second
        # thread's steps, then the owner's own task line.
        n = len(episode.tool_sequence)
        assert len(excerpts) in (2 * n + 1, 3 * n + 1)
        own, other, task = excerpts[: -n - 1], excerpts[-n - 1 : -1], excerpts[-1]
        assert not any("(thread " in line for line in own)
        assert all("] (thread " in line for line in other)
        assert "] user:" in task and "(thread " not in task
    for fragment in bundle.shown.values():
        assert _seq_at(paths[fragment.ref.source], fragment.ref.line) == fragment.ref.seq
