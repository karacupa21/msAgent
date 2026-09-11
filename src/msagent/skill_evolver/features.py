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

"""Deterministic knowledge-candidate extraction from recorded trajectories.

Everything here is computed by code over the typed trajectory model, without
an LLM: detectors turn recurring session patterns (a failed tool call fixed
by changed arguments, a user correcting the agent, a retry loop, a denied
approval, a procedure shared by several sessions, a skill that should have
been consulted) into :class:`Episode` records whose ``evidence`` items point
at real events of the source JSONL by :class:`EvidenceRef` (file name and
physical line — ``seq`` restarts per recorder writer and is display only).

Rules of FEATURES_VERSION 4:

- Calls are compared inside one *execution context*: a turn group (a turn
  plus the ``resume`` turns that continue it — the only continuation the
  recorder can express) split by subagent. Nothing is matched across a
  following ``dispatch`` turn or between subagents.
- Context items: ``error_recovery``, ``retry_loop`` and
  ``repeated_procedure`` cite the user message that opened their turn group
  as a non-required ``task`` item; ``retry_loop`` and ``repeated_procedure``
  cite the result or error of their calls. Context never changes weights or
  anchors. ``Episode.primary_seq`` (the smallest required own seq) is the
  identity consumers key on; ``evidence_seq`` is display.
- ``observed_procedure`` (weight 0.0, extracted only when the caller asks
  for demo evidence): at most OBSERVED_MAX_CHAINS chains per trajectory,
  each at most OBSERVED_MAX_CALLS consecutive ``ok`` domain calls of one
  stream, windows cut from the start of a run. A chain needs the group's
  user message, ``tool.start`` for every call, no ERROR_MARKERS text in any
  output and a non-empty last output. AI text is never evidence; ``status:
  ok`` alone is not proof — the observed output is.
- Every kind has REQUIRED_ROLES; :func:`has_required_evidence` says whether
  an episode is a usable package. ``gate_decision(demo=True)`` admits a
  thread with reason ``demo_override`` only when the ordinary rules fail and
  at least one episode is complete; score and weights stay honest.
- :func:`expand_episode_context` adds non-required neighbours
  (``surrounding_events`` calls on each side inside the same stream), the
  missing results of cited calls and the task; it never invents events.
- Arguments are compared in normalized form (volatile keys dropped, paths
  and whitespace normalized) and a *work object* — the path or the command
  a call operates on — tells one operation from another.
- Approval decisions are read by structure (:func:`classify_approval`); a
  shape that cannot be read is ``unknown``, never an assumed denial.
- Every episode names the events that make it what it is as
  :class:`EvidenceItem` records with a role; the ``required`` ones are the
  minimum without which the episode cannot be shown to the LLM at all
  (a recovery: the error, the changed call and its result; a correction:
  the correcting message; a retry: its first and last attempt; a denial:
  the decision; a shared procedure: its steps in two sessions; a skill gap:
  the first domain call and the first user message).
- Text copied into facts or snippets is cut around what matters — the
  correction marker, the changed part of an argument, the head and tail of
  an error — and every cut is marked with ``…``. Nothing absent is
  reconstructed: an empty result stays empty, an orphan stays ``orphan``.
- Episodes about the same events share ``anchors`` and form one *incident*.
  :func:`evidence_score` counts the heaviest episode of each incident and
  adds incidents up, so several detectors describing one chain do not pile
  up; :func:`gate_decision` says why a thread passes or skips the
  ``min_evidence_score`` gate. Passing the gate only admits the thread to
  the LLM stage; whether a rule is worth keeping is decided there.

Stdlib only: importing this module must not load langchain.
"""

from __future__ import annotations

import json
import logging
import os.path
import posixpath
import re
import shlex
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field, replace
from typing import Any, Literal, get_args

from msagent.skill_evolver.retrieval import BM25Index
from msagent.trajectory_recorder.model import (
    PRELUDE_RUN_ID,
    EvidenceRef,
    ToolCall,
    Trajectory,
    Turn,
)

logger = logging.getLogger(__name__)

EpisodeKind = Literal[
    "error_recovery",
    "user_correction",
    "retry_loop",
    "approval_denied",
    "repeated_procedure",
    "skill_gap",
    "observed_procedure",
]
EPISODE_KINDS: frozenset[str] = frozenset(get_args(EpisodeKind))
# Version of the detector rules and weights, recorded in the provenance of
# every proposal; bump it when a rule or a weight changes.
FEATURES_VERSION = 5

# Gate threshold when the skill evolver config does not set
# min_evidence_score; direct_skill_generation re-exports it. The strong
# correction rule of gate_decision() applies at this threshold or below.
DEFAULT_MIN_EVIDENCE_SCORE = 1.0

# Evidence weight of one episode of each kind (0.0..1.0).
EPISODE_WEIGHTS: dict[str, float] = {
    "error_recovery": 0.6,
    "user_correction": 0.9,
    "retry_loop": 0.7,
    "approval_denied": 1.0,
    "repeated_procedure": 1.0,
    "skill_gap": 0.4,
    # Demo evidence: a confirmed chain of ok calls carries no score at all.
    "observed_procedure": 0.0,
}
# Weight of a user_correction carrying only a weak marker: an observation
# for the LLM stage that never admits a thread on its own.
WEAK_CORRECTION_WEIGHT = 0.5

# observed_procedure (demo only): chains selected per trajectory and calls per chain.
OBSERVED_MAX_CHAINS = 3
OBSERVED_MAX_CALLS = 5

# Roles an episode must carry as *required* items to be a usable evidence package.
REQUIRED_ROLES: dict[str, frozenset[str]] = {
    "error_recovery": frozenset({"error", "fixed_call", "result"}),
    "user_correction": frozenset({"correction"}),
    "retry_loop": frozenset({"attempt"}),
    "approval_denied": frozenset({"approval"}),
    "repeated_procedure": frozenset({"step"}),
    "skill_gap": frozenset({"first_call", "user_message"}),
    "observed_procedure": frozenset({"task", "step", "result"}),
}

# Output text saying a call did not do its job although its status was ok.
# Each pattern is compiled on its own (an inline (?i) is per pattern) and
# matched anywhere; no bare "error" word, which would flag "0 errors".
ERROR_MARKERS: tuple[str, ...] = (
    r"Traceback \(most recent call last\)",
    r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b",
    r"(?i)\berror:",
    r"(?i)\bfatal:",
    r"\bFAILED\b",
    r"No such file or directory",
    r"command not found",
    r"Permission denied",
    r"(?i)\b(?:exit code|exit status)\s+[1-9]\d*\b",
    r"(?i)\bnon-zero exit\b",
    # A JSON error field with a non-empty text value: how MCP tools report a
    # failure inside an ok result ({"error": "SQL_EXECUTION_FAILED", ...}).
    r'"error"\s*:\s*"(?=[^"])',
    # An upper-case failure code; \bFAILED\b does not match after "_".
    r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_FAILED\b",
)
_ERROR_MARKER_RES = tuple(re.compile(pattern) for pattern in ERROR_MARKERS)

# Output text saying a search found nothing. Together with an empty output it
# makes the next change of a search key a forced retry (retry_loop) rather
# than a refinement after a hit.
NO_RESULT_MARKERS: tuple[str, ...] = (
    r"(?i)\bno (?:matches|results|files)\b",
    r"(?i)\bnothing found\b",
    r"(?i)\b0 (?:matches|results|rows)\b",
)
_NO_RESULT_MARKER_RES = tuple(re.compile(pattern) for pattern in NO_RESULT_MARKERS)

# Why the observed_procedure detector left a chain out (DetectorNote.reason).
DROP_NO_COMPLETED_CALLS = "no_completed_calls"
DROP_NO_TASK = "no_task_context"
DROP_MISSING_START = "missing_tool_start"
DROP_EMPTY_RESULT = "empty_result"
DROP_ERROR_IN_OUTPUT = "error_in_output"
DROP_CHAIN_LIMIT = "chain_limit"
DROP_DUPLICATE = "duplicate_chain"

# Explicit corrective instructions (en + zh), matched case-insensitively as
# substrings of the user message; with an observed change of the agent's
# actions they make a strong correction.
STRONG_CORRECTION_MARKERS: tuple[str, ...] = (
    "不对",
    "不，",
    "应该先",
    "本来应该",
    "错了",
    "no, ",
    "should have",
    "instead",
)
# Hedges that may or may not correct anything; alone they never make a
# strong correction.
WEAK_CORRECTION_MARKERS: tuple[str, ...] = (
    "actually",
    "首先",
    "其实",
    "rather",
)
# Negations correct only when they open the message ("不，不对");
# elsewhere ("先看日志，不，先看配置") they are ordinary text.
OPENING_MARKERS: frozenset[str] = frozenset({"不，", "no, "})
# Han characters (CJK Unified Ideographs and Ext. A); Chinese is not
# space-delimited, so a marker starting with one carries no word-start guard.
_HAN_RE = re.compile(r"[㐀-䶿一-鿿]")

# Catalog / introspection tools: not domain work, so they never count as
# evidence that a session did something a skill describes and are not steps
# of a shared procedure.
CATALOG_TOOLS: frozenset[str] = frozenset(
    {"get_skill", "fetch_skills", "get_tool", "fetch_tools", "run_tool"},
)

# Argument keys naming the object a call works on, in lookup order. A
# command's working directory is context, not its object, so "cwd" is absent.
PATH_KEYS: tuple[str, ...] = (
    "path",
    "file_path",
    "file",
    "filename",
    "directory",
    "dir",
    # A database file is the object of an SQL tool's call.
    "db_path",
    "database",
)
# Argument keys holding a shell command; "input" is the legacy wrapper the
# reader puts around a non-dict tool input.
COMMAND_KEYS: tuple[str, ...] = ("command", "cmd", "script", "input")
# Argument keys that may hold an SQL statement. A value counts as SQL only
# when it starts with one of _SQL_VERBS: a search tool's ``query`` is a pattern.
SQL_KEYS: tuple[str, ...] = ("sql", "query", "statement")
_SQL_VERBS: frozenset[str] = frozenset(
    {"select", "insert", "update", "delete", "pragma", "with", "create", "drop", "alter", "explain", "attach"},
)
_SQL_TABLE_RE = re.compile(r"\b(?:from|into|update|table|join)\s+[`\"']?([A-Za-z_][\w.]*)", re.IGNORECASE)
# Shell command segments are split on these operators; a segment whose program
# is ``cd`` names no step, and the tokens below never name the program.
_SHELL_SEGMENT_RE = re.compile(r"&&|\|\||[;|]")
_SHELL_PREFIX_TOKENS: frozenset[str] = frozenset({"sudo", "time", "env", "exec", "nohup"})
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Keys whose change between attempts means "searching again for a result".
SEARCH_KEYS: frozenset[str] = frozenset({"pattern", "query", "regex"})
# Keys that vary without changing what a call does; dropped before comparing.
VOLATILE_KEYS: frozenset[str] = frozenset({"offset", "limit", "timeout", "timeout_ms"})

# Legacy question/options interrupts answer with the chosen option string;
# only these exact answers are read, anything else is unknown.
LEGACY_DENIAL_ANSWERS: frozenset[str] = frozenset(
    {"reject", "rejected", "deny", "denied", "no", "n"},
)
LEGACY_APPROVAL_ANSWERS: frozenset[str] = frozenset(
    {"approve", "approved", "accept", "accepted", "yes", "y", "ok"},
)
# Decision types of the structured (deepagents / langchain HITL) shape.
DENYING_DECISIONS: frozenset[str] = frozenset({"reject"})
ACCEPTING_DECISIONS: frozenset[str] = frozenset({"approve", "edit", "respond"})

# A failed tool call counts as recovered when the same tool succeeds within
# this many subsequent tool calls of the same execution context.
RECOVERY_WINDOW = 5
# retry_loop: at least this many attempts at one operation.
RETRY_MIN_ATTEMPTS = 3
# Tool calls kept as context after a denied approval.
DENIAL_CONTEXT_CALLS = 3
# n-gram sizes mined by :func:`mine_cross_session`.
NGRAM_MIN = 2
NGRAM_MAX = 5
# skill_gap: minimal BM25 score of the best library match. One shared rare
# term scores ln(1 + (N - 0.5) / 1.5) in a library of N skills (0.98 at N=3,
# 1.39 at N=5, 3.5 at N=50), so libraries of four or more skills fire on a
# single distinctive shared term and tiny ones need two.
SKILL_GAP_MIN_SCORE = 1.0
# Longest text copied into ``Episode.facts`` or an evidence snippet; a cut
# is marked with ELLIPSIS and the marked text is never longer than this.
VALUE_LIMIT = 200
ELLIPSIS = "…"
# Error text keeps its head and its tail: a traceback ends with the exception.
ERROR_HEAD = 100
ERROR_TAIL = 180
# An observed outcome keeps its head and tail too (60 + ELLIPSIS + 139 = VALUE_LIMIT).
OUTCOME_HEAD = 60
OUTCOME_TAIL = 139
# Characters kept on each side of the changed part of an argument value.
DIFF_CONTEXT = 60
# Characters kept on each side of the correction marker of a user message.
MARKER_CONTEXT = 120

# Reasons of :func:`gate_decision`, printed by the dry run.
GATE_NO_EPISODES = "no episodes"
GATE_SCORE_REACHED = "score >= min_evidence_score"
GATE_SCORE_BELOW = "score < min_evidence_score"
GATE_STRONG_CORRECTION = "strong user correction"
# Demo mode: the ordinary rules failed but an episode is a complete package.
GATE_DEMO_OVERRIDE = "demo_override"
# GateDecision.detail when demo mode could not override a failing gate.
GATE_DEMO_NOT_APPLICABLE = "no episode has complete required evidence"


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One cited event of an episode.

    ``ref`` is the physical identity of the event; ``role`` says what the
    event is to this episode (kind-specific: ``error``, ``fixed_call``,
    ``correction``, ``attempt``, ``step``, ...). ``required`` marks the items
    without which the episode cannot be shown at all: the evidence bundle
    keeps them when it trims for budget and excludes the episode when even
    they do not fit. ``snippet`` is a content-bearing cut chosen by the
    detector (edges marked with ELLIPSIS); ``None`` means "show the head of
    the recorded event".
    """

    ref: EvidenceRef
    role: str
    required: bool
    snippet: str | None = None


@dataclass(frozen=True, slots=True)
class Episode:
    """One knowledge candidate mined from a trajectory.

    ``source`` is the file name of the episode's own thread and ``evidence``
    lists its cited events (never empty, refs unique, at least one required
    item and at least one item of ``source`` — a ``repeated_procedure`` also
    cites the steps of a second supporting thread). ``facts`` holds
    JSON-safe, kind-specific details. ``anchors`` are keys
    (``"<run_id>#<seq>"``) of the events the episode is *about* — its tool
    calls, its correcting turn, its approval — never of context-only events;
    episodes sharing an anchor form one incident. A trajectory-level
    observation (``skill_gap``) has no anchors.
    """

    kind: EpisodeKind
    thread_id: str
    source: str
    evidence: list[EvidenceItem]
    tool_sequence: list[str]
    facts: dict[str, Any]
    weight: float
    anchors: list[str] = field(default_factory=list)

    @property
    def evidence_seq(self) -> list[int]:
        """Sorted seqs of the own-thread events, for tables and displays.

        Seqs restart per recorder writer, so this is not an identity; cite
        events by ``EvidenceItem.ref``.
        """
        return sorted({item.ref.seq for item in self.evidence if item.ref.source == self.source})

    @property
    def primary_seq(self) -> int:
        """Smallest own-thread seq among the *required* items: the event the episode is about.

        Stable when context items (the task, neighbours) are added; exgraph
        keys Episode nodes by it. Falls back to ``evidence_seq[0]``.
        """
        seqs = [item.ref.seq for item in self.evidence if item.required and item.ref.source == self.source]
        return min(seqs) if seqs else self.evidence_seq[0]

    def __post_init__(self) -> None:
        if self.kind not in EPISODE_KINDS:
            raise ValueError(f"unknown episode kind {self.kind!r}")
        if not self.thread_id:
            raise ValueError(f"{self.kind} episode without thread_id")
        if not self.source:
            raise ValueError(f"{self.kind} episode without source")
        if not self.evidence:
            raise ValueError(f"{self.kind} episode without evidence")
        if any(not isinstance(item, EvidenceItem) for item in self.evidence):
            raise ValueError(f"invalid evidence in {self.kind}: {self.evidence!r}")
        refs = [item.ref for item in self.evidence]
        if len(set(refs)) != len(refs):
            raise ValueError(f"duplicate evidence in {self.kind}: {refs!r}")
        if not any(item.required for item in self.evidence):
            raise ValueError(f"{self.kind} episode without required evidence")
        if not any(ref.source == self.source for ref in refs):
            raise ValueError(f"{self.kind} episode cites no event of {self.source!r}")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight out of range in {self.kind}: {self.weight!r}")
        if any(not isinstance(a, str) or not a for a in self.anchors):
            raise ValueError(f"invalid anchors in {self.kind}: {self.anchors!r}")


@dataclass(frozen=True, slots=True)
class ApprovalVerdict:
    """What one recorded approval decision says, read by structure.

    ``denied`` lists the rejected actions as ``{"name", "args"}`` (name may
    be ``None`` when the request names no tool); ``reason`` says how the
    decision was read, or why it could not be.
    """

    status: Literal["denied", "approved", "unknown"]
    denied: list[dict[str, Any]]
    reason: str


@dataclass(frozen=True, slots=True)
class DetectorNote:
    """Why a detector left a candidate out: a diagnostic for reports and the dry run, never evidence."""

    detector: str
    reason: str
    detail: str
    # Own-thread seqs concerned (display only).
    seqs: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Whether a thread's episodes reach the LLM stage, and why.

    ``detail`` is set in demo mode only: which kinds carried complete
    required evidence, or why the override did not apply.
    """

    incidents: list[list[Episode]]
    score: float
    passes: bool
    reason: str
    detail: str = ""

    def describe(self) -> str:
        """``reason`` or ``reason; detail`` — the text the tables and the Gate line print."""
        return self.reason if not self.detail else f"{self.reason}; {self.detail}"


def has_required_evidence(episode: Episode) -> bool:
    """True when the episode's required items cover every role of REQUIRED_ROLES for its kind."""
    return REQUIRED_ROLES[episode.kind] <= {item.role for item in episode.evidence if item.required}


# ------------------------------------------------------------------ helpers

# A tool call with the turn it belongs to: ``ToolCall`` carries no run_id.
_Step = tuple[Turn, ToolCall]


def _flat_calls(traj: Trajectory) -> list[ToolCall]:
    """All tool calls in model order (turns in file order, spans in order).

    ``seq`` restarts when the recorder process restarts mid-thread, so it is
    never used to order calls across turns.
    """
    return [call for turn in traj.turns for call in turn.tool_calls]


def _groups(traj: Trajectory) -> list[list[Turn]]:
    """Turns grouped with their explicit continuations.

    A turn whose ``source`` is ``resume`` (the ``/threads`` path: no user
    message, a fresh run_id and no link field) continues the previous group;
    every other turn (``dispatch``, ``unknown``, ``prelude``) opens one.
    Adjacency plus ``source`` is the only continuation signal recorded.
    """
    groups: list[list[Turn]] = []
    for turn in traj.turns:
        if groups and turn.source == "resume":
            groups[-1].append(turn)
        else:
            groups.append([turn])
    return groups


def _streams(group: list[Turn]) -> list[list[_Step]]:
    """The group's calls split by subagent (``None`` = root), in model order.

    One stream is one execution context; detectors never compare calls of
    different streams.
    """
    streams: dict[str | None, list[_Step]] = {}
    for turn in group:
        for call in turn.tool_calls:
            streams.setdefault(call.subagent, []).append((turn, call))
    return list(streams.values())


def _anchor(run_id: str, seq: int) -> str:
    """Incident key of one event; run_id-scoped because seq restarts per writer."""
    return f"{run_id}#{seq}"


def _start(traj: Trajectory, call: ToolCall) -> EvidenceRef:
    """Ref of a call's ``tool.start`` (or of its result when the start is missing)."""
    return traj.event_ref(seq=call.seq_start, line=call.line_start)


def _end(traj: Trajectory, call: ToolCall) -> EvidenceRef | None:
    """Ref of a call's ``tool.result``/``tool.error``; ``None`` for an orphan."""
    if call.seq_end is None or call.line_end is None:
        return None
    return traj.event_ref(seq=call.seq_end, line=call.line_end)


def _turn_ref(traj: Trajectory, turn: Turn) -> EvidenceRef:
    """Ref of the event that opened a turn (its ``turn.start`` for real turns)."""
    return traj.event_ref(seq=turn.seq_start, line=turn.line_start)


def _end_item(
    traj: Trajectory,
    call: ToolCall,
    role: str,
    *,
    snippet: str | None = None,
    required: bool = True,
) -> list[EvidenceItem]:
    """An item for the end of ``call`` (required by default), or nothing for an orphan."""
    ref = _end(traj, call)
    return [] if ref is None else [EvidenceItem(ref, role, required, snippet)]


def _heads(traj: Trajectory) -> dict[int, Turn]:
    """Group head of every turn, keyed by ``id(turn)`` (``Turn`` is not hashable)."""
    return {id(turn): group[0] for group in _groups(traj) for turn in group}


def _task_item(traj: Trajectory, head: Turn, *, required: bool) -> list[EvidenceItem]:
    """The ``task`` item citing the user message that opened a turn group.

    Empty for the prelude and for a head without a message (a synthetic
    ``unknown`` turn, a message-less dispatch): an earlier group's message is
    never borrowed.
    """
    if head.run_id == PRELUDE_RUN_ID or not (head.user_message or "").strip():
        return []
    return [EvidenceItem(_turn_ref(traj, head), "task", required)]


def _clip_outcome(text: str) -> str:
    """The head and the tail of an observed output, cut marked (at most VALUE_LIMIT)."""
    return _clip_edges(text, head=OUTCOME_HEAD, tail=OUTCOME_TAIL)


def _error_marker(text: str) -> str | None:
    """The first ERROR_MARKERS match in ``text``, or ``None``."""
    for pattern in _ERROR_MARKER_RES:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _failed_call(call: ToolCall) -> bool:
    """Whether a call failed: by its recorded status, or by an ERROR_MARKERS text in an ``ok`` output.

    Tools that report a failure inside their result — an MCP tool answering
    ``{"error": "SQL_EXECUTION_FAILED", ...}``, a shell tool echoing a
    traceback — leave ``status == "ok"``; the marker is the only failure
    signal the record carries for them. An orphan is neither.
    """
    if call.status == "error":
        return True
    return call.status == "ok" and _error_marker(call.output_text or "") is not None


def _canonical(value: Any) -> str:
    """Stable JSON text used to compare and print argument values."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _text(value: Any) -> str:
    """A value as text: strings as they are, anything else as canonical JSON."""
    return value if isinstance(value, str) else _canonical(value)


def _clip(value: Any) -> Any:
    """JSON-safe, short copy of a value: text longer than VALUE_LIMIT is cut and marked."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = _text(value)
    if len(text) <= VALUE_LIMIT:
        return text
    return text[: VALUE_LIMIT - len(ELLIPSIS)] + ELLIPSIS


def _clip_edges(text: str, head: int = ERROR_HEAD, tail: int = ERROR_TAIL) -> str:
    """Keep the first ``head`` and last ``tail`` characters, marking the cut."""
    if len(text) <= head + tail + len(ELLIPSIS):
        return text
    return text[:head] + ELLIPSIS + text[-tail:]


def _window(text: str, start: int, end: int, context: int) -> str:
    """``text[start:end]`` with ``context`` characters around it; cut edges marked."""
    low, high = max(0, start - context), min(len(text), end + context)
    body = _clip(text[low:high])
    if low:
        body = ELLIPSIS + body
    if high < len(text) and not body.endswith(ELLIPSIS):
        body += ELLIPSIS
    return body


def _diff_window(old: str, new: str) -> tuple[str, str]:
    """The changed part of two texts with DIFF_CONTEXT around it, cuts marked.

    The common prefix and suffix are dropped, so a change at the end of a
    long argument stays visible instead of being clipped away with the head.
    """
    prefix = len(os.path.commonprefix([old, new]))
    suffix = len(os.path.commonprefix([old[prefix:][::-1], new[prefix:][::-1]]))
    return (
        _window(old, prefix, len(old) - suffix, DIFF_CONTEXT),
        _window(new, prefix, len(new) - suffix, DIFF_CONTEXT),
    )


def _clip_args(args: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments with every value clipped."""
    return {str(key): _clip(value) for key, value in args.items()}


def _changed_value(old: Any, new: Any) -> dict[str, Any]:
    """``{"old", "new"}`` of one changed argument: scalars as they are, text windowed on the change."""
    if all(value is None or isinstance(value, (bool, int, float)) for value in (old, new)):
        return {"old": old, "new": new}
    before, after = _diff_window(_text(old), _text(new))
    return {"old": before, "new": after}


def _args_diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Keys added, removed or changed between two argument dicts.

    Added and removed values are clipped; a changed text value is windowed
    around the change (:func:`_diff_window`).
    """
    added = {key: _clip(new[key]) for key in sorted(new.keys() - old.keys())}
    removed = {key: _clip(old[key]) for key in sorted(old.keys() - new.keys())}
    changed = {
        key: _changed_value(old[key], new[key])
        for key in sorted(old.keys() & new.keys())
        if _canonical(old[key]) != _canonical(new[key])
    }
    diff: dict[str, Any] = {}
    if added:
        diff["added"] = added
    if removed:
        diff["removed"] = removed
    if changed:
        diff["changed"] = changed
    return diff


def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Comparison form of tool arguments.

    VOLATILE_KEYS are dropped, whitespace in strings is collapsed and path
    values are normalized (``./a/b`` == ``a/b``); non-string values pass
    through unchanged.
    """
    normalized: dict[str, Any] = {}
    for key, value in args.items():
        if key in VOLATILE_KEYS:
            continue
        if isinstance(value, str):
            value = " ".join(value.split())
            if key in PATH_KEYS and value:
                value = posixpath.normpath(value)
        normalized[str(key)] = value
    return normalized


def _command_object(command: str) -> str:
    """The program of a shell command plus its first non-option token."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return ""
    operands = [token for token in tokens[1:] if not token.startswith("-")]
    return " ".join([tokens[0], *operands[:1]])


def _work_object(call: ToolCall) -> str | None:
    """What one call operates on, or ``None`` when nothing is known.

    The program and first operand of the first command-like argument
    (``msprof --collect train.py`` → ``msprof train.py``; a command wins over
    the directory it runs in); else the first path-like argument
    (normalized); else ``""`` — the tool itself, e.g. a search without a
    path. A call recorded without arguments has no object.
    """
    if not call.args:
        return None
    args = _normalize_args(call.args)
    for key in COMMAND_KEYS:
        value = args.get(key)
        if isinstance(value, str):
            return _command_object(value)
    for key in PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _command_heads(command: str) -> list[str]:
    """The program of every segment of a shell command, in order.

    Segments are split on ``&&``, ``||``, ``;`` and ``|``; leading
    environment assignments and the tokens of _SHELL_PREFIX_TOKENS are
    skipped; a ``cd`` segment names no program. The program is its base
    name (``/usr/bin/python3`` → ``python3``).
    """
    heads: list[str] = []
    for segment in _SHELL_SEGMENT_RE.split(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        program = next(
            (t for t in tokens if not _ENV_ASSIGNMENT_RE.match(t) and t not in _SHELL_PREFIX_TOKENS),
            None,
        )
        if program is None or program == "cd":
            continue
        heads.append(posixpath.basename(program) or program)
    return heads


def _sql_signature(text: str) -> str:
    """``VERB`` or ``VERB table`` of an SQL statement; ``""`` when ``text`` is not SQL."""
    tokens = text.split(None, 1)
    if not tokens or tokens[0].lower() not in _SQL_VERBS:
        return ""
    verb = tokens[0].upper()
    match = _SQL_TABLE_RE.search(text)
    return f"{verb} {match.group(1)}" if match else verb


def _step_key(call: ToolCall) -> str:
    """The comparison key of one procedure step: the tool name plus what the call does.

    A command argument contributes the program of each shell segment
    (``cd x && python3 -c …`` → ``execute:python3``, ``which sqlite3`` →
    ``execute:which``); an SQL argument contributes the statement's verb and
    first table (``execute_sql:SELECT sqlite_master``); every other call is
    keyed by its tool name alone. Keyed by names only, every run of the same
    shell or SQL tool was a "shared procedure" (v4).
    """
    args = _normalize_args(call.args) if call.args else {}
    for key in COMMAND_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            heads = _command_heads(value)
            return f"{call.name}:{'+'.join(heads)}" if heads else call.name
    for key in SQL_KEYS:
        value = args.get(key)
        if isinstance(value, str):
            signature = _sql_signature(value)
            if signature:
                return f"{call.name}:{signature}"
    return call.name


def _collapse(text: str) -> str:
    """Whitespace-collapsed text; the form markers are searched in and windows cut from."""
    return " ".join(text.split())


def _markers(text: str) -> list[tuple[str, int]]:
    """Correction markers found in the collapsed text with their positions, strong ones first.

    A marker must start a word (``internet,`` is not ``no,``) unless it opens
    with a Han character: Chinese is not space-delimited, so ``这不对`` does
    carry ``不对``. An OPENING_MARKERS negation must start the message.
    Matching is case-insensitive on the collapsed text itself (not a casefolded
    copy, whose length may differ), so positions index :func:`_collapse` output.
    """
    normalized = _collapse(text)
    found: list[tuple[str, int]] = []
    for marker in (*STRONG_CORRECTION_MARKERS, *WEAK_CORRECTION_MARKERS):
        if marker in OPENING_MARKERS:
            prefix = "^"
        else:
            prefix = "" if _HAN_RE.match(marker) else r"(?<!\w)"
        match = re.search(prefix + re.escape(marker), normalized, re.IGNORECASE)
        if match:
            found.append((marker, match.start()))
    return found


def _action(entry: Any) -> dict[str, Any]:
    """``{"name", "args"}`` of one requested action.

    Reads a deepagents ``action_requests`` entry (``name`` / ``args``) or a
    flat request (``tool`` / ``args``); anything else names no tool.
    """
    if not isinstance(entry, dict):
        return {"name": None, "args": {}}
    name = entry.get("name", entry.get("tool"))
    args = entry.get("args")
    return {
        "name": None if name is None else str(name),
        "args": _clip_args(args) if isinstance(args, dict) else {},
    }


def _decision_type(entry: Any) -> str | None:
    """Casefolded ``type`` (or ``action``) of one decision dict."""
    if not isinstance(entry, dict):
        return None
    value = entry.get("type", entry.get("action"))
    return value.strip().casefold() if isinstance(value, str) else None


def classify_approval(request: Any, decision: Any) -> ApprovalVerdict:
    """Read one recorded approval decision by its structure.

    - ``{"decisions": [...]}`` is matched by index against
      ``request["action_requests"]`` (the langchain HITL contract); one
      decision without such a list applies to the flat request itself.
      A length mismatch is ``unknown``.
    - A flat ``{"action": ...}`` / ``{"type": ...}`` dict applies to the
      request. ``reject`` denies; ``approve`` / ``edit`` / ``respond`` do not;
      any other type is ``unknown``.
    - A bare string (legacy question/options interrupt) must equal an
      answer of LEGACY_DENIAL_ANSWERS or LEGACY_APPROVAL_ANSWERS; free text
      is ``unknown``.
    - Anything else (``None``, numbers, lists) is ``unknown``.
    """
    if isinstance(decision, str):
        answer = decision.strip().casefold()
        if answer in LEGACY_DENIAL_ANSWERS:
            return ApprovalVerdict("denied", [_action(request)], "legacy answer")
        if answer in LEGACY_APPROVAL_ANSWERS:
            return ApprovalVerdict("approved", [], "legacy answer")
        return ApprovalVerdict("unknown", [], f"unrecognised answer {decision!r}")
    if not isinstance(decision, dict):
        return ApprovalVerdict("unknown", [], f"decision is {type(decision).__name__}")
    if "decisions" in decision:
        entries = decision["decisions"]
        if not isinstance(entries, list):
            return ApprovalVerdict("unknown", [], "decisions is not a list")
        if not entries:
            return ApprovalVerdict("unknown", [], "no decision recorded")
        requested = request.get("action_requests") if isinstance(request, dict) else None
        if isinstance(requested, list):
            actions = requested
        elif len(entries) == 1:
            actions = [request]
        else:
            count = len(entries)
            return ApprovalVerdict("unknown", [], f"{count} decisions without action_requests")
        if len(entries) != len(actions):
            detail = f"{len(entries)} decisions for {len(actions)} action_requests"
            return ApprovalVerdict("unknown", [], detail)
        reason = "structured decisions"
    else:
        entries, actions, reason = [decision], [request], "flat decision"
    denied: list[dict[str, Any]] = []
    for index, (entry, action) in enumerate(zip(entries, actions)):
        kind = _decision_type(entry)
        if kind in DENYING_DECISIONS:
            denied.append(_action(action))
        elif kind not in ACCEPTING_DECISIONS:
            return ApprovalVerdict("unknown", [], f"decision {index} has type {kind!r}")
    return ApprovalVerdict("denied" if denied else "approved", denied, reason)


def _episode(
    kind: EpisodeKind,
    traj: Trajectory,
    evidence: list[EvidenceItem],
    tool_sequence: list[str],
    facts: dict[str, Any],
    anchors: list[str],
    *,
    weight: float | None = None,
) -> Episode:
    """An episode of ``traj``; items citing one ref twice keep the first (a start-less call has start == end)."""
    unique: dict[EvidenceRef, EvidenceItem] = {}
    for item in evidence:
        unique.setdefault(item.ref, item)
    return Episode(
        kind=kind,
        thread_id=traj.thread_id,
        source=traj.source,
        evidence=list(unique.values()),
        tool_sequence=tool_sequence,
        facts=facts,
        weight=EPISODE_WEIGHTS[kind] if weight is None else weight,
        anchors=anchors,
    )


# ---------------------------------------------------------------- detectors


def _detect_error_recovery(traj: Trajectory) -> list[Episode]:
    """A failed call fixed by a later ``ok`` call of the same tool.

    The success must come within RECOVERY_WINDOW calls of the same stream
    (one turn group, one subagent): a following ``dispatch`` turn, another
    subagent and an orphan span never recover anything. The knowledge is the
    raw argument diff, so a recovery with identical arguments (a transient
    failure, or calls recorded without ``tool.start``) is not an episode.
    Two failures sharing one recovery yield two episodes: two diffs. Known
    imprecision: the tool name is the only link, so a success of the same
    tool on another work object is reported as a recovery too.

    Required evidence: the error (``tool.error``, or the ``tool.result``
    whose ``output_text`` carries it), the changed call and its result; the
    failed call's start and the group's user message (``task``) are context.
    """
    episodes: list[Episode] = []
    for group in _groups(traj):
        task = _task_item(traj, group[0], required=False)
        for stream in _streams(group):
            for index, (turn, failed) in enumerate(stream):
                if not _failed_call(failed):
                    continue
                window = stream[index + 1 : index + 1 + RECOVERY_WINDOW]
                for offset, (later, candidate) in enumerate(window):
                    if candidate.name != failed.name or candidate.status != "ok" or _failed_call(candidate):
                        continue
                    diff = _args_diff(failed.args, candidate.args)
                    if diff:
                        between = stream[index : index + offset + 2]
                        error_text = _clip_edges(failed.error or failed.output_text)
                        episodes.append(
                            _episode(
                                "error_recovery",
                                traj,
                                [
                                    *_end_item(traj, failed, "error", snippet=error_text),
                                    *_end_item(traj, candidate, "result"),
                                    EvidenceItem(_start(traj, candidate), "fixed_call", True),
                                    EvidenceItem(_start(traj, failed), "failed_call", False),
                                    *task,
                                ],
                                [call.name for _, call in between],
                                {
                                    "tool": failed.name,
                                    "error_type": failed.error_type,
                                    "error": error_text,
                                    "detected_by": "status" if failed.status == "error" else "output_marker",
                                    "args_diff": diff,
                                    "calls_between": offset,
                                    "subagent": failed.subagent,
                                },
                                [
                                    _anchor(turn.run_id, failed.seq_start),
                                    _anchor(later.run_id, candidate.seq_start),
                                ],
                            ),
                        )
                    break
    return episodes


_CallPair = tuple[ToolCall, ToolCall]


def _action_changes(
    before: list[ToolCall],
    after: list[ToolCall],
) -> tuple[dict[str, Any] | None, dict[str, _CallPair]]:
    """How the agent's actions changed between two turn groups.

    Tools added or removed by name, and for a tool used in both groups the
    diff of its last call before against its first call after, compared in
    normalized form. Returns the change record (``None`` when nothing
    changed) and, per tool whose arguments changed, that pair of calls.
    """
    names_before = {call.name for call in before}
    names_after = {call.name for call in after}
    last_before = {call.name: call for call in before}
    first_after: dict[str, ToolCall] = {}
    for call in after:
        first_after.setdefault(call.name, call)
    args_changed: dict[str, Any] = {}
    pairs: dict[str, _CallPair] = {}
    for name in sorted(names_before & names_after):
        diff = _args_diff(
            _normalize_args(last_before[name].args),
            _normalize_args(first_after[name].args),
        )
        if diff:
            args_changed[name] = diff
            pairs[name] = (last_before[name], first_after[name])
    changes = {
        "tools_added": sorted(names_after - names_before),
        "tools_removed": sorted(names_before - names_after),
        "args_changed": args_changed,
    }
    return (changes if any(changes.values()) else None), pairs


def _detect_user_correction(traj: Trajectory) -> list[Episode]:
    """A user message correcting the previous turn group, with a changed action.

    The message opening a turn group must contain a correction marker, and
    the group's actions must differ from the previous group's: a tool added
    or removed, or a tool used in both groups called with different
    normalized arguments (the diff is recorded). A STRONG_CORRECTION_MARKERS
    phrase makes a strong correction; only WEAK_CORRECTION_MARKERS make a
    weak one with WEAK_CORRECTION_WEIGHT. A marker without an observed change
    is not an episode; no intent recognition is attempted, so the diff shows
    what changed, not that the message caused it. A group without a user
    message cannot correct anything, and the synthetic prelude turn is never
    the corrected one.

    Required evidence: the correcting message, shown as the window around
    its first marker (a long message keeps the phrase, not its head). The
    corrected turn and the before/after calls of each changed tool are
    context.
    """
    episodes: list[Episode] = []
    groups = _groups(traj)
    for prev, group in zip(groups, groups[1:]):
        head = group[0]
        message = head.user_message
        if prev[0].run_id == PRELUDE_RUN_ID or message is None:
            continue
        found = _markers(message)
        if not found:
            continue
        before = [call for turn in prev for call in turn.tool_calls]
        after = [call for turn in group for call in turn.tool_calls]
        changes, pairs = _action_changes(before, after)
        if changes is None:
            continue
        markers = [marker for marker, _ in found]
        strong = any(marker in STRONG_CORRECTION_MARKERS for marker in markers)
        marker, position = found[0]
        snippet = _window(_collapse(message), position, position + len(marker), MARKER_CONTEXT)
        episodes.append(
            _episode(
                "user_correction",
                traj,
                [
                    EvidenceItem(_turn_ref(traj, head), "correction", True, snippet),
                    EvidenceItem(_turn_ref(traj, prev[0]), "corrected_turn", False),
                    *(EvidenceItem(_start(traj, earlier), "before_call", False) for earlier, _ in pairs.values()),
                    *(EvidenceItem(_start(traj, later), "after_call", False) for _, later in pairs.values()),
                ],
                [call.name for call in after],
                {
                    "correction_text": snippet,
                    "strength": "strong" if strong else "weak",
                    "markers": markers,
                    "tools_before": [call.name for call in before],
                    "tools_after": [call.name for call in after],
                    "changes": changes,
                    "run_id_before": prev[0].run_id,
                    "run_id_after": head.run_id,
                },
                [_anchor(head.run_id, head.seq_start)],
                weight=None if strong else WEAK_CORRECTION_WEIGHT,
            ),
        )
    return episodes


def _search_signature(call: ToolCall) -> str:
    """The search keys of a call, canonicalized."""
    args = _normalize_args(call.args)
    return _canonical({key: value for key, value in args.items() if key in SEARCH_KEYS})


def _fruitless(call: ToolCall) -> bool:
    """Whether an attempt yielded nothing: it failed, its output is empty, or the output says so."""
    if _failed_call(call):
        return True
    text = (call.output_text or "").strip()
    return not text or any(pattern.search(text) for pattern in _NO_RESULT_MARKER_RES)


def _forced_search(calls: list[ToolCall]) -> bool:
    """Whether a search key changed right after a fruitless attempt.

    A pattern refined after a hit is a fan-out over results, not a retry;
    the change is forced only when the attempt before it found nothing
    (:func:`_fruitless`).
    """
    return any(
        _search_signature(previous) != _search_signature(current) and _fruitless(previous)
        for previous, current in zip(calls, calls[1:])
    )


def _attempt_items(traj: Trajectory, call: ToolCall, *, required: bool) -> list[EvidenceItem]:
    """One attempt of a retry loop: its start and, unless it is an orphan, its result or error."""
    if _failed_call(call):
        end = _end_item(traj, call, "error", snippet=_clip_edges(call.error or call.output_text), required=required)
    else:
        end = _end_item(traj, call, "result", required=required)
    return [EvidenceItem(_start(traj, call), "attempt", required), *end]


def _retry_episode(traj: Trajectory, chain: list[_Step], subject: str, head: Turn) -> list[Episode]:
    """The retry_loop episode of one (tool, work object) chain, if it qualifies."""
    if len(chain) < RETRY_MIN_ATTEMPTS:
        return []
    calls = [call for _, call in chain]
    variants: dict[str, dict[str, Any]] = {}
    for call in calls:
        normalized = _normalize_args(call.args)
        variants.setdefault(_canonical(normalized), _clip_args(normalized))
    if len(variants) < 2:
        return []
    if any(_failed_call(call) for call in calls):
        reason = "failed attempt"
    elif _forced_search(calls):
        reason = "search key varies"
    else:
        return []
    last = len(calls) - 1
    return [
        _episode(
            "retry_loop",
            traj,
            [
                *(
                    item
                    for index, call in enumerate(calls)
                    for item in _attempt_items(traj, call, required=index in (0, last))
                ),
                *_task_item(traj, head, required=False),
            ],
            [call.name for call in calls],
            {
                "tool_name": calls[0].name,
                "work_object": subject,
                "attempts": len(calls),
                "reason": reason,
                "args_variants": list(variants.values()),
                "statuses": [call.status for call in calls],
                "run_id": chain[0][0].run_id,
                "outcome": _clip_outcome(calls[-1].error or calls[-1].output_text),
            },
            [_anchor(turn.run_id, call.seq_start) for turn, call in chain],
        ),
    ]


def _detect_retry_loop(traj: Trajectory) -> list[Episode]:
    """Three or more attempts at one operation, forced by failure or search.

    Attempts are calls of one tool on one work object inside one stream;
    calls recorded without arguments have no object and never chain.
    A chain of RETRY_MIN_ATTEMPTS or more calls (orphans count as attempts)
    with at least two distinct normalized argument sets is an episode when
    an attempt failed (by status or by an ERROR_MARKERS output, see
    :func:`_failed_call`), or when a search key changed right after an
    attempt that found nothing (:func:`_forced_search`). Equal argument
    keys alone are not a retry: reading three files is a fan-out, a
    parameter sweep that never failed is not a loop, and a query refined
    after a result — sixty different SQL statements that each returned rows
    — is analysis, not a retry.

    Required evidence: the first and the last attempt with their results or
    errors; the attempts between them and the group's user message (``task``)
    are context.
    """
    episodes: list[Episode] = []
    for group in _groups(traj):
        for stream in _streams(group):
            chains: dict[tuple[str, str], list[_Step]] = {}
            for turn, call in stream:
                subject = _work_object(call)
                if subject is None:
                    continue
                chains.setdefault((call.name, subject), []).append((turn, call))
            for (_, subject), chain in chains.items():
                episodes.extend(_retry_episode(traj, chain, subject, group[0]))
    return episodes


def _detect_approval_denied(traj: Trajectory) -> list[Episode]:
    """A human rejected at least one requested action; the next calls show the reaction.

    The decision is read by :func:`classify_approval`; only ``denied`` makes
    an episode, naming the rejected actions only, and an ``unknown`` shape is
    logged at debug level and ignored. The context is the next
    DENIAL_CONTEXT_CALLS calls inside the approval's turn group — the
    ``resume`` turn that continues an interrupted turn, never a following
    ``dispatch`` turn. Approvals carry no subagent, so the context takes
    calls of any subagent.

    Required evidence: the decision; the following calls are context.
    """
    episodes: list[Episode] = []
    for group in _groups(traj):
        for position, turn in enumerate(group):
            for approval in turn.approvals:
                verdict = classify_approval(approval.request, approval.decision)
                if verdict.status == "unknown":
                    logger.debug(
                        "approval %s (seq %s) of thread %s ignored: %s",
                        approval.interrupt_id,
                        approval.seq,
                        traj.thread_id,
                        verdict.reason,
                    )
                if verdict.status != "denied":
                    continue
                following = [call for call in turn.tool_calls if call.seq_start > approval.seq]
                for later in group[position + 1 :]:
                    following.extend(later.tool_calls)
                context = following[:DENIAL_CONTEXT_CALLS]
                episodes.append(
                    _episode(
                        "approval_denied",
                        traj,
                        [
                            EvidenceItem(traj.event_ref(seq=approval.seq, line=approval.line), "approval", True),
                            *(EvidenceItem(_start(traj, call), "next_call", False) for call in context),
                        ],
                        [call.name for call in context],
                        {
                            "interrupt_id": approval.interrupt_id,
                            "run_id": approval.run_id or turn.run_id,
                            "tools": [action["name"] for action in verdict.denied if action["name"] is not None],
                            "denied_actions": verdict.denied,
                            "request": _clip(approval.request),
                            "decision": approval.decision,
                            "next_tools": [
                                {
                                    "name": call.name,
                                    "args": _clip_args(call.args),
                                    "status": call.status,
                                }
                                for call in context
                            ],
                        },
                        [_anchor(turn.run_id, approval.seq)],
                    ),
                )
    return episodes


def _detect_skill_gap(traj: Trajectory, index: BM25Index) -> list[Episode]:
    """Domain work done without consulting a skill the library describes.

    Fires when tools other than the catalog tools were used, no skill was
    consulted, and the best BM25 match of the user messages plus tool names
    against the library scores at least SKILL_GAP_MIN_SCORE. The candidate is
    a description-fix suggestion for that skill, not a new skill. It is a
    trajectory-level observation and has no anchors.

    Required evidence: the first domain call and the first user message;
    later user messages are context.
    """
    if traj.skills_consulted or len(index) == 0:
        return []
    domain_tools: list[str] = []
    first_call: ToolCall | None = None
    for call in _flat_calls(traj):
        if call.name in CATALOG_TOOLS:
            continue
        if first_call is None:
            first_call = call
        if call.name not in domain_tools:
            domain_tools.append(call.name)
    if first_call is None:
        return []
    turns = [turn for turn in traj.turns if turn.user_message]
    query = "\n".join([turn.user_message or "" for turn in turns] + domain_tools)
    hits = index.search(query, top_k=1)
    if not hits or hits[0].score < SKILL_GAP_MIN_SCORE:
        return []
    hit = hits[0]
    return [
        _episode(
            "skill_gap",
            traj,
            [
                EvidenceItem(_start(traj, first_call), "first_call", True),
                *(EvidenceItem(_turn_ref(traj, turn), "user_message", index == 0) for index, turn in enumerate(turns)),
            ],
            domain_tools,
            {
                "candidate_skill": hit.doc.name,
                "score": round(hit.score, 4),
                "matched_terms": list(hit.matched),
                "domain_tools": domain_tools,
            },
            [],
        ),
    ]


def _procedure_segments(
    traj: Trajectory,
    *,
    min_len: int = NGRAM_MIN,
    split_on_markers: bool = True,
) -> list[list[_Step]]:
    """Runs of ``ok`` domain calls inside one stream: the steps of a procedure.

    Catalog calls are not steps and are skipped; a call that failed or
    never finished ends the segment, so a repeated failure is never mined
    as a procedure. With ``split_on_markers`` an ``ok`` call whose output
    carries an ERROR_MARKERS text (:func:`_failed_call`) ends the segment
    too — the cross-session miner's rule; the demo detector keeps such a
    call in its chain and drops the chain with an ``error_in_output`` note
    instead. Segments shorter than ``min_len`` are dropped.
    """
    segments: list[list[_Step]] = []
    for group in _groups(traj):
        for stream in _streams(group):
            current: list[_Step] = []
            for turn, call in stream:
                if call.name in CATALOG_TOOLS:
                    continue
                if call.status != "ok" or (split_on_markers and _failed_call(call)):
                    if len(current) >= min_len:
                        segments.append(current)
                    current = []
                    continue
                current.append((turn, call))
            if len(current) >= min_len:
                segments.append(current)
    return segments


def _note(notes: list[DetectorNote] | None, reason: str, detail: str, seqs: list[int]) -> None:
    """Record why the observed_procedure detector dropped a chain, when the caller collects notes."""
    if notes is not None:
        notes.append(DetectorNote("observed_procedure", reason, detail, seqs))


def _chain_problem(head: Turn, chain: list[_Step]) -> tuple[str, str] | None:
    """Why ``chain`` is not a confirmed procedure — ``(DROP_* reason, detail)`` — or ``None``.

    Checked in order: the group head has no user message (an earlier group's
    message is never borrowed), a call was recorded without ``tool.start``
    (its arguments are unknown), an output carries an ERROR_MARKERS text
    (any step: an erroneous intermediate result is not a confirmed
    procedure), the last output is empty (no observable outcome).
    """
    if head.run_id == PRELUDE_RUN_ID or not (head.user_message or "").strip():
        return DROP_NO_TASK, f"turn {head.run_id} ({head.source}) has no user message"
    for _, call in chain:
        if call.line_end == call.line_start:
            detail = f"{call.name} at seq {call.seq_start} was recorded without tool.start; its arguments are unknown"
            return DROP_MISSING_START, detail
    for _, call in chain:
        marker = _error_marker(call.output_text)
        if marker is not None:
            return (
                DROP_ERROR_IN_OUTPUT,
                f"{call.name} at seq {call.seq_end} returned ok but its output contains {marker!r}",
            )
    last = chain[-1][1]
    if not last.output_text.strip():
        return DROP_EMPTY_RESULT, f"last call {last.name} at seq {last.seq_end} has no output; no observable outcome"
    return None


def _observed_episode(traj: Trajectory, head: Turn, chain: list[_Step], chain_index: int) -> Episode:
    """The observed_procedure episode of one confirmed chain."""
    last = chain[-1][1]
    return _episode(
        "observed_procedure",
        traj,
        [
            EvidenceItem(_turn_ref(traj, head), "task", True),
            *(
                item
                for _, call in chain
                for item in (
                    EvidenceItem(_start(traj, call), "step", True),
                    *_end_item(traj, call, "result", snippet=_clip_edges(call.output_text)),
                )
            ),
        ],
        [call.name for _, call in chain],
        {
            "task": _clip(head.user_message),
            "run_id": chain[0][0].run_id,
            "subagent": chain[0][1].subagent,
            "chain_index": chain_index,
            "calls": len(chain),
            "steps": [
                {
                    "tool": call.name,
                    "work_object": _work_object(call),
                    "args": _clip_args(call.args),
                    "result": _clip_outcome(call.output_text),
                }
                for _, call in chain
            ],
            "outcome": _clip_outcome(last.output_text),
            "verification": f"observed output of {last.name}: {_clip_outcome(last.output_text)}",
        },
        [_anchor(turn.run_id, call.seq_start) for turn, call in chain],
    )


def _detect_observed_procedure(traj: Trajectory, notes: list[DetectorNote] | None = None) -> list[Episode]:
    """Short chains of ``ok`` domain calls with their task and outputs (demo evidence).

    Segments of :func:`_procedure_segments` (any length) are cut into
    consecutive windows of at most OBSERVED_MAX_CALLS calls from the start —
    the head of a procedure carries its inputs — and at most
    OBSERVED_MAX_CHAINS windows become episodes, in model order. A window is
    dropped for the first problem :func:`_chain_problem` finds, then as a
    duplicate, and only an otherwise valid window for the chain limit; every
    drop is reported through ``notes``. AI messages are never read: an assistant
    saying "done" without a completed call yields no chain at all
    (DROP_NO_COMPLETED_CALLS). Windows are disjoint, so no two chains share
    a ref; the duplicate check is a defensive invariant.

    Required evidence: the task (the group's user message), every step and
    its result. Weight 0.0: a chain admits a thread only through the demo
    gate, never through the score.
    """
    heads = _heads(traj)
    segments = _procedure_segments(traj, min_len=1, split_on_markers=False)
    if not segments:
        calls = [call for call in _flat_calls(traj) if call.name not in CATALOG_TOOLS]
        counts = Counter(call.status for call in calls)
        detail = (
            f"{len(calls)} domain calls: {counts['ok']} ok, {counts['error']} error, "
            f"{counts['orphan']} orphan; no completed chain"
        )
        _note(notes, DROP_NO_COMPLETED_CALLS, detail, [])
        return []
    episodes: list[Episode] = []
    seen: set[tuple[EvidenceRef, ...]] = set()
    for segment in segments:
        for start in range(0, len(segment), OBSERVED_MAX_CALLS):
            window = segment[start : start + OBSERVED_MAX_CALLS]
            names = ", ".join(call.name for _, call in window)
            seqs = [call.seq_start for _, call in window]
            head = heads[id(window[0][0])]
            problem = _chain_problem(head, window)
            if problem is not None:
                _note(notes, problem[0], problem[1], seqs)
                continue
            key = tuple(ref for _, call in window for ref in (_start(traj, call), _end(traj, call)) if ref is not None)
            if key in seen:
                _note(notes, DROP_DUPLICATE, f"chain {names} cites the same events as an earlier chain", seqs)
                continue
            if len(episodes) == OBSERVED_MAX_CHAINS:
                _note(
                    notes,
                    DROP_CHAIN_LIMIT,
                    f"chain {names} skipped: {OBSERVED_MAX_CHAINS} chains already selected",
                    seqs,
                )
                continue
            seen.add(key)
            episodes.append(_observed_episode(traj, head, window, len(episodes) + 1))
    return episodes


# --------------------------------------------------------------- public API


def extract_episodes(
    traj: Trajectory,
    *,
    skill_index: BM25Index | None = None,
    demo: bool = False,
    notes: list[DetectorNote] | None = None,
) -> list[Episode]:
    """Run every per-trajectory detector; ``skill_gap`` needs a skill index.

    ``demo`` adds the ``observed_procedure`` detector, whose drop reasons are
    appended to ``notes`` when a list is given.
    """
    episodes = [
        *_detect_error_recovery(traj),
        *_detect_user_correction(traj),
        *_detect_retry_loop(traj),
        *_detect_approval_denied(traj),
    ]
    if skill_index is not None:
        episodes.extend(_detect_skill_gap(traj, skill_index))
    if demo:
        episodes.extend(_detect_observed_procedure(traj, notes))
    return episodes


def _context_items(traj: Trajectory, call: ToolCall) -> list[EvidenceItem]:
    """Non-required items for a neighbouring call: its start and its result or error."""
    start, end = _start(traj, call), _end(traj, call)
    items: list[EvidenceItem] = []
    if end != start:
        items.append(EvidenceItem(start, "context_call", False))
    if end is not None:
        if call.status == "error":
            items.append(EvidenceItem(end, "context_error", False, _clip_edges(call.error or call.output_text)))
        else:
            items.append(EvidenceItem(end, "context_result", False))
    return items


def expand_episode_context(
    episodes: list[Episode],
    traj: Trajectory,
    *,
    surrounding_events: int,
    only: Collection[int] | None = None,
) -> list[Episode]:
    """Episodes of ``traj`` with the neighbours of their cited calls added as context.

    For every cited call, the ``surrounding_events`` calls before and after
    it inside the same stream contribute non-required ``context_call`` and
    ``context_result`` / ``context_error`` items, the call's own missing
    result is added the same way, and the group's user message becomes a
    ``task`` item. With ``surrounding_events == 0`` only the missing results
    and the task are added. ``only`` restricts the expansion to the listed
    indices. An episode that gains nothing is returned as the same object;
    an episode of another trajectory (the second thread of a
    ``repeated_procedure``) is never touched. Anchors, facts and weights are
    unchanged: nothing is invented.
    """
    if surrounding_events < 0:
        raise ValueError(f"surrounding_events must be >= 0, got {surrounding_events}")
    positions: dict[EvidenceRef, tuple[Turn, list[_Step], int]] = {}
    for group in _groups(traj):
        for stream in _streams(group):
            for index, (_, call) in enumerate(stream):
                positions.setdefault(_start(traj, call), (group[0], stream, index))
                end = _end(traj, call)
                if end is not None:
                    positions.setdefault(end, (group[0], stream, index))
    expanded: list[Episode] = []
    for position, episode in enumerate(episodes):
        if (only is not None and position not in only) or episode.source != traj.source:
            expanded.append(episode)
            continue
        refs = {item.ref for item in episode.evidence}
        new: list[EvidenceItem] = []
        for item in list(episode.evidence):
            found = positions.get(item.ref)
            if found is None:
                continue
            head, stream, index = found
            neighbours = stream[max(0, index - surrounding_events) : index + surrounding_events + 1]
            candidates = [context for _, call in neighbours for context in _context_items(traj, call)]
            for candidate in [*candidates, *_task_item(traj, head, required=False)]:
                if candidate.ref not in refs:
                    refs.add(candidate.ref)
                    new.append(candidate)
        expanded.append(episode if not new else replace(episode, evidence=[*episode.evidence, *new]))
    return expanded


def _key_runs(segment: list[_Step]) -> list[tuple[str, int]]:
    """``(step key, index)`` of the first call of every run of equal keys in a segment.

    Three listings in a row are one step ("look around"), not three; the
    first call of the run is the one an episode cites.
    """
    runs: list[tuple[str, int]] = []
    for index, (_, call) in enumerate(segment):
        key = _step_key(call)
        if not runs or runs[-1][0] != key:
            runs.append((key, index))
    return runs


def _is_extended(
    gram: tuple[str, ...],
    frequent: dict[tuple[str, ...], set[str]],
) -> bool:
    """True when a longer frequent n-gram contains ``gram`` with equal support."""
    if len(gram) >= NGRAM_MAX:
        return False
    count = len(frequent[gram])
    return any(
        len(threads) == count and (other[1:] == gram or other[:-1] == gram)
        for other, threads in frequent.items()
        if len(other) == len(gram) + 1
    )


def mine_cross_session(
    trajs: list[Trajectory],
    *,
    min_support: int = 2,
) -> list[Episode]:
    """Step-key n-grams (NGRAM_MIN..NGRAM_MAX) shared by ``min_support`` threads.

    Steps come from :func:`_procedure_segments`: calls that returned ``ok``,
    in one execution context, catalog calls removed. A step is compared by
    :func:`_step_key` — the tool name plus the program of a command or the
    verb and table of an SQL statement — and a run of equal keys (``ls, ls,
    ls``: one tool called repeatedly) is one step, cited by its first call
    (:func:`_key_runs`); an n-gram of one key is not a procedure. Support
    counts distinct ``thread_id`` values, so repetition
    inside one trajectory is not evidence. Only closed patterns are
    reported: an n-gram is dropped when a longer frequent n-gram containing
    it has the same support, so a shared five-step procedure yields one
    episode, not ten. ``facts["ngram"]`` and ``tool_sequence`` list the
    owner's tool names, ``facts["step_keys"]`` the keys. The episode belongs to
    the first supporting trajectory in input order and cites, as required
    evidence, the steps of that trajectory and of the lexicographically
    first other supporting thread — proof from two sessions, while
    ``facts["support"]`` counts every supporting thread (all listed in
    ``facts["thread_ids"]``). The owner's results and the user message of
    its turn group (``task``) are context. A shared n-gram says that several
    sessions issued these calls in this order and each returned ``ok`` — not
    that the task succeeded.
    """
    if min_support < 2:
        raise ValueError(f"min_support must be >= 2, got {min_support}")
    by_thread: dict[str, Trajectory] = {}
    for traj in trajs:
        by_thread.setdefault(traj.thread_id, traj)
    segments_by_thread = {thread_id: _procedure_segments(traj) for thread_id, traj in by_thread.items()}
    support: dict[tuple[str, ...], set[str]] = {}
    # (gram, thread) -> (segment index, index of the first call of each of its runs).
    first_seen: dict[tuple[tuple[str, ...], str], tuple[int, list[int]]] = {}
    for thread_id, segments in segments_by_thread.items():
        for segment_index, segment in enumerate(segments):
            runs = _key_runs(segment)
            keys = [key for key, _ in runs]
            for size in range(NGRAM_MIN, NGRAM_MAX + 1):
                for start in range(len(keys) - size + 1):
                    gram = tuple(keys[start : start + size])
                    if len(set(gram)) < 2:
                        continue
                    support.setdefault(gram, set()).add(thread_id)
                    indices = [index for _, index in runs[start : start + size]]
                    first_seen.setdefault((gram, thread_id), (segment_index, indices))
    frequent = {g: t for g, t in support.items() if len(t) >= min_support}
    order = {thread_id: index for index, thread_id in enumerate(by_thread)}

    def steps_of(gram: tuple[str, ...], thread_id: str) -> list[_Step]:
        segment_index, indices = first_seen[gram, thread_id]
        segment = segments_by_thread[thread_id][segment_index]
        return [segment[index] for index in indices]

    heads_by_thread: dict[str, dict[int, Turn]] = {}
    episodes: list[Episode] = []
    for gram in sorted(frequent, key=lambda g: (-len(frequent[g]), -len(g), g)):
        if _is_extended(gram, frequent):
            continue
        threads = frequent[gram]
        owner = min(threads, key=order.__getitem__)
        second = min(threads - {owner})
        own, other = steps_of(gram, owner), steps_of(gram, second)
        owner_traj = by_thread[owner]
        if owner not in heads_by_thread:
            heads_by_thread[owner] = _heads(owner_traj)
        owner_head = heads_by_thread[owner][id(own[0][0])]
        episodes.append(
            _episode(
                "repeated_procedure",
                owner_traj,
                [
                    *(
                        item
                        for _, call in own
                        for item in (
                            EvidenceItem(_start(owner_traj, call), "step", True),
                            *_end_item(
                                owner_traj, call, "result", snippet=_clip_edges(call.output_text), required=False
                            ),
                        )
                    ),
                    *(EvidenceItem(_start(by_thread[second], call), "step", True) for _, call in other),
                    *_task_item(owner_traj, owner_head, required=False),
                ],
                [call.name for _, call in own],
                {
                    "ngram": [call.name for _, call in own],
                    "step_keys": list(gram),
                    "support": len(threads),
                    "thread_ids": sorted(threads),
                },
                [_anchor(turn.run_id, call.seq_start) for turn, call in own],
            ),
        )
    return episodes


# ------------------------------------------------------------- incidents


def group_incidents(episodes: list[Episode]) -> list[list[Episode]]:
    """Episodes about the same events, grouped: connected components over anchors.

    An episode without anchors is keyed on its own identity, so an exact
    duplicate joins its twin and nothing else. A zero-weight episode (an
    observed_procedure chain) joins the incident of the first anchor it
    shares but never bridges two incidents, so demo changes neither the
    incident count nor the score. Incidents are ordered by their first
    episode; episodes keep list order inside an incident.
    """
    parent = list(range(len(episodes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owner: dict[str, int] = {}
    for index, episode in enumerate(episodes):
        if episode.weight <= 0.0:
            continue
        for key in _incident_keys(episode):
            root = find(owner.setdefault(key, index))
            if root != find(index):
                parent[root] = find(index)
    for index, episode in enumerate(episodes):
        if episode.weight > 0.0:
            continue
        shared = next((find(owner[key]) for key in _incident_keys(episode) if key in owner), None)
        if shared is not None:
            parent[index] = shared
        for key in _incident_keys(episode):
            owner.setdefault(key, index)
    incidents: dict[int, list[Episode]] = {}
    for index, episode in enumerate(episodes):
        incidents.setdefault(find(index), []).append(episode)
    return list(incidents.values())


def _incident_keys(episode: Episode) -> list[str]:
    """The anchors an episode is grouped by, or its own identity when it has none."""
    return episode.anchors or [f"{episode.kind}@{episode.thread_id}:{episode.evidence_seq}"]


def _incident_score(incidents: list[list[Episode]]) -> float:
    """Sum of the heaviest weight per incident; rounded so 0.6 + 0.7 + 0.7 reaches 2.0."""
    return round(float(sum(max(episode.weight for episode in incident) for incident in incidents)), 6)


def evidence_score(episodes: list[Episode]) -> float:
    """Sum over incidents of the heaviest episode in each.

    Several detectors describing one chain count once; independent incidents
    add up. Compared with the ``min_evidence_score`` config by
    :func:`gate_decision`.
    """
    return _incident_score(group_incidents(episodes))


def gate_decision(episodes: list[Episode], *, min_score: float, demo: bool = False) -> GateDecision:
    """Whether the episodes admit their thread to the LLM stage, and why.

    A thread with no episodes never passes. It passes when the incident
    score reaches ``min_score``, or — the strong correction rule — when it
    holds a strong ``user_correction`` and ``min_score`` is at most
    DEFAULT_MIN_EVIDENCE_SCORE: one explicit correction with an observed
    change of action is worth analysing at the default settings, while a
    stricter threshold opts out of the rule. In demo mode a thread the
    ordinary rules refuse still passes (GATE_DEMO_OVERRIDE) when at least
    one episode :func:`has_required_evidence`; otherwise the refusal
    carries GATE_DEMO_NOT_APPLICABLE as its detail. Score, weights and
    incidents are never altered.
    """
    incidents = group_incidents(episodes)
    score = _incident_score(incidents)
    if not episodes:
        return GateDecision(incidents, score, False, GATE_NO_EPISODES)
    if score >= min_score:
        return GateDecision(incidents, score, True, GATE_SCORE_REACHED)
    strong = any(
        episode.kind == "user_correction" and episode.facts.get("strength") == "strong" for episode in episodes
    )
    if strong and min_score <= DEFAULT_MIN_EVIDENCE_SCORE:
        return GateDecision(incidents, score, True, GATE_STRONG_CORRECTION)
    if demo:
        complete = sorted({episode.kind for episode in episodes if has_required_evidence(episode)})
        if complete:
            verb = "has" if len(complete) == 1 else "have"
            detail = f"{', '.join(complete)} {verb} required evidence"
            return GateDecision(incidents, score, True, GATE_DEMO_OVERRIDE, detail)
        return GateDecision(incidents, score, False, GATE_SCORE_BELOW, GATE_DEMO_NOT_APPLICABLE)
    return GateDecision(incidents, score, False, GATE_SCORE_BELOW)
