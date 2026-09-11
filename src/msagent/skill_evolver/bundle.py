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

"""Evidence bundle: the only view of a session the classify stage's LLM gets.

:func:`build_evidence_bundle` renders :class:`Episode` records (the output
of the code-only detectors in :mod:`msagent.skill_evolver.features`) into
one markdown block per episode, heaviest first, with the structured facts and
short excerpts of the events the episode cites. There is no transcript and
no chronological narrative: the model sees evidence, not the session.

Every excerpt line carries a bundle-local id (``[ev3]``) and the returned
:class:`EvidenceBundle` keeps a registry of exactly those fragments — text
and :class:`EvidenceRef` — so the classify stage can reject a candidate
citing anything the model was not shown. A bare event number is never
citable: an episode's optional excerpts may be trimmed under budget, and an
episode whose *required* excerpts do not fit is excluded altogether rather
than shown as a stump.

Stdlib only: importing this module must not load langchain.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from msagent.skill_evolver.features import Episode
from msagent.trajectory_recorder.model import (
    AiMessage,
    Approval,
    EvidenceRef,
    ToolCall,
    Trajectory,
    Turn,
)

logger = logging.getLogger(__name__)

# Header of one episode block; ``idx`` is the 1-based rank among the rendered blocks.
EPISODE_HEADER = "### Episode E{idx} — {kind} (weight {weight:.2f}, thread {thread})"
# Second line of an observed_procedure block (demo evidence selected by code).
OBSERVED_LINE = (
    "Observed procedure: {calls} call(s) in one execution context (selected by code); "
    "ok status is not proof of task success"
)
# Code-rejection name of an episode excluded because its required excerpts do not fit.
EXCLUDED_CODE = "insufficient_context_budget"
# Characters of the thread id shown in the header and on cross-thread excerpts.
THREAD_ID_CHARS = 8
# Length limits of one rendered fact line and of one excerpt line.
FACT_LIMIT = 800
EXCERPT_LIMIT = 300
# Marks text that was cut.
ELLIPSIS = "…"
# Prefix of the bundle-local fragment ids the model cites.
FRAGMENT_ID_PREFIX = "ev"

_SEPARATOR = "\n\n"
# ``Turn.source`` of the turns the reader opens itself (``reader._route``) at
# the seq of a real event that had no turn to belong to.
_SYNTHETIC_SOURCES = frozenset({"unknown", "prelude"})

# Outcome of one episode in the bundle: shown in full, shown with only its
# required excerpts (``trimmed``), or left out because even those did not
# fit (``excluded`` — insufficient context).
BundleStatus = Literal["shown", "trimmed", "excluded"]


@dataclass(frozen=True, slots=True)
class ShownFragment:
    """One excerpt the model saw: its id, the event it came from and the exact text."""

    id: str
    ref: EvidenceRef
    role: str
    required: bool
    # The excerpt line after "- [evN] ", i.e. "<label>: <text>".
    text: str


@dataclass(frozen=True, slots=True)
class BundleEpisode:
    """One input episode with what became of it."""

    episode: Episode
    status: BundleStatus
    # "E<rank>" of a shown or trimmed block; ``None`` for an excluded episode.
    episode_id: str | None = None
    # For an excluded episode: the characters its smallest rendering needed
    # and how many the budget still had when it was tried; ``None`` otherwise.
    needed: int | None = None
    room: int | None = None


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """The classify input: its text, the registry of shown fragments, the episode outcomes."""

    text: str
    # Fragment id -> fragment; exactly the "[evN]" lines of ``text``.
    shown: dict[str, ShownFragment]
    # Every input episode, in rendering order.
    episodes: list[BundleEpisode]
    # Episode id ("E<rank>") -> episode, for every rendered block.
    episode_ids: dict[str, Episode] = field(default_factory=dict)

    @property
    def kept(self) -> list[Episode]:
        """The episodes the model sees (shown or trimmed), in rendering order."""
        return [item.episode for item in self.episodes if item.status != "excluded"]


@dataclass(slots=True)
class _Record:
    """What the model sees for one event: a label, its text and its thread."""

    label: str
    text: str
    thread_id: str


_Records = dict[EvidenceRef, _Record]


# ------------------------------------------------------------------ helpers


def _json(value: Any) -> str:
    """Stable JSON text of a fact, argument or message value."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _clip(text: str, limit: int) -> str:
    """Whitespace-collapsed copy of ``text``, cut to ``limit`` characters."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - len(ELLIPSIS)] + ELLIPSIS


# ------------------------------------------------------------------ records


def _turn_record(turn: Turn, thread_id: str) -> _Record:
    if turn.user_message is None:
        return _Record("turn.start", f"(source={turn.source})", thread_id)
    return _Record("user", _json(turn.user_message), thread_id)


def _start_record(call: ToolCall, thread_id: str) -> _Record:
    return _Record(f"tool.start {call.name}", _json(call.args), thread_id)


def _end_record(call: ToolCall, thread_id: str) -> _Record:
    if call.status == "error":
        # A tool.result with status=error carries its text in output_text.
        detail = f"{call.error_type or 'error'}: {call.error or call.output_text or ''}"
        return _Record(f"tool.error {call.name} (error)", detail.rstrip(": "), thread_id)
    label = f"tool.result {call.name} ({call.status})"
    return _Record(label, call.output_text or "(no output)", thread_id)


def _ai_record(message: AiMessage, thread_id: str) -> _Record:
    text = _json(message.text)
    if message.tool_call_names:
        text += f" [tool calls: {', '.join(message.tool_call_names)}]"
    return _Record("ai", text, thread_id)


def _approval_record(approval: Approval, thread_id: str) -> _Record:
    detail = f"request={_json(approval.request)} decision={_json(approval.decision)}"
    return _Record("approval.decision", detail, thread_id)


def _index_thread(traj: Trajectory) -> _Records:
    """Records of one trajectory keyed by :class:`EvidenceRef`.

    Tool, AI and approval records go first; a synthetic turn (opened by the
    reader at the line of a real event) is added only when its ref is still
    free, so it never shadows that event. Refs are unique per physical
    line, so a recorder restart that repeats a ``seq`` yields two records.
    """
    by_ref: _Records = {}
    thread = traj.thread_id
    for turn in traj.turns:
        for call in turn.tool_calls:
            start = traj.event_ref(seq=call.seq_start, line=call.line_start)
            end = None
            if call.seq_end is not None and call.line_end is not None:
                end = traj.event_ref(seq=call.seq_end, line=call.line_end)
            if end != start:
                by_ref.setdefault(start, _start_record(call, thread))
            if end is not None:
                by_ref.setdefault(end, _end_record(call, thread))
        for message in turn.ai_messages:
            by_ref.setdefault(traj.event_ref(seq=message.seq, line=message.line), _ai_record(message, thread))
        for approval in turn.approvals:
            ref = traj.event_ref(seq=approval.seq, line=approval.line)
            by_ref.setdefault(ref, _approval_record(approval, thread))
    for turn in traj.turns:
        ref = traj.event_ref(seq=turn.seq_start, line=turn.line_start)
        by_ref.setdefault(ref, _turn_record(turn, thread))
    return by_ref


def _index_records(trajectories: list[Trajectory]) -> _Records:
    """Records of every trajectory; a source seen twice keeps its first copy."""
    records: _Records = {}
    seen: set[str] = set()
    for traj in trajectories:
        if traj.source in seen:
            continue
        seen.add(traj.source)
        records.update(_index_thread(traj))
    return records


# ---------------------------------------------------------------- rendering


def _check_refs(episode: Episode, records: _Records, sources: set[str]) -> None:
    """Every cited ref must resolve: episodes and trajectories must be the same data."""
    if episode.source not in sources:
        where = f"{episode.kind} episode cites source {episode.source!r}"
        raise ValueError(f"{where}, which is not among the trajectories")
    missing = [f"{item.ref.source}:{item.ref.line}" for item in episode.evidence if item.ref not in records]
    if missing:
        where = f"{episode.kind} episode of {episode.source!r}"
        raise ValueError(f"{where} cites {missing}, which is not in its trajectory")


def _render_block(
    rank: int,
    episode: Episode,
    records: _Records,
    by_ref: dict[EvidenceRef, ShownFragment],
    *,
    required_only: bool,
    excerpt_chars: int,
) -> tuple[str, list[ShownFragment]]:
    """One markdown block and the fragments it introduces (not yet in ``by_ref``).

    A ref already shown by an earlier block keeps its id and text: one event
    is one fragment, even when a lighter episode would have cut it
    differently. With ``required_only`` the optional excerpts are replaced by
    a count, which is not citable. Excerpt text is cut to ``excerpt_chars``.
    """
    items = [item for item in episode.evidence if item.required] if required_only else list(episode.evidence)
    omitted = len(episode.evidence) - len(items)
    lines = [
        EPISODE_HEADER.format(
            idx=rank,
            kind=episode.kind,
            weight=episode.weight,
            thread=episode.thread_id[:THREAD_ID_CHARS],
        ),
    ]
    if episode.kind == "repeated_procedure":
        support = episode.facts.get("support")
        lines.append(f"Support: {support} threads (counted by code); excerpts from 2 of them")
    if episode.kind == "observed_procedure":
        lines.append(OBSERVED_LINE.format(calls=episode.facts.get("calls")))
    if episode.tool_sequence:
        lines.append("Tools: " + _clip(", ".join(episode.tool_sequence), FACT_LIMIT))
    if episode.facts:
        lines.append("Facts:")
        for key, value in episode.facts.items():
            lines.append(f"- {key}: {_clip(_json(value), FACT_LIMIT)}")
    lines.append("Excerpts:")
    new: dict[EvidenceRef, ShownFragment] = {}
    for item in items:
        fragment = by_ref.get(item.ref) or new.get(item.ref)
        record = records[item.ref]
        if fragment is None:
            text = f"{record.label}: {_clip(item.snippet or record.text, excerpt_chars)}"
            fragment = ShownFragment(
                id=f"{FRAGMENT_ID_PREFIX}{len(by_ref) + len(new) + 1}",
                ref=item.ref,
                role=item.role,
                required=item.required,
                text=text,
            )
            new[item.ref] = fragment
        where = "" if item.ref.source == episode.source else f"(thread {record.thread_id[:THREAD_ID_CHARS]}) "
        lines.append(f"- [{fragment.id}] {where}{fragment.text}")
    if omitted:
        lines.append(f"- {ELLIPSIS} {omitted} more events not shown")
    return "\n".join(lines), list(new.values())


# --------------------------------------------------------------- public API


def build_evidence_bundle(
    episodes: list[Episode],
    trajectories: list[Trajectory],
    *,
    max_chars: int = 30000,
    excerpt_chars: int = EXCERPT_LIMIT,
    demo: bool = False,
) -> EvidenceBundle:
    """Render episodes for the classify stage; return the text and its registry.

    One block per episode, heaviest first (stable for equal weights); with
    ``demo`` the ``observed_procedure`` blocks come first — they carry the
    admission and are short, so heavier episodes' optional excerpts can never
    starve their required ones — and the rest keeps the weight order. Each
    episode is tried in full, then with its required excerpts only
    (``trimmed``); when even those do not fit ``max_chars`` the episode is
    ``excluded`` — insufficient context (EXCLUDED_CODE) — and later episodes
    are still tried. ``shown`` holds exactly the fragments printed with an
    ``[evN]`` id: a classification citing any other id is fabricated.
    Fragment ids are assigned in order of first appearance; one event has one
    id across blocks. ``episode_ids`` maps the ``E<rank>`` ids of the
    rendered blocks to their episodes. Excerpts are cut to ``excerpt_chars``.

    Raises ``ValueError`` on a non-positive budget, an ``excerpt_chars`` that
    leaves no room beside ELLIPSIS, and on an episode whose source is not
    among ``trajectories`` or whose evidence ref is not in it (episodes and
    trajectories must come from the same data). Nothing raises for size: an
    empty bundle is a legitimate outcome.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if excerpt_chars <= len(ELLIPSIS):
        raise ValueError(f"excerpt_chars must be greater than {len(ELLIPSIS)}, got {excerpt_chars}")
    records = _index_records(trajectories)
    sources = {traj.source for traj in trajectories}
    if demo:
        ranked = sorted(episodes, key=lambda episode: (episode.kind != "observed_procedure", -episode.weight))
    else:
        ranked = sorted(episodes, key=lambda episode: -episode.weight)
    for episode in ranked:
        _check_refs(episode, records, sources)

    by_ref: dict[EvidenceRef, ShownFragment] = {}
    blocks: list[str] = []
    outcomes: list[BundleEpisode] = []
    total = 0
    for episode in ranked:
        status: BundleStatus = "excluded"
        attempts = [False]
        if any(not item.required for item in episode.evidence):
            attempts.append(True)
        needed = 0
        room = max_chars - total
        for required_only in attempts:
            block, new = _render_block(
                len(blocks) + 1,
                episode,
                records,
                by_ref,
                required_only=required_only,
                excerpt_chars=excerpt_chars,
            )
            cost = len(block) + (len(_SEPARATOR) if blocks else 0)
            needed = cost
            if total + cost <= max_chars:
                blocks.append(block)
                total += cost
                by_ref.update((fragment.ref, fragment) for fragment in new)
                status = "trimmed" if required_only else "shown"
                break
        if status == "excluded":
            logger.warning(
                "bundle: excluded %s episode of thread %r: its required evidence does not fit max_chars=%d "
                "(needs %d chars, %d left)",
                episode.kind,
                episode.thread_id,
                max_chars,
                needed,
                room,
            )
            outcomes.append(BundleEpisode(episode, status, None, needed=needed, room=room))
        else:
            outcomes.append(BundleEpisode(episode, status, f"E{len(blocks)}"))
    shown = {fragment.id: fragment for fragment in by_ref.values()}
    episode_ids = {item.episode_id: item.episode for item in outcomes if item.episode_id}
    return EvidenceBundle(_SEPARATOR.join(blocks), shown, outcomes, episode_ids)
