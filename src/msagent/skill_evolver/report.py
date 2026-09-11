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

"""Decision report: why a thread did or did not produce a proposal.

Rules of REPORT_VERSION 1:

* One JSON file per thread and non-dry run at
  ``<ProjectPaths.root>/skill-evolver/decisions/<batch_dir_name(thread)>-<UTC>.json``
  (private state, 0700 directories, 0600 files); written also when the gate
  refused the thread, when nothing was saved and when the thread failed.
  Dry runs write nothing. ``/skill-review accept`` never touches it.
* The pipeline assembles the payload; this module stamps ``report_version``
  and ``evidence_text_file`` and writes atomically (temp + replace; a name
  collision gets a ``-2``, ``-3`` suffix). Expected payload keys:
  ``command, generated_at, thread_id, source, synthetic,
  policy {requested, selection, demo_mode, overrides},
  config {requested, effective}, prompts {contract_version, variants, hashes},
  episodes {total, counts, detector_notes}, gate, second_gate, bundles[],
  stages[], classifier {verdict, candidates, decisions, rejected},
  code_rejections[], plans {rendered, proposals, render_errors,
  rejected_targets, deferred}, quality_review[], coverage[], proposals[],
  llm {model, context_window, calls_used, limit, bound, budget_exhausted,
  note}, stop_message, failed``.
* ``<stem>.evidence.md`` (the bundle text per round, no model replies, no
  think blocks) is written next to the report only when the pipeline passes
  ``evidence_text`` (``diagnostics.save_evidence_text``).
* ``<stem>.draft-<n>-<plan slug>.rejected.md`` — the last SKILL.md draft of
  every refused plan (``diagnostics.save_rejected_drafts``), one file per
  entry of ``rejected_drafts``, listed in the document's
  ``rejected_draft_files``; never a proposal, never under the skills root.
* A thread is ``synthetic`` when its agent name or a component of its
  working directory contains ``synthetic``.

Stdlib only (plus the thread-folder name rule of :mod:`msagent.skill_evolver.writer`).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

from msagent.skill_evolver.writer import MAX_COLLISIONS, batch_dir_name

REPORT_VERSION = 1
DECISIONS_DIR = ("skill-evolver", "decisions")
SYNTHETIC_MARKER = "synthetic"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"
EVIDENCE_SUFFIX = ".evidence.md"
DRAFT_SUFFIX = ".rejected.md"
# Longest plan slug inside a rejected-draft file name.
DRAFT_SLUG_CHARS = 60


def draft_file_names(stem: str, drafts: Sequence[Mapping[str, str]]) -> list[str]:
    """``<stem>.draft-<n>-<plan slug>.rejected.md`` for every refused plan's draft, in order."""
    return [
        f"{stem}.draft-{number}-{_slug(str(draft.get('plan', '')))}{DRAFT_SUFFIX}"
        for number, draft in enumerate(drafts, 1)
    ]


def _slug(text: str) -> str:
    """A file-name-safe cut of a plan label (``create: Ascend profiler …`` → ``create-ascend-profiler-…``)."""
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")[:DRAFT_SLUG_CHARS].strip("-") or "plan"


def decisions_dir(state_dir: Path) -> Path:
    """``<state_dir>/skill-evolver/decisions``, created private (0700) best effort.

    When the directory cannot be created, every ``write_report`` into it
    fails and the caller logs that per thread; the run itself goes on.
    """
    path = state_dir.joinpath(*DECISIONS_DIR)
    for level in (path.parent, path):
        try:
            level.mkdir(parents=True, exist_ok=True)
            level.chmod(0o700)
        except OSError:
            pass
    return path


def is_synthetic(agent: str, working_dir: str) -> bool:
    """Synthetic fixtures announce themselves by agent name or working directory."""
    parts = (agent, *PurePath(working_dir).parts)
    return any(SYNTHETIC_MARKER in part.casefold() for part in parts)


def write_report(
    directory: Path,
    *,
    thread_id: str,
    payload: Mapping[str, Any],
    evidence_text: str | None = None,
    rejected_drafts: Sequence[Mapping[str, str]] = (),
) -> Path:
    """Write the report of one thread under ``directory``; returns the JSON path.

    Every entry of ``rejected_drafts`` (``{plan, code, content}``) becomes a
    ``<stem>.draft-<n>-<plan slug>.rejected.md`` sibling (see
    :func:`draft_file_names`) with a trailing note naming the plan and the
    rejection code; the document lists them in ``rejected_draft_files``.
    """
    stamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
    path = _reserve(directory, f"{batch_dir_name(thread_id)}-{stamp}")
    document = dict(payload)
    document["report_version"] = REPORT_VERSION
    document["evidence_text_file"] = None
    if evidence_text is not None:
        evidence_path = path.with_name(path.stem + EVIDENCE_SUFFIX)
        _write_atomically(evidence_path, evidence_text)
        document["evidence_text_file"] = evidence_path.name
    names = draft_file_names(path.stem, rejected_drafts)
    for name, draft in zip(names, rejected_drafts):
        note = (
            f"<!-- rejected draft of plan {draft.get('plan', '')} ({draft.get('code', '')}); "
            "kept by diagnostics.save_rejected_drafts, never a proposal -->\n"
        )
        _write_atomically(path.with_name(name), str(draft.get("content", "")).rstrip("\n") + "\n\n" + note)
    document["rejected_draft_files"] = names
    _write_atomically(path, json.dumps(document, sort_keys=True, ensure_ascii=False, indent=2, default=str) + "\n")
    return path


def _reserve(directory: Path, stem: str) -> Path:
    """Create ``<stem>.json`` or the first free ``<stem>-N.json`` (O_EXCL, 0600)."""
    for suffix in range(1, MAX_COLLISIONS + 1):
        candidate = directory / (f"{stem}.json" if suffix == 1 else f"{stem}-{suffix}.json")
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate
    raise RuntimeError(f"too many decision reports named {stem!r} under {directory}")


def _write_atomically(path: Path, text: str) -> None:
    """Write ``text`` to a private temp file in the same directory, fsync, then replace ``path``."""
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
