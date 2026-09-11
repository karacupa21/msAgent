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

"""Proposal writer: SKILL.md drafts that no skill scanner can see.

A generated and validated skill is written to
``<root>/.proposals/<thread>/<name>/SKILL.md`` next to a mandatory
``provenance.json``. :meth:`SkillFactory.load_skills` skips
dot-directories, and the extra ``<thread>`` level keeps the files below the
depth at which the agent's skill sources are enumerated, so a proposal
reaches the library only when a human moves it. Before anything is written
the whole package (SKILL.md and provenance.json) is scanned with
:func:`~msagent.skill_evolver.validator.scan_secrets`; a hit is
:class:`SecretsDetected` and nothing is created. Stdlib only.

Provenance contract (``provenance_version`` PROVENANCE_VERSION) is scoped to
the generation plan a proposal came from and tells three things apart: the
episodes the detectors extracted for the thread (with their bundle outcome:
shown, trimmed, excluded; an evidence item's ``id`` is null when this
proposal's candidates do not cite it), the fragments those candidates cite
(``evidence_shown``, id -> file, line, seq, redacted text; a subset of what
the classify model saw), and what reached this generation call
(``candidates`` are exactly its plan, ``generation_evidence`` maps a
candidate id to the fragment ids actually quoted to it and
``generation_evidence_omitted`` to the ids dropped for the budget, so an
incomplete set is never marked complete). Every candidate joins its events
through ``evidence_refs`` -> ``evidence_shown``; ``candidates_rejected``
lists the thread's classify rejections with their reason.

Version history: v5 renames the SKILL.md stage from render to generate:
``render_evidence`` / ``render_evidence_omitted`` became
``generation_evidence`` / ``generation_evidence_omitted``, and the stage key
``render`` of ``prompt_variants``, ``prompt_hashes`` and
``config.requested.prompts`` became ``generate``.
v4 adds ``policy`` (requested, selection, source), ``demo``,
``config`` (requested, effective), ``versions``, ``prompt_hashes``
(``sha256:<hex>`` per stage), ``quality_review``, ``verification``
(level, note), ``observed_procedure_source`` (the physical lines of the
observed_procedure episodes this plan cites), ``covering_skill``,
``target.base_sha256`` for updates, redacted evidence text and the package
secrets guard; ``/skill-review`` additionally reads ``demo`` and
``policy.selection`` (absent in older proposals: not a demo). v3 scoped the
record to its plan; v2 held every kept candidate of the thread and
the whole registry in each proposal; v1 (no ``provenance_version``) has seqs
in ``candidates[].evidence_refs`` and no registry. Every version shares
``category``, ``thread_ids``, ``generated_at`` and ``target``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from msagent.skill_evolver.bundle import EvidenceBundle
from msagent.skill_evolver.classify import Candidate
from msagent.skill_evolver.features import FEATURES_VERSION
from msagent.skill_evolver.generate import select_generation_evidence
from msagent.skill_evolver.validator import NAME_RE, redact_secrets, scan_secrets

PROPOSALS_DIR = ".proposals"
SKILL_FILE = "SKILL.md"
PROVENANCE_FILE = "provenance.json"
# Version of the provenance.json contract: 1 (unversioned) predates the
# evidence registry, 2 recorded the whole thread in every proposal, 3 is
# scoped to its plan, 4 adds policy/demo/review/verification, 5 renames the
# render stage keys to generation/generate; see the module docstring.
PROVENANCE_VERSION = 5
REQUIRED_PROVENANCE_KEYS: frozenset[str] = frozenset(
    {
        "provenance_version",
        "thread_ids",
        "episodes",
        "evidence_shown",
        "candidates",
        "model",
        "prompt_variants",
        "features_version",
        "generated_at",
        "policy",
        "demo",
        "config",
        "versions",
        "prompt_hashes",
        "quality_review",
        "verification",
    },
)
# ``verification.level`` values: the generator only ever records the first.
VERIFICATION_LEVELS = ("evidence_supported", "execution_verified")
# Stages whose template hash every proposal records.
PROMPT_HASH_STAGES = ("classify", "generate", "review")
# Keys of the ``policy`` record.
POLICY_KEYS = ("requested", "selection", "source")
# Upper bound of the -2, -3, ... suffix search for one proposal name.
MAX_COLLISIONS = 1000
# Folder name of a thread's proposals, after unsafe characters are replaced.
_BATCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class SecretsDetected(ValueError):
    """The proposal package carries a secret-looking value; nothing was written."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _string_leaves(value: Any) -> Iterator[str]:
    """Every str leaf of a JSON-like value, before json.dumps escapes it."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _string_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_leaves(item)


def scan_package(files: Mapping[str, str], *, raw: Mapping[str, Any] | None = None) -> list[str]:
    """``<file>:<line>: possible secret (<label>)`` for every hit in the package texts; values never echoed.

    ``raw`` is the provenance mapping before encoding: json.dumps escapes ``"``
    as ``\\"``, which hides a quoted credential assignment from the encoded
    scan, so its string fields are scanned unescaped as well.
    """
    findings = [
        f"{name}:{hit.line}: possible secret ({hit.label})"
        for name, text in files.items()
        for hit in scan_secrets(text)
    ]
    if raw is not None:
        findings.extend(
            f"{PROVENANCE_FILE}: possible secret ({hit.label}) in a text field"
            for leaf in _string_leaves(raw)
            for hit in scan_secrets(leaf)
        )
    return findings


def build_provenance(
    *,
    thread_ids: Sequence[str],
    bundle: EvidenceBundle,
    candidates: Sequence[Candidate],
    rejected: Sequence[tuple[Candidate, str]],
    sources: Mapping[str, str],
    model: str,
    prompt_variants: Mapping[str, str],
    category: str,
    target: Mapping[str, Any],
    policy: Mapping[str, Any],
    demo: bool,
    config: Mapping[str, Any],
    versions: Mapping[str, int],
    prompt_hashes: Mapping[str, str],
    quality_review: Mapping[str, Any],
    verification: Mapping[str, str],
    covering_skill: Mapping[str, Any] | None = None,
    generation_evidence: Mapping[str, Sequence[str]] | None = None,
    generation_evidence_omitted: Mapping[str, Sequence[str]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """The JSON record that says where one proposal came from.

    ``candidates`` are exactly the candidates this SKILL.md was generated
    from (one plan) and ``rejected`` the thread's classify rejections with
    why. ``thread_ids`` keep their order (the analysed thread first) minus
    duplicates; ``sources`` maps every cited file name to its path;
    ``category`` is the library folder a new skill is meant for and
    ``target`` names the skill an update revises (with ``base_sha256`` of
    its current file). ``policy`` is ``{requested, selection, source}``,
    ``config`` ``{requested, effective}``, ``versions`` is merged with the
    features and provenance versions, ``prompt_hashes`` covers the three
    stages, ``quality_review`` is the review record and ``verification``
    ``{level, note}`` with a level of :data:`VERIFICATION_LEVELS`.
    ``covering_skill`` names the library skill a demo proposal duplicates.
    ``generation_evidence`` / ``generation_evidence_omitted`` are the
    generation call's actual selection; when None they are recomputed
    without a budget. ``observed_procedure_source`` is derived from the
    bundle. Evidence text is stored redacted. ``generated_at`` defaults to
    now (UTC, ISO 8601). Raises ``ValueError`` for a demo proposal that is
    not a create, an update without ``base_sha256``, an unknown
    verification level, or incomplete ``policy`` / ``prompt_hashes``.
    """
    if not isinstance(demo, bool):
        raise ValueError(f"demo must be a bool, got {type(demo).__name__}")
    action = target.get("action")
    if demo and action != "create":
        raise ValueError(f"demo proposal must create a new skill, got target.action {action!r}")
    if action == "update" and not target.get("base_sha256"):
        raise ValueError("update provenance needs target.base_sha256")
    level = verification.get("level")
    if level not in VERIFICATION_LEVELS:
        raise ValueError(f"unknown verification level {level!r}")
    missing_policy = [key for key in POLICY_KEYS if key not in policy]
    if missing_policy:
        raise ValueError(f"policy is missing {missing_policy}")
    missing_hashes = [stage for stage in PROMPT_HASH_STAGES if stage not in prompt_hashes]
    if missing_hashes:
        raise ValueError(f"prompt_hashes is missing {missing_hashes}")

    ordered: list[str] = []
    for thread_id in thread_ids:
        if thread_id not in ordered:
            ordered.append(thread_id)
    cited = {ref for candidate in candidates for ref in candidate.evidence_refs}
    # Only what this proposal's candidates cite: a fragment shown to the
    # classifier but cited by another plan of the thread is not evidence here.
    shown = {fid: fragment for fid, fragment in bundle.shown.items() if fid in cited}
    shown_ids = {fragment.ref: fragment.id for fragment in shown.values()}
    episode_rows: list[dict[str, Any]] = []
    for outcome in bundle.episodes:
        episode = outcome.episode
        episode_rows.append(
            {
                "kind": episode.kind,
                "weight": episode.weight,
                "thread_id": episode.thread_id,
                "source": episode.source,
                "bundle_status": outcome.status,
                "evidence": [
                    {
                        "id": shown_ids.get(item.ref),
                        "source": item.ref.source,
                        "line": item.ref.line,
                        "seq": item.ref.seq,
                        "role": item.role,
                        "required": item.required,
                    }
                    for item in episode.evidence
                ],
            },
        )
    evidence_shown = {
        fragment.id: {
            "source": fragment.ref.source,
            "line": fragment.ref.line,
            "seq": fragment.ref.seq,
            "role": fragment.role,
            "required": fragment.required,
            "text": redact_secrets(fragment.text),
        }
        for fragment in shown.values()
    }
    # The physical lines of every observed_procedure episode this plan cites.
    cited_refs = {fragment.ref for fragment in shown.values()}
    observed_source = [
        {"source": item.ref.source, "line": item.ref.line, "seq": item.ref.seq, "role": item.role}
        for outcome in bundle.episodes
        if outcome.episode.kind == "observed_procedure"
        and any(item.ref in cited_refs for item in outcome.episode.evidence)
        for item in outcome.episode.evidence
    ]
    if generation_evidence is None:
        generation_evidence = {c.candidate_id: select_generation_evidence(c, shown).ids for c in candidates}
    if generation_evidence_omitted is None:
        generation_evidence_omitted = {candidate.candidate_id: [] for candidate in candidates}
    return {
        "provenance_version": PROVENANCE_VERSION,
        "thread_ids": ordered,
        "sources": dict(sources),
        "episodes": episode_rows,
        "evidence_shown": evidence_shown,
        "candidates": [candidate.model_dump() for candidate in candidates],
        "candidates_rejected": [
            {"title": candidate.title, "reason": reason, "evidence_refs": list(candidate.evidence_refs)}
            for candidate, reason in rejected
        ],
        "generation_evidence": {cid: list(ids) for cid, ids in generation_evidence.items()},
        "generation_evidence_omitted": {cid: list(ids) for cid, ids in generation_evidence_omitted.items()},
        "observed_procedure_source": observed_source,
        "model": model,
        "prompt_variants": dict(prompt_variants),
        "prompt_hashes": dict(prompt_hashes),
        "features_version": FEATURES_VERSION,
        "versions": {"features": FEATURES_VERSION, "provenance": PROVENANCE_VERSION, **versions},
        "generated_at": generated_at or _utc_now(),
        "category": category,
        "target": dict(target),
        "policy": dict(policy),
        "demo": demo,
        "config": dict(config),
        "quality_review": dict(quality_review),
        "verification": dict(verification),
        "covering_skill": dict(covering_skill) if covering_skill is not None else None,
    }


def batch_dir_name(thread_id: str) -> str:
    """Folder that groups the proposals of one thread (its id, made path-safe)."""
    name = _UNSAFE_RE.sub("-", thread_id.strip())[:64]
    if not _BATCH_RE.fullmatch(name):
        raise ValueError(f"unsafe thread id for a proposal folder: {thread_id!r}")
    return name


def _reserve_dir(base: Path, name: str) -> Path:
    """Create ``<base>/<name>`` or the first free ``<name>-N``; mkdir is atomic."""
    base.mkdir(parents=True, exist_ok=True)
    for suffix in range(1, MAX_COLLISIONS + 1):
        candidate = base / (name if suffix == 1 else f"{name}-{suffix}")
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"too many proposals named {name!r} under {base}")


def write_proposal(
    content: str,
    *,
    root: Path,
    name: str,
    provenance: Mapping[str, Any],
    thread_id: str,
) -> Path:
    """Write SKILL.md + provenance.json to ``<root>/.proposals/<thread>/<name>/``.

    Name collisions get ``-2``, ``-3``, ... suffixes. ``provenance`` must carry
    every key of :data:`REQUIRED_PROVENANCE_KEYS` with non-empty
    ``thread_ids`` and ``candidates``; a demo proposal must target a new
    skill; the package must pass :func:`scan_package` (else
    :class:`SecretsDetected`). All checks run before any directory is
    created; provenance is written before SKILL.md so a skill never exists
    without it. Returns the SKILL.md path.
    """
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"unsafe proposal name {name!r}")
    batch = batch_dir_name(thread_id)
    missing = sorted(REQUIRED_PROVENANCE_KEYS - set(provenance))
    if missing:
        raise ValueError(f"provenance is missing {missing}")
    if not provenance["thread_ids"] or not provenance["candidates"]:
        raise ValueError("provenance must list thread_ids and candidates")
    if provenance.get("demo") is True and (provenance.get("target") or {}).get("action") != "create":
        raise ValueError("demo proposal must target a new skill (create)")
    payload = json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True)
    text = content if content.endswith("\n") else content + "\n"
    findings = scan_package({SKILL_FILE: text, PROVENANCE_FILE: payload}, raw=provenance)
    if findings:
        raise SecretsDetected("proposal package contains possible secrets; nothing written: " + "; ".join(findings))
    skill_dir = _reserve_dir(root / PROPOSALS_DIR / batch, name)
    provenance_path = skill_dir / PROVENANCE_FILE
    provenance_path.write_text(payload + "\n", encoding="utf-8", newline="\n")
    (skill_dir / SKILL_FILE).write_text(text, encoding="utf-8", newline="\n")
    return skill_dir / SKILL_FILE
