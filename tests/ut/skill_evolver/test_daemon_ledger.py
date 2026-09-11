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

"""The idempotency ledger and the tick lock."""

from __future__ import annotations

from pathlib import Path

import pytest

from msagent.skill_evolver.daemon.ledger import (
    STATUS_FAILED,
    STATUS_MINED,
    STATUS_POISONED,
    Ledger,
    content_sha256,
    methodology_fingerprint,
)
from msagent.skill_evolver.daemon.lock import tick_lock

ALL_REASONS = ["content_change", "methodology_change"]
PROJECT = "proj-abc123"
THREAD = "thread-1"


def _record(ledger: Ledger, path: Path, *, sha: str, fingerprint: str, status: str = STATUS_MINED, **kwargs) -> str:
    return ledger.record(
        project_id=PROJECT,
        thread_id=THREAD,
        source_path=path,
        size_bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
        content_sha=sha,
        fingerprint=fingerprint,
        status=status,
        max_attempts=kwargs.pop("max_attempts", 3),
        **kwargs,
    )


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "Profiler_thread-1.jsonl"
    path.write_text('{"v":1}\n', encoding="utf-8")
    return path


def test_content_sha_changes_with_the_file(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text("a\n", encoding="utf-8")
    first = content_sha256(path)
    path.write_text("a\nb\n", encoding="utf-8")
    assert content_sha256(path) != first


def test_fingerprint_is_order_independent_but_value_sensitive() -> None:
    left = methodology_fingerprint({"features": 4, "classify": "aa", "policy": "strict_knowledge"})
    right = methodology_fingerprint({"policy": "strict_knowledge", "classify": "aa", "features": 4})
    assert left == right
    assert methodology_fingerprint({"features": 5, "classify": "aa", "policy": "strict_knowledge"}) != left


def test_first_run_mines_and_second_run_skips(tmp_path: Path, source: Path) -> None:
    with Ledger.open(tmp_path / "state") as ledger:
        assert ledger.verdict(None, content_sha="sha", fingerprint="fp", remine_on=ALL_REASONS).reason == "new"
        _record(ledger, source, sha="sha", fingerprint="fp")

        entry = ledger.entry(PROJECT, THREAD)
        assert entry is not None and entry.status == STATUS_MINED
        verdict = ledger.verdict(entry, content_sha="sha", fingerprint="fp", remine_on=ALL_REASONS)
        assert verdict.mine is False
        assert verdict.reason == "already mined"


def test_a_grown_thread_is_mined_again(tmp_path: Path, source: Path) -> None:
    """The user continued the conversation, so there is new evidence."""
    with Ledger.open(tmp_path / "state") as ledger:
        _record(ledger, source, sha="sha-1", fingerprint="fp")
        entry = ledger.entry(PROJECT, THREAD)
        verdict = ledger.verdict(entry, content_sha="sha-2", fingerprint="fp", remine_on=ALL_REASONS)
        assert verdict.mine is True
        assert verdict.reason == "content_change"


def test_new_methodology_re_mines(tmp_path: Path, source: Path) -> None:
    """New prompts or a new policy mean the recorded verdict no longer applies."""
    with Ledger.open(tmp_path / "state") as ledger:
        _record(ledger, source, sha="sha", fingerprint="fp-1")
        entry = ledger.entry(PROJECT, THREAD)
        verdict = ledger.verdict(entry, content_sha="sha", fingerprint="fp-2", remine_on=ALL_REASONS)
        assert verdict.mine is True
        assert verdict.reason == "methodology_change"


def test_remine_reasons_can_be_switched_off(tmp_path: Path, source: Path) -> None:
    with Ledger.open(tmp_path / "state") as ledger:
        _record(ledger, source, sha="sha", fingerprint="fp-1")
        entry = ledger.entry(PROJECT, THREAD)
        assert ledger.verdict(entry, content_sha="sha", fingerprint="fp-2", remine_on=["content_change"]).mine is False
        assert ledger.verdict(entry, content_sha="new", fingerprint="fp-1", remine_on=["content_change"]).mine is True


def test_repeated_failure_poisons_the_thread(tmp_path: Path, source: Path) -> None:
    """A genuinely broken trajectory must stop costing budget, loudly and once."""
    with Ledger.open(tmp_path / "state") as ledger:
        for expected in (STATUS_FAILED, STATUS_FAILED, STATUS_POISONED):
            stored = _record(ledger, source, sha="sha", fingerprint="fp", status=STATUS_FAILED, max_attempts=3)
            assert stored == expected
        entry = ledger.entry(PROJECT, THREAD)
        assert entry is not None and entry.attempts == 3
        verdict = ledger.verdict(entry, content_sha="sha", fingerprint="fp", remine_on=ALL_REASONS)
        assert verdict.mine is False
        assert verdict.reason == "poisoned"


def test_a_failed_thread_is_due_again(tmp_path: Path, source: Path) -> None:
    """Failures are often transient, so the next tick retries; poisoning is what stops it."""
    with Ledger.open(tmp_path / "state") as ledger:
        _record(ledger, source, sha="sha", fingerprint="fp", status=STATUS_FAILED)
        entry = ledger.entry(PROJECT, THREAD)
        verdict = ledger.verdict(entry, content_sha="sha", fingerprint="fp", remine_on=[])
        assert verdict.mine is True
        assert verdict.reason == "retry_after_failure"


def test_a_changed_file_resets_the_failure_count(tmp_path: Path, source: Path) -> None:
    """Failures count per file content: a rewritten thread deserves a fresh chance."""
    with Ledger.open(tmp_path / "state") as ledger:
        _record(ledger, source, sha="sha-1", fingerprint="fp", status=STATUS_FAILED, max_attempts=2)
        stored = _record(ledger, source, sha="sha-2", fingerprint="fp", status=STATUS_FAILED, max_attempts=2)
        assert stored == STATUS_FAILED
        entry = ledger.entry(PROJECT, THREAD)
        assert entry is not None and entry.attempts == 1


def test_record_keeps_first_seen_and_updates_the_rest(tmp_path: Path, source: Path) -> None:
    with Ledger.open(tmp_path / "state") as ledger:
        _record(ledger, source, sha="sha-1", fingerprint="fp", proposals=1, llm_calls=7)
        first = ledger.entry(PROJECT, THREAD)
        assert first is not None

        _record(ledger, source, sha="sha-2", fingerprint="fp", proposals=2, llm_calls=9)
        second = ledger.entry(PROJECT, THREAD)
        assert second is not None
        assert second.first_seen == first.first_seen
        assert (second.proposals, second.llm_calls, second.content_sha) == (2, 9, "sha-2")
        assert len(ledger.entries(PROJECT)) == 1


def test_tick_stamp_survives_reopening(tmp_path: Path) -> None:
    state = tmp_path / "state"
    with Ledger.open(state) as ledger:
        assert ledger.last_tick() == 0.0
        ledger.set_last_tick(1234.5)
    with Ledger.open(state) as ledger:
        assert ledger.last_tick() == 1234.5


def test_the_tick_lock_admits_one_holder(tmp_path: Path) -> None:
    """A second tick must find the lock busy instead of mining the same threads."""
    state = tmp_path / "state"
    with tick_lock(state) as first:
        assert first is True
        with tick_lock(state) as second:
            assert second is False
    # Released again once the first holder is done.
    with tick_lock(state) as third:
        assert third is True
