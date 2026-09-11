#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""What the daemon has already mined, so a tick never pays for the same thread twice.

Nothing in the interactive commands records this: re-running ``/skill-mine`` re-mines
every selected thread and spends the LLM budget again. The ledger is a stdlib sqlite
file shared by all projects; it never imports langchain, so it is testable without an
LLM.

A thread is re-mined when its **content** changed (the user continued the thread, so
new evidence exists) or when the **methodology** changed (prompts, FEATURES_VERSION or
the effective policy moved, so the old verdict no longer describes what would happen).
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from msagent.core.logging import get_logger

logger = get_logger(__name__)

LEDGER_FILE_NAME = "ledger.sqlite3"
LEDGER_VERSION = 1

# Terminal outcomes of one thread, mirroring the categories of the /skill-mine summary.
STATUS_MINED = "mined"
STATUS_GATE_SKIP = "gate_skip"
STATUS_NOTHING = "nothing"
STATUS_REJECTED = "rejected"
STATUS_FAILED = "failed"
STATUS_POISONED = "poisoned"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mined (
    project_id  TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    source_path TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    mtime_ns    INTEGER NOT NULL,
    content_sha TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status      TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    proposals   INTEGER NOT NULL DEFAULT 0,
    llm_calls   INTEGER NOT NULL DEFAULT 0,
    first_seen  TEXT NOT NULL,
    last_run    TEXT NOT NULL,
    detail      TEXT,
    PRIMARY KEY (project_id, thread_id)
);
CREATE TABLE IF NOT EXISTS tick (
    id       INTEGER PRIMARY KEY CHECK (id = 1),
    last_run REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utc_now() -> str:
    """UTC ISO 8601, the timestamp format the decision reports use."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_sha256(path: Path) -> str:
    """Hash of the trajectory file as it stands right now."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def methodology_fingerprint(parts: dict[str, object]) -> str:
    """Stable hash of everything that decides *how* a thread would be mined.

    Fed the feature version, the three prompt hashes and the effective rules; a change
    in any of them means a re-run would not repeat the recorded verdict.
    """
    payload = "\n".join(f"{key}={parts[key]!r}" for key in sorted(parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One (project, thread) row."""

    project_id: str
    thread_id: str
    source_path: str
    size_bytes: int
    mtime_ns: int
    content_sha: str
    fingerprint: str
    status: str
    attempts: int
    proposals: int
    llm_calls: int
    first_seen: str
    last_run: str
    detail: str | None


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a candidate should be mined now, and why."""

    mine: bool
    reason: str

    @property
    def skipped(self) -> bool:
        return not self.mine


class Ledger:
    """The sqlite ledger. Open it once per tick and close it at the end."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    @contextmanager
    def open(cls, state_dir: Path) -> Iterator[Ledger]:
        """Open (creating it if needed) the ledger under ``state_dir``."""
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / LEDGER_FILE_NAME
        connection = sqlite3.connect(path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('ledger_version', ?)",
                (str(LEDGER_VERSION),),
            )
            yield cls(connection)
        finally:
            connection.close()

    # ---------------------------------------------------------------------- tick

    def last_tick(self) -> float:
        """Epoch seconds of the previous tick, 0.0 when there was none."""
        with closing(self._db.execute("SELECT last_run FROM tick WHERE id = 1")) as cursor:
            row = cursor.fetchone()
        return float(row["last_run"]) if row is not None else 0.0

    def set_last_tick(self, now: float) -> None:
        """Stamp the tick so ``min_interval_seconds`` can be enforced."""
        self._db.execute(
            "INSERT INTO tick (id, last_run) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET last_run = excluded.last_run",
            (float(now),),
        )

    # --------------------------------------------------------------------- rows

    def entry(self, project_id: str, thread_id: str) -> LedgerEntry | None:
        """The recorded row for one thread, or ``None``."""
        with closing(
            self._db.execute(
                "SELECT * FROM mined WHERE project_id = ? AND thread_id = ?",
                (project_id, thread_id),
            )
        ) as cursor:
            row = cursor.fetchone()
        return None if row is None else LedgerEntry(**dict(row))

    def entries(self, project_id: str) -> list[LedgerEntry]:
        """Every recorded row of one project, newest run first."""
        with closing(
            self._db.execute(
                "SELECT * FROM mined WHERE project_id = ? ORDER BY last_run DESC",
                (project_id,),
            )
        ) as cursor:
            return [LedgerEntry(**dict(row)) for row in cursor.fetchall()]

    def verdict(
        self,
        entry: LedgerEntry | None,
        *,
        content_sha: str,
        fingerprint: str,
        remine_on: list[str],
    ) -> Verdict:
        """Decide whether a candidate is worth an LLM budget right now."""
        if entry is None:
            return Verdict(True, "new")
        if entry.status == STATUS_POISONED:
            return Verdict(False, "poisoned")
        if entry.content_sha != content_sha:
            if "content_change" in remine_on:
                return Verdict(True, "content_change")
            return Verdict(False, "content changed but content_change is not in remine_on")
        if entry.fingerprint != fingerprint:
            if "methodology_change" in remine_on:
                return Verdict(True, "methodology_change")
            return Verdict(False, "methodology changed but methodology_change is not in remine_on")
        if entry.status == STATUS_FAILED:
            # A failure is often transient (a model endpoint down), so try again next tick.
            # record() counts consecutive failures of the same content and poisons the
            # row at max_attempts, which is what keeps this from looping forever.
            return Verdict(True, "retry_after_failure")
        return Verdict(False, f"already {entry.status}")

    def record(
        self,
        *,
        project_id: str,
        thread_id: str,
        source_path: Path,
        size_bytes: int,
        mtime_ns: int,
        content_sha: str,
        fingerprint: str,
        status: str,
        proposals: int = 0,
        llm_calls: int = 0,
        detail: str | None = None,
        max_attempts: int,
    ) -> str:
        """Write the outcome of one thread; returns the status actually stored.

        ``attempts`` counts consecutive failures of the same file. Once it reaches
        ``max_attempts`` the row is stored as ``poisoned`` and never retried, so a
        genuinely broken trajectory cannot burn the budget forever.
        """
        previous = self.entry(project_id, thread_id)
        same_file = previous is not None and previous.content_sha == content_sha
        attempts = (previous.attempts if previous is not None and same_file else 0) + (
            1 if status == STATUS_FAILED else 0
        )
        stored = status
        if status == STATUS_FAILED and attempts >= max_attempts:
            stored = STATUS_POISONED
            logger.error(
                "skill-daemon: %s failed %d times; excluding it from future ticks (%s)",
                source_path,
                attempts,
                detail or "no detail",
            )
        now = utc_now()
        first_seen = previous.first_seen if previous is not None else now
        self._db.execute(
            """
            INSERT INTO mined (project_id, thread_id, source_path, size_bytes, mtime_ns,
                               content_sha, fingerprint, status, attempts, proposals,
                               llm_calls, first_seen, last_run, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, thread_id) DO UPDATE SET
                source_path = excluded.source_path,
                size_bytes  = excluded.size_bytes,
                mtime_ns    = excluded.mtime_ns,
                content_sha = excluded.content_sha,
                fingerprint = excluded.fingerprint,
                status      = excluded.status,
                attempts    = excluded.attempts,
                proposals   = excluded.proposals,
                llm_calls   = excluded.llm_calls,
                last_run    = excluded.last_run,
                detail      = excluded.detail
            """,
            (
                project_id,
                thread_id,
                str(source_path),
                size_bytes,
                mtime_ns,
                content_sha,
                fingerprint,
                stored,
                attempts,
                proposals,
                llm_calls,
                first_seen,
                now,
                detail,
            ),
        )
        return stored
