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

"""How a finished background tick reaches the user.

The daemon writes one small file per project; the CLI reads it at startup and prints a
single line. It is a notification only — proposals stay inactive and are still promoted
by hand through ``/skill-review`` (ARCHITECTURE_skill_evolve.md, section 9).

Reading is **fail-open** by design: a missing, truncated or hand-edited inbox must never
stop a user from starting a session. Writing is not — a tick that cannot record its own
result says so.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from msagent.core.logging import get_logger

logger = get_logger(__name__)

INBOX_VERSION = 1
INBOX_DIR_NAME = "skill-evolver"
INBOX_FILE_NAME = "daemon-inbox.json"
# Enough to explain what the daemon has been doing without growing without bound.
MAX_RECENT = 50


def inbox_path(state_dir: Path) -> Path:
    """``<project state>/skill-evolver/daemon-inbox.json``."""
    return Path(state_dir) / INBOX_DIR_NAME / INBOX_FILE_NAME


@dataclass(slots=True)
class Inbox:
    """What the daemon last did in one project."""

    last_tick: str = ""
    last_seen: str = ""
    recent: list[dict[str, Any]] = field(default_factory=list)

    def unseen(self) -> list[dict[str, Any]]:
        """Proposals written after the user last saw the banner."""
        if not self.last_seen:
            return list(self.recent)
        return [item for item in self.recent if str(item.get("at", "")) > self.last_seen]

    def as_payload(self) -> dict[str, Any]:
        return {
            "version": INBOX_VERSION,
            "last_tick": self.last_tick,
            "last_seen": self.last_seen,
            "recent": self.recent[-MAX_RECENT:],
        }


def read_inbox(state_dir: Path) -> Inbox:
    """The recorded inbox, or an empty one. Never raises: this runs at session start."""
    path = inbox_path(state_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Inbox()
    except (OSError, json.JSONDecodeError):
        logger.warning("skill-daemon: unreadable inbox %s; ignoring it", path, exc_info=True)
        return Inbox()
    if not isinstance(payload, dict):
        logger.warning("skill-daemon: inbox %s is not a mapping; ignoring it", path)
        return Inbox()
    recent = payload.get("recent")
    return Inbox(
        last_tick=str(payload.get("last_tick", "")),
        last_seen=str(payload.get("last_seen", "")),
        recent=[item for item in recent if isinstance(item, dict)] if isinstance(recent, list) else [],
    )


def write_inbox(state_dir: Path, inbox: Inbox) -> Path:
    """Replace the inbox atomically (temp + fsync + os.replace), 0600."""
    path = inbox_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(inbox.as_payload(), ensure_ascii=False, indent=2) + "\n"
    descriptor = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return path


def record_proposals(
    state_dir: Path,
    *,
    thread_id: str,
    proposals: list[Path],
    at: str | None = None,
) -> None:
    """Append this thread's proposals to the project inbox."""
    if not proposals:
        return
    stamp = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    inbox = read_inbox(state_dir)
    for proposal in proposals:
        inbox.recent.append(
            {
                "thread_id": thread_id,
                "skill": proposal.parent.name,
                "path": str(proposal),
                "at": stamp,
            }
        )
    inbox.recent = inbox.recent[-MAX_RECENT:]
    write_inbox(state_dir, inbox)


def stamp_tick(state_dir: Path, *, at: str | None = None) -> None:
    """Record that a tick ran, even when it wrote nothing."""
    inbox = read_inbox(state_dir)
    inbox.last_tick = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_inbox(state_dir, inbox)


def mark_seen(state_dir: Path, *, at: str | None = None) -> None:
    """Remember that the user has been shown the current state."""
    inbox = read_inbox(state_dir)
    inbox.last_seen = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_inbox(state_dir, inbox)


def pending_proposals(working_dir: Path) -> int:
    """How many proposals are waiting for review in this project."""
    from msagent.skill_evolver.config import load_skill_evolver_config, output_root

    config = load_skill_evolver_config()
    root = output_root(config, working_dir)
    return sum(1 for _ in (root / ".proposals").glob("*/*/SKILL.md"))


def notice_line(state_dir: Path, working_dir: Path) -> str | None:
    """The startup banner, or ``None`` when there is nothing to say."""
    inbox = read_inbox(state_dir)
    if not inbox.last_tick:
        return None
    pending = pending_proposals(working_dir)
    if pending == 0:
        return None
    fresh = len(inbox.unseen())
    what = f"{pending} proposal{'s' if pending != 1 else ''} waiting"
    if fresh:
        what = f"{what} ({fresh} new)"
    return f"Skill daemon: {what} - /skill-review"


def print_daemon_notice(sink: Any, state_dir: Path | None, working_dir: Path) -> None:
    """Print the banner if there is one. Fail-open: never blocks a session start."""
    if state_dir is None:
        return
    try:
        line = notice_line(Path(state_dir), Path(working_dir))
        if line is None:
            return
        sink.print_info(line)
        mark_seen(Path(state_dir))
    except Exception:
        logger.warning("skill-daemon: could not show the startup notice", exc_info=True)
