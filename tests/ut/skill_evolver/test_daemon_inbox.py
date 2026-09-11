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

"""The startup banner: what the daemon leaves behind and how the CLI reads it."""

from __future__ import annotations

import json
from pathlib import Path

from msagent.skill_evolver.daemon.inbox import (
    inbox_path,
    mark_seen,
    notice_line,
    pending_proposals,
    print_daemon_notice,
    read_inbox,
    record_proposals,
    stamp_tick,
)


class _Sink:
    def __init__(self) -> None:
        self.info: list[str] = []

    def print_info(self, content: str) -> None:
        self.info.append(content)


def _proposal(working_dir: Path, thread: str, name: str) -> Path:
    path = working_dir / "skills" / ".proposals" / thread / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\nname: x\n---\n", encoding="utf-8")
    return path


def test_a_missing_inbox_reads_as_empty(tmp_path: Path) -> None:
    inbox = read_inbox(tmp_path)
    assert inbox.last_tick == "" and inbox.recent == []


def test_a_corrupt_inbox_never_raises(tmp_path: Path) -> None:
    """This runs at session start; a hand-edited file must not block the CLI."""
    path = inbox_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert read_inbox(tmp_path).recent == []


def test_recorded_proposals_survive_a_round_trip(tmp_path: Path) -> None:
    work = tmp_path / "work"
    proposal = _proposal(work, "thread-1", "demo-csv")
    record_proposals(tmp_path, thread_id="thread-1", proposals=[proposal], at="2026-09-10T10:00:00+00:00")

    inbox = read_inbox(tmp_path)
    assert [item["skill"] for item in inbox.recent] == ["demo-csv"]
    assert inbox.recent[0]["path"] == str(proposal)
    assert json.loads(inbox_path(tmp_path).read_text(encoding="utf-8"))["version"] == 1


def test_unseen_tracks_what_the_user_has_already_been_shown(tmp_path: Path) -> None:
    work = tmp_path / "work"
    record_proposals(
        tmp_path,
        thread_id="t1",
        proposals=[_proposal(work, "t1", "a")],
        at="2026-09-10T10:00:00+00:00",
    )
    mark_seen(tmp_path, at="2026-09-10T11:00:00+00:00")
    assert read_inbox(tmp_path).unseen() == []

    record_proposals(
        tmp_path,
        thread_id="t2",
        proposals=[_proposal(work, "t2", "b")],
        at="2026-09-10T12:00:00+00:00",
    )
    assert [item["skill"] for item in read_inbox(tmp_path).unseen()] == ["b"]


def test_pending_counts_proposals_on_disk(tmp_path: Path) -> None:
    work = tmp_path / "work"
    assert pending_proposals(work) == 0
    _proposal(work, "t1", "a")
    _proposal(work, "t1", "b")
    assert pending_proposals(work) == 2


def test_no_banner_before_the_daemon_has_ever_run(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _proposal(work, "t1", "a")
    assert notice_line(tmp_path, work) is None


def test_no_banner_when_nothing_is_waiting(tmp_path: Path) -> None:
    stamp_tick(tmp_path, at="2026-09-10T10:00:00+00:00")
    assert notice_line(tmp_path, tmp_path / "work") is None


def test_the_banner_names_the_totals(tmp_path: Path) -> None:
    work = tmp_path / "work"
    stamp_tick(tmp_path, at="2026-09-10T10:00:00+00:00")
    record_proposals(tmp_path, thread_id="t1", proposals=[_proposal(work, "t1", "a")], at="2026-09-10T10:00:01+00:00")
    _proposal(work, "t2", "b")

    line = notice_line(tmp_path, work)
    assert line is not None
    assert "2 proposals waiting" in line and "(1 new)" in line and "/skill-review" in line


def test_printing_the_banner_marks_it_seen(tmp_path: Path) -> None:
    work = tmp_path / "work"
    stamp_tick(tmp_path, at="2026-09-10T10:00:00+00:00")
    record_proposals(tmp_path, thread_id="t1", proposals=[_proposal(work, "t1", "a")], at="2026-09-10T10:00:01+00:00")

    sink = _Sink()
    print_daemon_notice(sink, tmp_path, work)
    assert len(sink.info) == 1 and "(1 new)" in sink.info[0]

    # Second start: still waiting, but nothing is new any more.
    sink = _Sink()
    print_daemon_notice(sink, tmp_path, work)
    assert len(sink.info) == 1 and "new" not in sink.info[0]


def test_the_banner_is_fail_open(tmp_path: Path) -> None:
    """A session must start even when the state directory is unusable."""
    sink = _Sink()
    print_daemon_notice(sink, None, tmp_path)
    print_daemon_notice(sink, tmp_path / "does-not-exist", tmp_path / "work")
    assert sink.info == []
