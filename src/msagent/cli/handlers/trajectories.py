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

"""/trajectories: browse the JSONL trajectories recorded for this project.

A read-only wrapper around ``trajectory_recorder.export``: ``list`` renders
the summaries of every recorded thread, ``show`` renders one thread as
markdown. Nothing is written and no LLM is involved. In the shared store
(``output.scope: shared``) both see only the threads of this workspace.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich import box
from rich.markup import escape
from rich.table import Table

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import console
from msagent.core.logging import get_logger
from msagent.trajectory_recorder.export import (
    TrajectorySummary,
    find_trajectory_file,
    list_trajectories,
    render_markdown,
    resolve_trajectories_dir,
    workspace_filter,
)
from msagent.trajectory_recorder.reader import iter_events
from msagent.utils.time import format_relative_time

logger = get_logger(__name__)

USAGE = "/trajectories [list | show <thread-id>]"
# Thread ids are uuid4; this prefix stays unique in practice and is still a
# valid --thread / show argument (find_trajectory_file resolves prefixes).
THREAD_ID_WIDTH = 12


def short_thread(thread_id: str) -> str:
    """Clip a thread id to a still-resolvable prefix."""
    if len(thread_id) <= THREAD_ID_WIDTH:
        return thread_id
    return f"{thread_id[:THREAD_ID_WIDTH]}…"


def build_summary_table(summaries: list[TrajectorySummary]) -> Table:
    """Render recorded trajectories, newest first."""
    table = Table(
        box=box.SIMPLE_HEAD,
        title="Recorded trajectories",
        title_style="accent",
        title_justify="left",
        header_style="accent",
        border_style="border",
        expand=False,
        pad_edge=False,
    )
    table.add_column("thread", style="command", no_wrap=True)
    table.add_column("agent", style="secondary", no_wrap=True)
    table.add_column("turns", justify="right", style="muted")
    table.add_column("events", justify="right", style="muted")
    table.add_column("size", justify="right", style="muted", no_wrap=True)
    table.add_column("modified", style="timestamp", no_wrap=True)
    table.add_column("first message", style="default", overflow="ellipsis")
    for summary in summaries:
        modified = format_relative_time(summary.path.stat().st_mtime)
        table.add_row(
            escape(short_thread(summary.thread_id)),
            escape(summary.agent or "unknown"),
            str(summary.turns),
            str(summary.events),
            f"{summary.size_bytes / 1024:.1f}KB",
            escape(modified),
            escape(summary.first_user_message or "(no user message)"),
        )
    return table


class TrajectoriesHandler:
    """Browse the recorded trajectories of the current project."""

    def __init__(self, session) -> None:
        self.session = session

    async def handle(self, args: list[str]) -> None:
        """List recorded trajectories, or show one: [list | show <thread-id>]."""
        try:
            action = args[0].lower() if args else "list"
            if action == "list":
                await self._list()
            elif action == "show":
                await self._show(args[1:])
            else:
                console.print_error(f"Unknown subcommand: {escape(action)}")
                console.print(f"[muted]{escape(USAGE)}[/muted]")
                console.print("")
        except Exception as exc:
            console.print_error(escape(f"Error reading trajectories: {exc}"))
            console.print("")
            logger.exception("Trajectory browsing failed")

    def _trajectories_dir(self) -> Path:
        """Directory holding this project's recorded trajectories."""
        ctx = self.session.context
        state_dir = initializer.get_project_paths(Path(ctx.working_dir)).root
        return resolve_trajectories_dir(state_dir=state_dir)

    def _workspace(self) -> Path | None:
        """This workspace when the store is shared by all of them, else ``None``."""
        return workspace_filter(Path(self.session.context.working_dir))

    @staticmethod
    def _where(directory: Path, workspace: Path | None) -> str:
        """Where the lookup happened, naming the workspace in the shared store."""
        if workspace is None:
            return f"in {directory}"
        return f"in {directory} for workspace {workspace}"

    async def _list(self) -> None:
        """Print one row per recorded thread, newest first."""
        directory = self._trajectories_dir()
        workspace = self._workspace()
        summaries = await asyncio.to_thread(
            list_trajectories,
            directory,
            workspace=workspace,
        )
        if not summaries:
            where = self._where(directory, workspace)
            console.print_info(f"No trajectories recorded {where}")
            console.print("")
            return
        console.console.print(build_summary_table(summaries))
        console.print("")

    async def _show(self, args: list[str]) -> None:
        """Render one thread as markdown; the id may be a unique prefix."""
        if not args:
            console.print_error("show requires a thread id")
            console.print(f"[muted]{escape(USAGE)}[/muted]")
            console.print("")
            return

        thread_id = args[0].strip()
        directory = self._trajectories_dir()
        workspace = self._workspace()
        path = find_trajectory_file(directory, thread_id, workspace=workspace)
        if path is None:
            message = " ".join(
                (
                    f"No recorded trajectory for thread '{thread_id}'",
                    f"{self._where(directory, workspace)};",
                    "give a full id or a unique prefix",
                ),
            )
            console.print_error(escape(message))
            console.print("")
            return

        text = await asyncio.to_thread(self._render, path)
        # Markdown, not console.print: a trajectory may contain square
        # brackets, which the console would parse as style markup.
        from msagent.cli.ui.renderer import TransparentMarkdown

        console.console.print(TransparentMarkdown(text))
        console.print("")

    @staticmethod
    def _render(path: Path) -> str:
        """Markdown of one trajectory file."""
        return render_markdown(iter_events(path))
