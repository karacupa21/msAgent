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

"""Entry point of the background miner: ``msagent-skill-daemon``.

``--once`` is the shape the scheduler wants: run a batch, print a summary, exit.
``--watch`` is the same batch in a sleep loop, for a host without a timer (WSL
without systemd, say). Both call the same ``run_once``.

Exit codes: 0 the tick ran (or was disabled, busy, or too soon), 2 the configuration
is broken, 1 anything unexpected. A thread that failed inside a healthy tick is a
warning, not a failed unit — the ledger and the decision report carry the detail.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from msagent.core.logging import get_logger
from msagent.skill_evolver.daemon.config import (
    ENV_CONFIG_PATH,
    SkillDaemonConfigError,
    reset_config_cache,
)
from msagent.skill_evolver.daemon.runner import TickResult, run_once

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2

DEFAULT_WATCH_INTERVAL = 3600


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msagent-skill-daemon",
        description="Mine recorded trajectories into SKILL.md proposals in the background.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="run a single tick and exit (the default)")
    mode.add_argument(
        "--watch",
        nargs="?",
        type=int,
        const=DEFAULT_WATCH_INTERVAL,
        metavar="SECONDS",
        help=f"loop, sleeping SECONDS between ticks (default {DEFAULT_WATCH_INTERVAL}); for hosts without a timer",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="select and gate threads without creating an LLM or touching the ledger",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore schedule.min_interval_seconds for this run",
    )
    parser.add_argument(
        "--project",
        metavar="DIR_OR_ID",
        help="restrict the tick to one project working directory or project id",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log to stderr at INFO level")
    return parser


def _configure_logging(verbose: bool) -> None:
    """Send the daemon log to stderr, where systemd collects it into the journal."""
    root = logging.getLogger()
    root.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.handlers = [handler]


def _report(result: TickResult) -> None:
    print(result.summary())
    for path in result.proposal_paths:
        print(f"  proposal: {path}")


async def _tick(args: argparse.Namespace) -> TickResult:
    return await run_once(
        dry_run=args.dry_run,
        force=args.force,
        project_filter=args.project,
    )


async def _watch(args: argparse.Namespace, interval: int) -> int:
    """Run ticks forever. Config is re-read each time, so edits apply without a restart."""
    while True:
        reset_config_cache()
        result = await _tick(args)
        _report(result)
        await asyncio.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        if args.watch is not None:
            return asyncio.run(_watch(args, args.watch))
        result = asyncio.run(_tick(args))
        _report(result)
        if result.failures:
            logger.warning("skill-daemon: %d threads failed; see the decision reports", result.failures)
        return EXIT_OK
    except SkillDaemonConfigError as exc:
        for line in exc.lines():
            print(f"error: {line}", file=sys.stderr)
        print(
            f"fix the file above (or point {ENV_CONFIG_PATH} elsewhere); nothing was mined",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    except KeyboardInterrupt:
        return EXIT_OK
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        logger.exception("skill-daemon tick failed")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
