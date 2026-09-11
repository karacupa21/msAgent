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

"""One tick at a time: an advisory whole-file lock held for the run.

The kernel drops the lock when the process dies, so a killed tick leaves no stale
lock behind — which a pid file could not promise. Linux is the supported deployment;
the Windows branch exists so the module imports and the tests run there.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from msagent.core.logging import get_logger

logger = get_logger(__name__)

LOCK_FILE_NAME = "tick.lock"

_WINDOWS = sys.platform == "win32"

if _WINDOWS:  # pragma: no cover - exercised only on Windows
    import msvcrt
else:
    import fcntl


def _try_lock(handle) -> bool:
    """Take an exclusive, non-blocking lock on an open file. False when held elsewhere."""
    try:
        if _WINDOWS:  # pragma: no cover - exercised only on Windows
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle) -> None:
    """Release the lock; closing the handle would do it too, this is explicit."""
    try:
        if _WINDOWS:  # pragma: no cover - exercised only on Windows
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        logger.warning("Could not release the skill-daemon tick lock", exc_info=True)


@contextmanager
def tick_lock(state_dir: Path) -> Iterator[bool]:
    """Yield True when this process owns the tick, False when another one does.

    A busy lock is not an error: the previous tick is still running, and the
    scheduler will call again.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / LOCK_FILE_NAME
    handle = path.open("a+", encoding="utf-8")
    # msvcrt.locking locks one byte at the current position, so pin it before locking
    # and again before unlocking; flock ignores the position.
    handle.seek(0)
    acquired = _try_lock(handle)
    try:
        if acquired:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
        yield acquired
    finally:
        if acquired:
            _unlock(handle)
        handle.close()
