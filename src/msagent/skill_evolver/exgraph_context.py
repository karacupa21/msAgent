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


"""Fail-open bridge: stored experience graph → classify evidence text.

Modes (``config.exgraph.yml`` / ``MSAGENT_EXGRAPH_EVIDENCE_MODE``):

- ``episodes`` — original bundle, no shard write (evolver first pass)
- ``hybrid`` — persist shard, append relation appendix (default when on)
- ``graph`` — persist shard, relations first, bundle as supporting tail

The last-N trajectory pool is untouched. This module never calls
``load_trajectories``. Failures return the original bundle.
"""

from __future__ import annotations

import logging
from pathlib import Path

from msagent.trajectory_recorder.model import Trajectory

logger = logging.getLogger(__name__)


def attach_stored_graph(
    bundle_text: str,
    trajectory: Trajectory,
    *,
    working_dir: Path | None = None,
    state_dir: Path | None = None,
) -> str:
    """Apply ``evidence_mode`` to the classify string. Never raises."""
    try:
        from msagent.exgraph.config import resolve_evidence_mode

        mode = resolve_evidence_mode()
        if mode == "episodes":
            return bundle_text
        from msagent.exgraph.enrich import remember_thread

        remember_thread(trajectory, working_dir=working_dir, state_dir=state_dir)
        if mode == "graph":
            from msagent.exgraph.consumer import render_graph_primary

            extra = render_graph_primary(
                trajectory.thread_id,
                working_dir=working_dir,
                state_dir=state_dir,
            )
            if not extra:
                return bundle_text
            if not bundle_text.strip():
                return extra
            return extra + "\n\n## Episode bundle (supporting, citable)\n\n" + bundle_text
        from msagent.exgraph.consumer import render_thread_context

        extra = render_thread_context(
            trajectory.thread_id,
            working_dir=working_dir,
            state_dir=state_dir,
        )
    except Exception:
        logger.debug("exgraph context skipped", exc_info=True)
        return bundle_text
    if not extra:
        return bundle_text
    return bundle_text + "\n\n" + extra
