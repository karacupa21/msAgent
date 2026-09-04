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

"""Locate trajectory files. The only module that imports trajectory_recorder."""

from __future__ import annotations

from pathlib import Path

from msagent.trajectory_recorder.export import find_trajectory_file, resolve_trajectories_dir
from msagent.trajectory_recorder.model import Trajectory
from msagent.trajectory_recorder.reader import load_trajectory


def resolve_graph_dir(*, working_dir: Path | None = None, state_dir: Path | None = None) -> Path:
    """Locate the experience-graph directory for a project."""
    from msagent.exgraph.config import load_exgraph_config

    config = load_exgraph_config()
    directory = Path(config.output.directory).expanduser()
    if directory.is_absolute():
        return directory
    if state_dir is None:
        from msagent.core.paths import AppPaths

        state_dir = AppPaths.resolve().for_project(working_dir or Path.cwd()).root
    return Path(state_dir) / directory


def load_source_trajectory(
    *,
    thread_id: str,
    working_dir: Path | None = None,
    state_dir: Path | None = None,
    path: Path | None = None,
) -> tuple[Path, Trajectory]:
    """Load one recorded trajectory by path or thread id."""
    if path is not None:
        source = Path(path)
        return source, load_trajectory(source)
    trajectories_dir = resolve_trajectories_dir(working_dir=working_dir, state_dir=state_dir)
    source = find_trajectory_file(trajectories_dir, thread_id)
    if source is None:
        raise FileNotFoundError(
            f"No recorded trajectory for thread '{thread_id}' in {trajectories_dir}",
        )
    return source, load_trajectory(source)
