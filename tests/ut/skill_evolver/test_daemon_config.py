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

"""config.skill.daemon.yml: defaults, the env kill switch and loud failure."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from msagent.skill_evolver.daemon import config as daemon_config
from msagent.skill_evolver.daemon.config import (
    ENV_CONFIG_PATH,
    ENV_DISABLED,
    ENV_ENABLED,
    SkillDaemonConfig,
    SkillDaemonConfigError,
    is_daemon_enabled,
    load_daemon_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"


def _write(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


def test_packaged_default_is_off_and_complete() -> None:
    """The shipped file parses and keeps the daemon disabled until someone opts in."""
    config = load_daemon_config(force_reload=True)
    assert config.enabled is False
    assert config.is_active is False
    assert config.schedule.quiet_period_seconds == 900
    assert config.mining.policy == "strict_knowledge"
    assert config.mining.remine_on == ["content_change", "methodology_change"]


def test_user_file_wins_over_the_packaged_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(
        tmp_path / "config.skill.daemon.yml",
        """
        enabled: true
        schedule:
          quiet_period_seconds: 60
          max_threads_per_tick: 2
        mining:
          policy: reusable_workflow
          model: cheap
        """,
    )
    monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
    config = load_daemon_config(force_reload=True)
    assert config.enabled is True
    assert config.schedule.quiet_period_seconds == 60
    assert config.schedule.max_threads_per_tick == 2
    # Untouched sections keep their defaults.
    assert config.schedule.min_interval_seconds == 1800
    assert config.mining.policy == "reusable_workflow"
    assert config.mining.model == "cheap"
    assert daemon_config.config_source() == path


def test_disabled_env_beats_enabled_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A kill switch that can be out-voted is not a kill switch."""
    path = _write(tmp_path / "config.skill.daemon.yml", "enabled: true\n")
    monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
    load_daemon_config(force_reload=True)

    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_DISABLED, "1")
    assert is_daemon_enabled() is False
    assert load_daemon_config().enabled is False


def test_enabled_env_opts_in_without_editing_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path / "config.skill.daemon.yml", "enabled: false\n")
    monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
    load_daemon_config(force_reload=True)
    assert load_daemon_config().enabled is False

    monkeypatch.setenv(ENV_ENABLED, "yes")
    assert load_daemon_config().enabled is True
    # The flag is re-read on every call, so no reload is needed to turn it back off.
    monkeypatch.delenv(ENV_ENABLED)
    assert load_daemon_config().enabled is False


def test_unknown_key_is_reported_with_its_dotted_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No silent degradation to defaults: the run stops and names the key."""
    path = _write(
        tmp_path / "config.skill.daemon.yml",
        """
        enabled: true
        schedule:
          quiet_period_secondz: 60
        """,
    )
    monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
    with pytest.raises(SkillDaemonConfigError) as excinfo:
        load_daemon_config(force_reload=True)
    lines = excinfo.value.lines()
    assert any("schedule.quiet_period_secondz" in line for line in lines), lines


def test_out_of_range_value_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(
        tmp_path / "config.skill.daemon.yml",
        """
        schedule:
          max_llm_calls_per_tick: 0
        """,
    )
    monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
    with pytest.raises(SkillDaemonConfigError) as excinfo:
        load_daemon_config(force_reload=True)
    assert any("schedule.max_llm_calls_per_tick" in line for line in excinfo.value.lines())


def test_broken_yaml_raises_instead_of_defaulting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path / "config.skill.daemon.yml", "enabled: true\n  bad indent:\n")
    monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
    with pytest.raises(SkillDaemonConfigError):
        load_daemon_config(force_reload=True)


def test_empty_file_means_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty document is a valid configuration, unlike a malformed one."""
    path = _write(tmp_path / "config.skill.daemon.yml", "")
    monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
    assert load_daemon_config(force_reload=True) == SkillDaemonConfig()


def test_remines_on_reads_the_configured_reasons() -> None:
    config = SkillDaemonConfig.model_validate({"mining": {"remine_on": ["content_change"]}})
    assert config.remines_on("content_change") is True
    assert config.remines_on("methodology_change") is False


_ISOLATION_PROBE = """
import sys
import msagent.skill_evolver.daemon.config
import msagent.skill_evolver.daemon.ledger
import msagent.skill_evolver.daemon.discovery as discovery
import msagent.skill_evolver.daemon.inbox
assert discovery.__file__.startswith(sys.argv[1]), discovery.__file__
leaked = sorted(m for m in sys.modules if m.startswith(("langchain", "langgraph")))
assert not leaked, leaked
"""


def test_selection_modules_do_not_import_langchain() -> None:
    """Config, ledger, discovery and inbox must run in CI without an LLM stack (CLAUDE.md).

    The pytest process already has langchain loaded, so the check needs a fresh
    interpreter; PYTHONPATH must win over any wheel installed in the virtualenv.
    """
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", _ISOLATION_PROBE, str(SRC_DIR)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
