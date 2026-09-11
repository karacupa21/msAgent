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

"""One tick of the background miner: budget, idempotency and the real pipeline.

The end-to-end test drives the same synthetic demo fixture the interactive commands
use, on a scripted LLM, and checks that a background run produces an ordinary
inactive proposal plus a decision report that says it came from the daemon.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# The handlers package must be initialized before the evolver modules are imported
# directly (a pre-existing import cycle the CLI never triggers).
import msagent.cli.handlers  # noqa: F401
from msagent.cli.bootstrap.initializer import initializer
from msagent.skill_evolver import mining as mining_module
from msagent.skill_evolver.bundle import build_evidence_bundle
from msagent.skill_evolver.config import DirectSkillGenerationConfig
from msagent.skill_evolver.daemon import runner as runner_module
from msagent.skill_evolver.daemon.config import ENV_CONFIG_PATH, ENV_DISABLED, ENV_ENABLED, daemon_state_dir
from msagent.skill_evolver.daemon.inbox import read_inbox
from msagent.skill_evolver.daemon.ledger import STATUS_MINED, Ledger
from msagent.skill_evolver.daemon.runner import STATUS_BUSY, STATUS_DISABLED, STATUS_TOO_SOON, run_once
from msagent.skill_evolver.daemon.lock import tick_lock
from msagent.skill_evolver.direct_skill_generation import DirectSkillGenerationHandler
from msagent.skill_evolver.features import extract_episodes
from msagent.skill_evolver.mining import SkillMiningHandler
from msagent.skill_evolver.pipeline import PlanTally
from msagent.trajectory_recorder.config import reset_config_cache
from msagent.trajectory_recorder.reader import load_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "trajectories"
DEMO = FIXTURES / "skill_evolver_demo_success.jsonl"
DEMO_THREAD_ID = "thread-demo-synthetic"
DEMO_AGENT = "SyntheticDemo"
DEMO_SKILL_NAME = "demo-csv-column-sum"

CLASSIFY_TEMPLATE = "Library:\n{skill_library}\n\nPolicy:\n{selection_policy}\n\nBundle:\n{evidence_bundle}\n"
RENDER_TEMPLATE = "Policy:\n{render_policy}\n\nCandidates:\n{candidates}\n\nExisting:\n{existing_skill}\n"
REVIEW_TEMPLATE = (
    "Policy:\n{review_policy}\n\nSkill:\n{skill_md}\n\nCandidates:\n{candidates}\n\n"
    "Evidence:\n{evidence}\n\nExisting:\n{existing_skill}\n"
)
REVIEW_PASS = json.dumps({"verdict": "pass", "issues": []})
SUM_COMMAND = (
    'python3 -c "import csv,sys; print(sum(float(r[sys.argv[2]]) for r in csv.DictReader(open(sys.argv[1]))))"'
)
DEMO_SKILL = "\n".join(
    [
        "---",
        f"name: {DEMO_SKILL_NAME}",
        "description: Use when a CSV column must be summed and the total checked against an expected value.",
        "---",
        "",
        "# CSV column sum",
        "",
        "## Inputs",
        "",
        "- `<csv-path>`: a CSV file with a header row.",
        "- `<column>`: the numeric column to sum.",
        "- `<expected-total>`: the number the total is compared with.",
        "",
        "## Workflow",
        "",
        f"1. Run `{SUM_COMMAND} <csv-path> <column>` and compare the printed number with `<expected-total>`.",
        "",
        "## Outputs",
        "",
        "The column total, compared with the expected value.",
        "",
    ]
)

NOW = 1_800_000_000.0
QUIET = 60


class _NullStatus:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _ConsoleSpy:
    """Enough of the themed console for the pipeline and the mining loop."""

    def __init__(self) -> None:
        self.info: list[str] = []
        self.success: list[str] = []
        self.warning: list[str] = []
        self.error: list[str] = []
        self.plain: list[str] = []
        self.console = SimpleNamespace(status=lambda *_a, **_k: _NullStatus(), print=lambda *_a, **_k: None)

    def print(self, *args, **_kwargs) -> None:
        self.plain.extend(str(arg) for arg in args)

    def print_info(self, content: str) -> None:
        self.info.append(content)

    def print_success(self, content: str) -> None:
        self.success.append(content)

    def print_warning(self, content: str) -> None:
        self.warning.append(content)

    def print_error(self, content: str) -> None:
        self.error.append(content)


class _FakeContext:
    """Context.create without an agents config: the pipeline only reads three fields."""

    @classmethod
    async def create(cls, *, agent, model, approval_mode=None, working_dir, stream_output=True, **_kwargs):
        return SimpleNamespace(
            agent=agent,
            thread_id="daemon",
            working_dir=working_dir,
            model=model or "default",
        )


def _boom(*_args, **_kwargs):
    raise AssertionError("the LLM must not be created")


async def _fake_stage_prompt(_self, _root, _cfg, stage):
    templates = {"classify": CLASSIFY_TEMPLATE, "render": RENDER_TEMPLATE, "review": REVIEW_TEMPLATE}
    return templates[stage], f"packaged/{stage}/prompt_v2.md"


class _Env:
    """The world one tick sees: one project, one aged trajectory, a scripted LLM."""

    def __init__(self, tmp_path: Path, spy: _ConsoleSpy, fake_llm_cls) -> None:
        self.working_dir = (tmp_path / "work").resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.spy = spy
        self.fake_llm_cls = fake_llm_cls
        self.llm = fake_llm_cls()
        self.config = DirectSkillGenerationConfig(demo_mode=True, policy="reusable_workflow")
        self.skills: list[Any] = []
        self.state_dir = initializer.get_project_paths(self.working_dir).root
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "project.json").write_text(
            json.dumps({"working_dir": str(self.working_dir)}),
            encoding="utf-8",
        )
        self.trajectories_dir = self.state_dir / "trajectories"
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)
        self.source = self.install()

    def install(self, *, age: float = QUIET + 60) -> Path:
        target = self.trajectories_dir / f"{DEMO_AGENT}_{DEMO_THREAD_ID}.jsonl"
        shutil.copy(DEMO, target)
        self.age(target, age)
        return target

    @staticmethod
    def age(path: Path, age: float) -> None:
        stamp = NOW - age
        os.utime(path, (stamp, stamp))

    def append_turn(self, run_id: str = "run-d2") -> None:
        """Grow the thread the way a continued session would: one more complete turn."""

        def event(seq: int, kind: str, **fields: Any) -> str:
            base = {
                "v": 1,
                "event": kind,
                "ts": f"2026-09-01T10:10:{seq:02d}.000+00:00",
                "seq": seq,
                "rec": "rec-d",
                "thread_id": DEMO_THREAD_ID,
                "agent": DEMO_AGENT,
                "run_id": run_id,
            }
            base.update(fields)
            return json.dumps(base, ensure_ascii=False)

        lines = [
            event(90, "turn.start", source="dispatch", model="default", approval_mode="active", user_message="again"),
            event(91, "turn.end", status="completed", duration_ms=5),
        ]
        with self.source.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def script(self, *replies: str) -> None:
        self.llm = self.fake_llm_cls(*replies)

    def demo_replies(self, skill: str = DEMO_SKILL) -> tuple[str, str, str]:
        """classify (citing the fixture's real fragment ids), render, review."""
        trajectory = load_trajectory(self.source)
        bundle = build_evidence_bundle(
            extract_episodes(trajectory, demo=True),
            [trajectory],
            max_chars=self.config.bundle_max_chars,
            excerpt_chars=self.config.excerpt_max_chars,
            demo=True,
        )
        episode_id, episode = next(
            (eid, ep) for eid, ep in bundle.episode_ids.items() if ep.kind == "observed_procedure"
        )
        refs = {item.ref for item in episode.evidence}
        fragments = [fid for fid, fragment in bundle.shown.items() if fragment.ref in refs]
        classify = json.dumps(
            {
                "contract_version": 2,
                "verdict": "save",
                "candidates": [
                    {
                        "title": "CSV column sum",
                        "rule": "Sum a CSV column with csv.DictReader and compare the total with the expected value.",
                        "evidence_refs": fragments,
                        "future_applicability": "high",
                        "target": {"action": "create", "existing_skill": None},
                    }
                ],
                "decisions": [
                    {
                        "episode_ids": [episode_id],
                        "decision": "accept",
                        "reason_code": "accepted",
                        "explanation": "scripted",
                        "evidence_refs": [],
                    }
                ],
            }
        )
        return classify, skill, REVIEW_PASS

    def proposal(self, name: str = DEMO_SKILL_NAME) -> Path:
        return self.working_dir / "skills" / ".proposals" / DEMO_THREAD_ID / name / "SKILL.md"

    def reports(self) -> list[dict[str, Any]]:
        folder = self.state_dir / "skill-evolver" / "decisions"
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]

    def ledger_entry(self):
        with Ledger.open(daemon_state_dir()) as ledger:
            return ledger.entry(self.state_dir.name, DEMO_THREAD_ID)


def _daemon_config(tmp_path: Path, **overrides: Any) -> Path:
    payload = {
        "enabled": True,
        "schedule": {
            "quiet_period_seconds": QUIET,
            "min_interval_seconds": 0,
            "max_threads_per_tick": 5,
            "max_llm_calls_per_tick": 60,
        },
        "mining": {"policy": "reusable_workflow", "demo": True},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key].update(value)
        else:
            payload[key] = value
    path = tmp_path / "config.skill.daemon.yml"
    path.write_text(json.dumps(payload), encoding="utf-8")  # JSON is valid YAML
    return path


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_llm_cls):
    monkeypatch.delenv("MSAGENT_TRAJECTORY_CONFIG", raising=False)
    monkeypatch.delenv(ENV_DISABLED, raising=False)
    monkeypatch.delenv(ENV_ENABLED, raising=False)
    reset_config_cache()

    spy = _ConsoleSpy()
    state = _Env(tmp_path, spy, fake_llm_cls)
    monkeypatch.setenv(ENV_CONFIG_PATH, str(_daemon_config(tmp_path)))
    monkeypatch.setattr(runner_module, "console", spy)
    monkeypatch.setattr(mining_module, "console", spy)
    monkeypatch.setattr(runner_module, "Context", _FakeContext)

    async def fake_load_skills(_self):
        return list(state.skills)

    async def fake_load_llm_config(_model, _working_dir):
        return SimpleNamespace(model="fake-model", context_window=128_000)

    monkeypatch.setattr(DirectSkillGenerationHandler, "_load_config", staticmethod(lambda: state.config))
    monkeypatch.setattr(DirectSkillGenerationHandler, "_load_stage_prompt", _fake_stage_prompt)
    monkeypatch.setattr(SkillMiningHandler, "_load_skills", fake_load_skills)
    monkeypatch.setattr(initializer, "load_llm_config", fake_load_llm_config)
    monkeypatch.setattr(initializer.llm_factory, "create", lambda _config: state.llm)
    yield state
    reset_config_cache()


# ------------------------------------------------------------- refusal paths


@pytest.mark.asyncio
async def test_a_disabled_daemon_scans_nothing(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    monkeypatch.setenv(ENV_DISABLED, "1")
    result = await run_once(now=NOW)
    assert result.status == STATUS_DISABLED
    assert result.threads == 0


@pytest.mark.asyncio
async def test_a_busy_lock_ends_the_tick(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two schedulers must not mine the same threads twice over."""
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    with tick_lock(daemon_state_dir()) as held:
        assert held is True
        result = await run_once(now=NOW)
    assert result.status == STATUS_BUSY


@pytest.mark.asyncio
async def test_min_interval_refuses_an_early_tick(env: _Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    monkeypatch.setenv(ENV_CONFIG_PATH, str(_daemon_config(tmp_path, schedule={"min_interval_seconds": 3600})))
    with Ledger.open(daemon_state_dir()) as ledger:
        ledger.set_last_tick(NOW - 10)
    result = await run_once(now=NOW)
    assert result.status == STATUS_TOO_SOON
    assert "3600s" in result.detail


@pytest.mark.asyncio
async def test_force_overrides_min_interval(env: _Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_CONFIG_PATH, str(_daemon_config(tmp_path, schedule={"min_interval_seconds": 3600})))
    with Ledger.open(daemon_state_dir()) as ledger:
        ledger.set_last_tick(NOW - 10)
    env.script(*env.demo_replies())
    result = await run_once(now=NOW, force=True)
    assert result.ran is True
    assert result.proposals == 1


@pytest.mark.asyncio
async def test_a_dry_run_never_creates_an_llm(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    result = await run_once(now=NOW, dry_run=True)
    assert result.ran is True
    assert result.threads == 1
    assert result.proposals == 0
    # Nothing was recorded, so the next real tick still has work to do.
    assert env.ledger_entry() is None


@pytest.mark.asyncio
async def test_a_live_trajectory_is_left_alone(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file written seconds ago belongs to a session that may still be going."""
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    env.age(env.source, 1)
    result = await run_once(now=NOW)
    assert result.ran is True
    assert result.threads == 0


# ------------------------------------------------------------- the real tick


@pytest.mark.asyncio
async def test_a_tick_writes_an_inactive_proposal_and_records_itself(env: _Env) -> None:
    env.script(*env.demo_replies())
    result = await run_once(now=NOW)

    assert result.ran is True
    assert (result.threads, result.proposals, result.failures) == (1, 1, 0)
    proposal = env.proposal()
    assert proposal.is_file()
    assert proposal.read_text(encoding="utf-8").startswith("---")
    # Still invisible to every skill scanner: it lives under a dot-directory.
    assert ".proposals" in proposal.parts

    entry = env.ledger_entry()
    assert entry is not None
    assert (entry.status, entry.proposals) == (STATUS_MINED, 1)
    assert entry.llm_calls == 3

    report = env.reports()[-1]
    assert report["command"] == "skill-daemon"
    assert report["thread_id"] == DEMO_THREAD_ID
    assert report["proposals"] == [str(proposal)]


@pytest.mark.asyncio
async def test_a_second_tick_is_a_no_op(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of the ledger: never pay for the same thread twice."""
    env.script(*env.demo_replies())
    first = await run_once(now=NOW)
    assert first.proposals == 1

    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    second = await run_once(now=NOW + 1)
    assert second.ran is True
    assert (second.threads, second.proposals) == (0, 0)


@pytest.mark.asyncio
async def test_a_grown_trajectory_is_mined_again(env: _Env) -> None:
    env.script(*env.demo_replies())
    assert (await run_once(now=NOW)).proposals == 1

    env.append_turn()
    env.age(env.source, QUIET + 60)
    env.script(*env.demo_replies())
    again = await run_once(now=NOW + 1)
    assert again.threads == 1
    assert again.proposals == 1


@pytest.mark.asyncio
async def test_the_budget_stops_the_tick_before_it_overspends(
    env: _Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tick may not start a thread it cannot afford in the worst case."""
    monkeypatch.setenv(ENV_CONFIG_PATH, str(_daemon_config(tmp_path, schedule={"max_llm_calls_per_tick": 1})))
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    result = await run_once(now=NOW)
    assert result.ran is True
    assert result.stopped_by_budget is True
    assert result.proposals == 0
    assert env.ledger_entry() is None


@pytest.mark.asyncio
async def test_max_threads_per_tick_caps_the_batch(env: _Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_CONFIG_PATH, str(_daemon_config(tmp_path, schedule={"max_threads_per_tick": 1})))
    second = env.trajectories_dir / f"{DEMO_AGENT}_thread-demo-two.jsonl"
    second.write_text(
        env.source.read_text(encoding="utf-8").replace(DEMO_THREAD_ID, "thread-demo-two"),
        encoding="utf-8",
    )
    env.age(second, QUIET + 30)

    calls: list[str] = []

    async def fake_mine_thread(_self, item, *, run, position, total):
        calls.append(item.thread_id)
        return PlanTally(plans=1, proposals=1)

    monkeypatch.setattr(SkillMiningHandler, "_mine_thread", fake_mine_thread)
    result = await run_once(now=NOW)
    assert len(calls) == 1
    assert result.threads == 1


@pytest.mark.asyncio
async def test_a_failing_thread_is_recorded_and_does_not_kill_the_tick(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_mine_thread(_self, _item, *, run, position, total):
        return None

    monkeypatch.setattr(SkillMiningHandler, "_mine_thread", failing_mine_thread)
    result = await run_once(now=NOW)
    assert result.ran is True
    assert result.failures == 1
    entry = env.ledger_entry()
    assert entry is not None and entry.status == "failed" and entry.attempts == 1


@pytest.mark.asyncio
async def test_a_failing_thread_is_retried_until_poisoned(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient failure is retried each tick; a persistent one stops costing budget.

    Goes through verdict() and record() together, which is what the ledger tests alone
    cannot show: a retry must actually reach the pipeline for attempts to count up.
    """
    calls: list[str] = []

    async def failing_mine_thread(_self, item, *, run, position, total):
        calls.append(item.thread_id)
        return None

    monkeypatch.setattr(SkillMiningHandler, "_mine_thread", failing_mine_thread)
    for tick in range(3):
        await run_once(now=NOW + tick)
    entry = env.ledger_entry()
    assert entry is not None
    assert (entry.status, entry.attempts) == ("poisoned", 3)

    result = await run_once(now=NOW + 3)
    assert result.threads == 0
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_the_inbox_records_the_proposal_for_the_cli_banner(env: _Env) -> None:
    env.script(*env.demo_replies())
    await run_once(now=NOW)

    inbox = read_inbox(env.state_dir)
    assert inbox.last_tick
    assert [item["skill"] for item in inbox.recent] == [DEMO_SKILL_NAME]
    assert inbox.unseen() == inbox.recent


@pytest.mark.asyncio
async def test_a_corrupt_pool_file_costs_its_own_agent_only(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """/skill-mine ends the command on a corrupt neighbour; a timer must not lose everything.

    The file is not empty, so preflight admits the directory, but load_trajectories still
    refuses it. The tick reports the failure loudly and keeps going.
    """
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    broken = env.trajectories_dir / f"{DEMO_AGENT}_thread-broken.jsonl"
    broken.write_text(
        json.dumps({"v": 1, "event": "turn.end", "ts": "2026-09-01T10:00:00.000+00:00", "seq": 1}) + "\n",
        encoding="utf-8",
    )
    env.age(broken, QUIET + 30)

    result = await run_once(now=NOW)
    assert result.ran is True
    assert result.failures == 1
    assert result.proposals == 0


@pytest.mark.asyncio
async def test_a_missing_agent_costs_its_own_threads_only(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trajectories outlive agents; a renamed one must not abort the tick.

    Nothing is recorded either: the trajectory is fine, the configuration moved, so
    restoring the agent should make the thread mineable again.
    """

    class _GoneAgent:
        @classmethod
        async def create(cls, **_kwargs):
            raise ValueError("Agent 'SyntheticDemo' not found. Available: ['Profiler']")

    monkeypatch.setattr(runner_module, "Context", _GoneAgent)
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    result = await run_once(now=NOW)
    assert result.ran is True
    assert (result.threads, result.failures) == (0, 0)
    assert env.ledger_entry() is None


@pytest.mark.asyncio
async def test_project_filter_restricts_the_tick(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(initializer.llm_factory, "create", _boom)
    result = await run_once(now=NOW, dry_run=True, project_filter=str(env.working_dir / "elsewhere"))
    assert result.projects == 0
    assert result.threads == 0
