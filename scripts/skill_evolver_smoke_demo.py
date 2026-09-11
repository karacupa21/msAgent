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

"""Real-LLM smoke test of the Skill Evolver demo path (ARCHITECTURE_skill_evolve.md, section 21).

Runs the real ``/skill-mine --thread thread-demo-synthetic --demo --policy reusable_workflow``
handler (no fakes) against the synthetic fixture inside an isolated working directory, with
the LLM configuration the CLI itself would use (``AppPaths.resolve()``, storage layout seeding,
``initializer.load_llm_config``). Nothing is written into the repository: the proposal lands in
``<work>/skills/.proposals/`` and the decision report in the project state of ``<work>``.

Usage (from the repository root, with the CLI's LLM credentials in the environment)::

    .venv/bin/python scripts/skill_evolver_smoke_demo.py [--dry-run] [--model ALIAS] [--work DIR]

``--dry-run`` exercises the same path without creating an LLM.

The demo needs the workspace trajectory scope (the default). The fixture records
``working_dir: /synthetic/demo`` (the ``is_synthetic`` marker), so under
``output.scope: shared`` /skill-mine only sees the threads recorded in ``--work``
and reports the demo thread as missing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "trajectories" / "skill_evolver_demo_success.jsonl"
# Outside the repository, so the run leaves no untracked files behind.
DEFAULT_WORK = Path(tempfile.gettempdir()) / "msagent-skill-evolver-smoke"


async def run(work: Path, model: str, dry_run: bool) -> int:
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO))
    from msagent.core.paths import AppPaths
    from msagent.core.storage_layout import validate_and_initialize_storage_layout

    app_paths = AppPaths.resolve()
    validate_and_initialize_storage_layout(app_paths)

    import msagent.cli.handlers  # noqa: F401  (import-cycle workaround, as in the tests)
    from msagent.cli.bootstrap.initializer import initializer
    from msagent.skill_evolver.mining import SkillMiningHandler
    from msagent.skill_evolver.report import decisions_dir
    from msagent.trajectory_recorder.export import resolve_trajectories_dir

    work.mkdir(parents=True, exist_ok=True)
    (work / "skills").mkdir(exist_ok=True)
    state_dir = initializer.get_project_paths(work).root
    trajectories_dir = resolve_trajectories_dir(state_dir=state_dir)
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    target = trajectories_dir / "SyntheticDemo_thread-demo-synthetic.jsonl"
    if not target.exists():
        shutil.copy(FIXTURE, target)
    print(f"[smoke] home={app_paths.home}")
    print(f"[smoke] working_dir={work}")
    print(f"[smoke] trajectory={target}")

    context = SimpleNamespace(agent="Profiler", thread_id="smoke", working_dir=work, model=model)
    session = SimpleNamespace(context=context, graph=None)
    args = ["--thread", "thread-demo-synthetic", "--demo", "--policy", "reusable_workflow"]
    if dry_run:
        args.append("--dry-run")
    print(f"[smoke] /skill-mine {' '.join(args)}")
    await SkillMiningHandler(session).handle(args)

    proposals = sorted((work / "skills" / ".proposals").glob("*/*/provenance.json"), key=lambda p: p.stat().st_mtime)
    if proposals:
        prov = json.loads(proposals[-1].read_text(encoding="utf-8"))
        print(f"[smoke] proposal: {proposals[-1].parent / 'SKILL.md'}")
        summary = {
            "model": prov.get("model"),
            "demo": prov.get("demo"),
            "policy": prov.get("policy"),
            "prompt_hashes": prov.get("prompt_hashes"),
            "config.effective": prov.get("config", {}).get("effective"),
            "quality_review": prov.get("quality_review"),
            "verification": prov.get("verification"),
            "target": prov.get("target"),
            "versions": prov.get("versions"),
        }
        print("[smoke] provenance summary:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("[smoke] SKILL.md:")
        print((proposals[-1].parent / "SKILL.md").read_text(encoding="utf-8"))
    else:
        print("[smoke] no proposal written")
    reports = sorted(decisions_dir(state_dir).glob("*.json"), key=lambda p: p.stat().st_mtime)
    print(f"[smoke] decision report: {reports[-1] if reports else 'none'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dry-run", action="store_true", help="run the same path without creating an LLM")
    parser.add_argument("--model", default="default", help="LLM alias of the session (config.llms.yml)")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK, help="isolated working directory")
    ns = parser.parse_args()
    return asyncio.run(run(ns.work.resolve(), ns.model, ns.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
