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

"""Tests for skill_evolver.config: schema v2, v1 migration, overrides, effective rules."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import msagent.cli.handlers  # noqa: F401
from msagent.skill_evolver import config as module
from msagent.skill_evolver.config import (
    CROSS_SESSION_LIMIT,
    STAGES,
    DirectSkillGenerationConfig,
    SkillEvolverConfig,
    SkillEvolverConfigError,
    apply_overrides,
    effective_rules,
    load_skill_evolver_config,
    output_root,
    resolve_output_dir,
)
from msagent.skill_evolver.features import DEFAULT_MIN_EVIDENCE_SCORE

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGED = REPO_ROOT / "resources" / "configs" / "default" / "config.skill.evolver.yml"
FILE_NAME = "config.skill.evolver.yml"

V1_TEXT = (
    "active: default\nprompt_file: prompt_v1.md\nmin_evidence_score: 2.5\n"
    "max_plans: 5\ncategory: prof\noutput_dir: out\n"
)


def _write(config_dir: Path, text: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / FILE_NAME
    path.write_text(text, encoding="utf-8")
    return path


def _problems(config_dir: Path) -> list[tuple[str, str]]:
    with pytest.raises(SkillEvolverConfigError) as info:
        load_skill_evolver_config(config_dir)
    return info.value.problems


# ------------------------------------------------------------------ defaults


def test_packaged_default_is_schema_v2_with_spec_defaults(tmp_path: Path) -> None:
    cfg = load_skill_evolver_config(tmp_path / "no-user-config")

    assert cfg.source == "packaged"
    assert cfg.schema_version == 2
    assert cfg.demo_mode is False
    assert cfg.policy == "strict_knowledge"
    assert cfg.min_evidence_score == 1.0 == DEFAULT_MIN_EVIDENCE_SCORE == module.DEFAULT_MIN_EVIDENCE_SCORE
    assert cfg.excerpt_max_chars == 1000
    assert cfg.bundle_max_chars == 60000
    assert cfg.surrounding_events == 2
    assert cfg.cross_session_limit == 20 == CROSS_SESSION_LIMIT
    assert cfg.on_nothing == "expand_context_once"
    assert cfg.max_plans == 3
    assert cfg.max_llm_calls == 16
    assert cfg.category == "default"
    assert cfg.output_dir is None
    assert cfg.contract_version == 2
    assert all(cfg.prompt_for(stage) == "prompt_v2.md" for stage in STAGES)
    assert STAGES == ("classify", "generate", "review")
    assert cfg.save_decision_report is True
    assert cfg.save_evidence_text is False
    assert cfg.save_rejected_drafts is True
    assert cfg.legacy_format is False
    assert cfg.prompt_file is None
    assert cfg.notes == () and cfg.overrides == ()


def test_packaged_file_equals_dataclass_defaults_and_has_no_version_key() -> None:
    data = yaml.safe_load(PACKAGED.read_text(encoding="utf-8"))
    packaged = load_skill_evolver_config(PACKAGED.parent / "not-a-config-dir")
    defaults = DirectSkillGenerationConfig(source="packaged")

    assert data["schema_version"] == 2
    assert "version" not in data  # tests/ut/configs/test_config_versions.py must keep ignoring this file
    assert SkillEvolverConfig.model_validate(data).evidence.cross_session_limit == 20
    assert packaged == defaults
    assert SkillEvolverConfig(schema_version=2).evidence.cross_session_limit == 20


def test_user_file_wins_and_env_home_is_the_default_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("MSAGENT_HOME", str(home))
    path = _write(home / "config", "schema_version: 2\ngeneration:\n  max_plans_per_thread: 7\n")

    cfg = load_skill_evolver_config()

    assert cfg.max_plans == 7
    assert cfg.source == str(path)


def test_missing_packaged_file_yields_dataclass_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "packaged_config_file", lambda: tmp_path / "absent.yml")

    cfg = load_skill_evolver_config(tmp_path / "empty")

    assert cfg == DirectSkillGenerationConfig()
    assert cfg.source == "defaults"


# --------------------------------------------------------------- v1 migration


def test_v1_flat_file_is_migrated_in_memory(tmp_path: Path) -> None:
    path = _write(tmp_path / "cfg", V1_TEXT)
    before = path.read_bytes()

    cfg = load_skill_evolver_config(tmp_path / "cfg")

    assert cfg.min_evidence_score == 2.5
    assert cfg.max_plans == 5
    assert cfg.category == "prof"
    assert cfg.output_dir == Path("out")
    assert cfg.policy == "strict_knowledge"
    assert cfg.demo_mode is False
    assert cfg.legacy_format is True
    assert cfg.prompt_file == "prompt_v1.md"
    assert cfg.active == "default"
    assert cfg.source == str(path)
    assert all(cfg.prompt_for(stage) == "prompt_v1.md" for stage in STAGES)
    assert any(note.startswith("active:") for note in cfg.notes)
    assert any(note.startswith("prompt_file:") for note in cfg.notes)
    assert path.read_bytes() == before


def test_v1_file_without_prompt_file_uses_stage_names(tmp_path: Path) -> None:
    _write(tmp_path / "cfg", "active: default\nmin_evidence_score: 1\n")

    cfg = load_skill_evolver_config(tmp_path / "cfg")

    assert cfg.legacy_format is True
    assert cfg.prompt_file is None
    assert cfg.min_evidence_score == 1.0
    assert cfg.prompt_for("review") == "prompt_v2.md"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("min_evidence_score: abc\n", ("min_evidence_score", "must be a finite non-negative number, got 'abc'")),
        ("min_evidence_score: -1\n", ("min_evidence_score", "must be a finite non-negative number, got -1")),
        ("min_evidence_score: true\n", ("min_evidence_score", "must be a finite non-negative number, got True")),
        ("max_plans: 0\n", ("max_plans", "must be a positive whole number, got 0")),
        ("max_plans: abc\n", ("max_plans", "must be a whole number, got 'abc'")),
        ("max_plans: -2\n", ("max_plans", "must be a positive whole number, got -2")),
        ("category: a/b\n", ("category", "must be one directory name without separators, got 'a/b'")),
        (
            "prompt_file: ../../etc/passwd\n",
            ("prompt_file", "must be a file name without path separators or '..', got '../../etc/passwd'"),
        ),
        ("active: ../x\n", ("active", "must be one directory name without separators, got '../x'")),
        ("prompt_file: 5\n", ("prompt_file", "must be text, got 5")),
        ("unknown_thing: 1\n", ("unknown_thing", "unknown key")),
        ("schema_versoin: 2\n", ("schema_versoin", "unknown key")),
    ],
)
def test_v1_invalid_values_stop_with_v1_key_path(tmp_path: Path, text: str, expected: tuple[str, str]) -> None:
    _write(tmp_path / "cfg", text)

    problems = _problems(tmp_path / "cfg")

    assert problems == [expected]


def test_v1_collects_every_problem_at_once(tmp_path: Path) -> None:
    _write(tmp_path / "cfg", "bogus: 1\nmax_plans: 0\nmin_evidence_score: nope\n")

    problems = _problems(tmp_path / "cfg")

    assert [path for path, _ in problems] == ["bogus", "max_plans", "min_evidence_score"]


# ----------------------------------------------------------------- v2 errors


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("schema_version: 2\nfoo: 1\n", ("foo", "unknown key")),
        ("schema_version: 2\ngeneration:\n  max_plan: 2\n", ("generation.max_plan", "unknown key")),
        ("schema_version: 2\ndemo_mode: 'false'\n", ("demo_mode", "must be a boolean (true/false), got 'false'")),
        ("schema_version: 2\ndemo_mode: 1\n", ("demo_mode", "must be a boolean (true/false), got 1")),
        (
            "schema_version: 2\ngate:\n  min_evidence_score: true\n",
            ("gate.min_evidence_score", "must be a finite non-negative number, got True"),
        ),
        (
            "schema_version: 2\ngate:\n  min_evidence_score: abc\n",
            ("gate.min_evidence_score", "must be a finite non-negative number, got 'abc'"),
        ),
        (
            "schema_version: 2\ngate:\n  min_evidence_score: .inf\n",
            ("gate.min_evidence_score", "must be a finite non-negative number, got inf"),
        ),
        ("schema_version: 2\ngate: 5\n", ("gate", "must be a mapping")),
        (
            "schema_version: 2\nevidence:\n  surrounding_events: -1\n",
            ("evidence.surrounding_events", "must be a whole number >= 0, got -1"),
        ),
        (
            "schema_version: 2\nevidence:\n  excerpt_max_chars: true\n",
            ("evidence.excerpt_max_chars", "must be a whole number, got True"),
        ),
        (
            "schema_version: 2\nevidence:\n  bundle_max_chars: 2.5\n",
            ("evidence.bundle_max_chars", "must be a whole number, got 2.5"),
        ),
        (
            "schema_version: 2\ngeneration:\n  max_llm_calls_per_thread: 0\n",
            ("generation.max_llm_calls_per_thread", "must be a positive whole number, got 0"),
        ),
        (
            "schema_version: 2\ngeneration:\n  max_plans_per_thread: '3'\n",
            ("generation.max_plans_per_thread", "must be a whole number, got '3'"),
        ),
        (
            "schema_version: 2\nclassification:\n  policy: lenient\n",
            ("classification.policy", "must be one of strict_knowledge, reusable_workflow, got 'lenient'"),
        ),
        (
            "schema_version: 2\non_nothing:\n  action: retry\n",
            ("on_nothing.action", "must be one of stop, expand_context_once, got 'retry'"),
        ),
        ("schema_version: '2'\n", ("schema_version", "must be a whole number, got '2'")),
        ("schema_version: true\n", ("schema_version", "must be a whole number, got True")),
        ("schema_version: 3\n", ("schema_version", "unsupported schema_version 3; this build supports 2")),
        (
            "schema_version: 2\nmin_evidence_score: 1.0\n",
            ("min_evidence_score", "legacy v1 key in a schema_version 2 file; use gate.min_evidence_score"),
        ),
        (
            "schema_version: 2\nmax_plans: 3\n",
            ("max_plans", "legacy v1 key in a schema_version 2 file; use generation.max_plans_per_thread"),
        ),
        (
            "schema_version: 2\nprompt_file: prompt_v1.md\n",
            (
                "prompt_file",
                "legacy v1 key in a schema_version 2 file; remove it (prompts are configured under prompts:)",
            ),
        ),
        (
            "schema_version: 2\nprompts:\n  classify: ../x.md\n",
            ("prompts.classify", "must be a file name without path separators or '..', got '../x.md'"),
        ),
        (
            "schema_version: 2\nprompts:\n  generate: /abs/x.md\n",
            ("prompts.generate", "must be a file name without path separators or '..', got '/abs/x.md'"),
        ),
        ("schema_version: 2\nprompts:\n  review: 3\n", ("prompts.review", "must be text, got 3")),
        (
            "schema_version: 2\ngeneration:\n  category: a/b\n",
            ("generation.category", "must be one directory name without separators, got 'a/b'"),
        ),
        (
            "schema_version: 2\ngeneration:\n  output_dir: ''\n",
            ("generation.output_dir", "must be a non-empty path, got ''"),
        ),
        (
            "schema_version: 2\nprompts:\n  contract_version: 1\n",
            ("prompts.contract_version", "must equal 2 (prompt contract of this build)"),
        ),
        (
            "schema_version: 2\ndiagnostics:\n  save_evidence_text: yes please\n",
            ("diagnostics.save_evidence_text", "must be a boolean (true/false), got 'yes please'"),
        ),
    ],
)
def test_v2_rejects_unknown_keys_bad_types_and_alias_conflicts(
    tmp_path: Path, text: str, expected: tuple[str, str]
) -> None:
    _write(tmp_path / "cfg", text)

    problems = _problems(tmp_path / "cfg")

    assert problems == [expected]


def test_v2_legacy_keys_are_all_reported_before_validation(tmp_path: Path) -> None:
    _write(tmp_path / "cfg", "schema_version: 2\nactive: default\noutput_dir: x\ncategory: c\n")

    problems = _problems(tmp_path / "cfg")

    assert [path for path, _ in problems] == ["active", "category", "output_dir"]
    assert problems[0][1].startswith("legacy v1 key in a schema_version 2 file; remove it")


def test_v2_accepts_yaml_int_for_score_zero_surrounding_and_output_dir(tmp_path: Path) -> None:
    _write(
        tmp_path / "cfg",
        "schema_version: 2\ndemo_mode: true\ngate:\n  min_evidence_score: 1\nevidence:\n  surrounding_events: 0\n"
        "generation:\n  output_dir: ~/skills\nclassification:\n  policy: reusable_workflow\n",
    )

    cfg = load_skill_evolver_config(tmp_path / "cfg")

    assert cfg.min_evidence_score == 1.0 and isinstance(cfg.min_evidence_score, float)
    assert cfg.surrounding_events == 0
    assert cfg.output_dir == Path("~/skills")
    assert cfg.demo_mode is True
    assert cfg.policy == "reusable_workflow"
    assert cfg.legacy_format is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("schema_version: [\n", "invalid YAML"),
        ("- a\n- b\n", "must be a mapping"),
        ("just text\n", "must be a mapping"),
    ],
)
def test_invalid_yaml_and_non_mapping_are_config_errors(tmp_path: Path, text: str, expected: str) -> None:
    _write(tmp_path / "cfg", text)

    problems = _problems(tmp_path / "cfg")

    assert len(problems) == 1
    assert problems[0][0] == "<file>"
    assert problems[0][1].startswith(expected)


def test_empty_file_is_a_v1_file_with_defaults(tmp_path: Path) -> None:
    _write(tmp_path / "cfg", "# nothing\n")

    cfg = load_skill_evolver_config(tmp_path / "cfg")

    assert cfg.legacy_format is True
    assert cfg.max_plans == 3


def test_error_lines_name_the_file_and_sort_by_path(tmp_path: Path) -> None:
    exc = SkillEvolverConfigError(
        [("gate.min_evidence_score", "must be a finite non-negative number, got 'abc'"), ("foo", "unknown key")],
        file=Path("/x/config.skill.evolver.yml"),
    )

    assert exc.lines() == [
        "config.skill.evolver.yml: foo: unknown key",
        "config.skill.evolver.yml: gate.min_evidence_score: must be a finite non-negative number, got 'abc'",
    ]
    assert str(exc) == "; ".join(exc.lines())
    assert SkillEvolverConfigError([("a", "b")], file=None).lines() == ["config: a: b"]


# ---------------------------------------------------- overrides and effective


def test_overrides_and_effective_rules_precedence(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "cfg", "schema_version: 2\ndemo_mode: true\nclassification:\n  policy: reusable_workflow\n"
    )
    before = path.read_bytes()
    loaded = load_skill_evolver_config(tmp_path / "cfg")

    cfg = apply_overrides(loaded, policy="strict_knowledge", demo=False)
    rules = effective_rules(cfg, working_dir=tmp_path)

    assert (loaded.policy, loaded.demo_mode) == ("reusable_workflow", True)
    assert (cfg.policy, cfg.demo_mode) == ("strict_knowledge", False)
    assert cfg.overrides == ("--policy strict_knowledge", "--no-demo")
    assert (rules.policy, rules.demo, rules.selection) == ("strict_knowledge", False, "strict_knowledge")

    demo = effective_rules(apply_overrides(loaded, demo=True), working_dir=tmp_path)
    assert (demo.policy, demo.demo, demo.selection) == ("reusable_workflow", True, "demo_workflow")
    assert apply_overrides(loaded, demo=True).overrides == ("--demo",)
    assert apply_overrides(loaded) == loaded

    with pytest.raises(ValueError, match="strict_knowledge, reusable_workflow, got 'x'"):
        apply_overrides(loaded, policy="x")
    assert path.read_bytes() == before
    assert load_skill_evolver_config(tmp_path / "cfg") == loaded


def test_demo_keeps_limits_unchanged(tmp_path: Path) -> None:
    cfg = DirectSkillGenerationConfig(
        max_plans=2,
        max_llm_calls=9,
        bundle_max_chars=1234,
        excerpt_max_chars=77,
        surrounding_events=5,
        cross_session_limit=4,
        min_evidence_score=3.5,
        on_nothing="stop",
        category="cat",
    )

    plain = effective_rules(cfg, working_dir=tmp_path).as_record()
    demo = effective_rules(apply_overrides(cfg, demo=True), working_dir=tmp_path).as_record()

    assert (plain["demo"], plain["selection"]) == (False, "strict_knowledge")
    assert (demo["demo"], demo["selection"]) == (True, "demo_workflow")
    assert {k: v for k, v in demo.items() if k not in ("demo", "selection")} == {
        k: v for k, v in plain.items() if k not in ("demo", "selection")
    }
    assert (plain["max_plans"], plain["max_llm_calls"], plain["bundle_max_chars"], plain["excerpt_max_chars"]) == (
        2,
        9,
        1234,
        77,
    )
    assert (plain["surrounding_events"], plain["cross_session_limit"], plain["min_evidence_score"]) == (5, 4, 3.5)
    assert plain["output_root"] == str(tmp_path / "skills")


def test_effective_rules_record_is_flat_and_json_able(tmp_path: Path) -> None:
    rules = effective_rules(DirectSkillGenerationConfig(output_dir=Path("rel")), working_dir=tmp_path)

    record = rules.as_record()

    assert record["output_root"] == str(tmp_path / "rel")
    assert record["selection"] == "strict_knowledge"
    assert set(record) == {
        "policy",
        "demo",
        "selection",
        "min_evidence_score",
        "max_plans",
        "max_llm_calls",
        "on_nothing",
        "excerpt_max_chars",
        "bundle_max_chars",
        "surrounding_events",
        "cross_session_limit",
        "category",
        "output_root",
        "save_decision_report",
        "save_evidence_text",
        "save_rejected_drafts",
    }


def test_requested_record_has_v2_shape_and_provenance(tmp_path: Path) -> None:
    _write(tmp_path / "cfg", V1_TEXT)
    cfg = apply_overrides(load_skill_evolver_config(tmp_path / "cfg"), demo=True)

    record = cfg.as_record()

    assert record["schema_version"] == 2
    assert record["gate"] == {"min_evidence_score": 2.5}
    assert record["generation"] == {
        "max_plans_per_thread": 5,
        "max_llm_calls_per_thread": 16,
        "category": "prof",
        "output_dir": "out",
    }
    assert record["prompts"] == {
        "contract_version": 2,
        "classify": "prompt_v1.md",
        "generate": "prompt_v1.md",
        "review": "prompt_v1.md",
    }
    assert record["demo_mode"] is True
    assert record["legacy_format"] is True
    assert record["overrides"] == ["--demo"]
    assert len(record["notes"]) == 2
    assert record["source"] == str(tmp_path / "cfg" / FILE_NAME)
    assert DirectSkillGenerationConfig().as_record()["generation"]["output_dir"] is None


def test_prompt_for_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown prompt stage 'bogus'"):
        DirectSkillGenerationConfig().prompt_for("bogus")


def test_output_dir_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    work = tmp_path / "work"

    assert resolve_output_dir(DirectSkillGenerationConfig(), work) is None
    assert output_root(DirectSkillGenerationConfig(), work) == work / "skills"
    assert output_root(DirectSkillGenerationConfig(output_dir=Path("rel/skills")), work) == work / "rel" / "skills"
    assert output_root(DirectSkillGenerationConfig(output_dir=Path("~/x")), work) == tmp_path / "home" / "x"
    assert output_root(DirectSkillGenerationConfig(output_dir=Path("/abs/skills")), work) == Path("/abs/skills")
    assert resolve_output_dir(DirectSkillGenerationConfig(output_dir=Path("rel")), work) == work / "rel"


def test_exgraph_output_dir_reader_uses_the_loader_and_fails_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from msagent.exgraph.skills import _evolver_output_dir

    home = tmp_path / "home"
    monkeypatch.setenv("MSAGENT_HOME", str(home))
    work = tmp_path / "work"
    _write(home / "config", "schema_version: 2\ngeneration:\n  output_dir: out\n")

    assert _evolver_output_dir(work) == work / "out"

    _write(home / "config", "schema_version: 2\ngeneration:\n  output_dir: 5\n")

    assert _evolver_output_dir(work) is None


def test_config_module_loads_without_langchain_or_pipeline_modules() -> None:
    code = (
        "import sys; import msagent.skill_evolver.config; "
        "bad = sorted(m for m in sys.modules if m.startswith('langchain') or m in ("
        "'msagent.skill_evolver.direct_skill_generation', 'msagent.skill_evolver.mining', "
        "'msagent.skill_evolver.features', 'msagent.cli.bootstrap.initializer')); "
        "print(bad)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT)

    assert result.stdout.strip() == "[]"
