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

"""Tests for the proposal writer and for proposals staying out of every scanner."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from msagent.skill_evolver.bundle import BundleEpisode, EvidenceBundle, ShownFragment, build_evidence_bundle
from msagent.skill_evolver.classify import Candidate
from msagent.skill_evolver.features import FEATURES_VERSION, Episode, EvidenceItem, extract_episodes
from msagent.skill_evolver.writer import (
    PROPOSALS_DIR,
    PROVENANCE_VERSION,
    REQUIRED_PROVENANCE_KEYS,
    VERIFICATION_LEVELS,
    SecretsDetected,
    batch_dir_name,
    build_provenance,
    scan_package,
    write_proposal,
)
from msagent.skills.factory import SkillFactory
from msagent.trajectory_recorder.model import EvidenceRef
from msagent.trajectory_recorder.reader import load_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "trajectories"
FIXTURE = FIXTURES / "skill_evolver_signals.jsonl"
THREAD_ID = "thread-signals"
SOURCE = f"{THREAD_ID}.jsonl"
NAME = "build-before-test"

SKILL = "\n".join(
    [
        "---",
        f"name: {NAME}",
        "description: Use when the test suite needs generated code.",
        "---",
        "",
        "# Build before test",
        "",
        "## Inputs",
        "",
        "The repository.",
        "",
        "## Workflow",
        "",
        "1. Run make.",
        "2. Run the tests.",
        "",
        "## Outputs",
        "",
        "A green run.",
        "",
    ]
)


# ------------------------------------------------------------------ builders


def _candidate(**overrides: Any) -> Candidate:
    data: dict[str, Any] = {
        "title": "Build before test",
        "rule": "Run make before invoking the test suite.",
        "evidence_refs": ["ev1", "ev2"],
        "future_applicability": "high",
        "target": {"action": "create", "existing_skill": None},
        "candidate_id": "c1",
    }
    data.update(overrides)
    return Candidate.model_validate(data)


def _ref(seq: int, source: str = SOURCE) -> EvidenceRef:
    return EvidenceRef(source=source, line=seq, seq=seq)


def _episode(
    seqs: list[int],
    thread_id: str = THREAD_ID,
    *,
    optional: list[int] = (),
    kind: str = "error_recovery",
    roles: dict[int, str] | None = None,
) -> Episode:
    source = f"{thread_id}.jsonl"
    return Episode(
        kind=kind,
        thread_id=thread_id,
        source=source,
        evidence=[
            EvidenceItem(_ref(seq, source), (roles or {}).get(seq, "event"), seq not in optional) for seq in seqs
        ],
        tool_sequence=["bash"],
        facts={"tool": "bash"},
        weight=0.6 if kind == "error_recovery" else 0.0,
    )


def _fragment(fragment_id: str, seq: int, text: str, *, required: bool = True) -> ShownFragment:
    return ShownFragment(id=fragment_id, ref=_ref(seq), role="event", required=required, text=text)


def _bundle() -> EvidenceBundle:
    """One shown episode citing seq 4 and 5 as fragments ev1 and ev2."""
    shown = {
        "ev1": _fragment("ev1", 4, 'tool.start bash: {"cmd": "make"}'),
        "ev2": _fragment("ev2", 5, "tool.error bash (error): exit 2"),
    }
    return EvidenceBundle("(text)", shown, [BundleEpisode(_episode([4, 5]), "shown")])


POLICY = {"requested": "strict_knowledge", "selection": "strict_knowledge", "source": "config"}
CONFIG = {"requested": {"schema_version": 2}, "effective": {"policy": "strict_knowledge", "demo": False}}
VERSIONS = {"config_schema": 2, "prompts_contract": 2}
PROMPT_HASHES = {"classify": "sha256:c", "generate": "sha256:g", "review": "sha256:v"}
QUALITY_REVIEW = {
    "verdict": "pass",
    "issues": [],
    "unknown_refs": [],
    "calls": 1,
    "corrected": False,
    "initial_issues": [],
}
VERIFICATION = {"level": "evidence_supported", "note": "not executed by generator"}
PROMPT_VARIANTS = {
    "classify": "classify/prompt_v2.md",
    "generate": "generate/prompt_v2.md",
    "review": "review/prompt_v2.md",
}
CREATE_TARGET = {"action": "create", "existing_skill": None, "existing_path": None, "base_sha256": None}
UPDATE_TARGET = {"action": "update", "existing_skill": "real", "existing_path": "/x", "base_sha256": "abc"}


def _v4_kwargs(**overrides: Any) -> dict[str, Any]:
    """The provenance v4 keyword arguments every build_provenance call needs."""
    kwargs: dict[str, Any] = {
        "policy": POLICY,
        "demo": False,
        "config": CONFIG,
        "versions": VERSIONS,
        "prompt_hashes": PROMPT_HASHES,
        "quality_review": QUALITY_REVIEW,
        "verification": VERIFICATION,
    }
    kwargs.update(overrides)
    return kwargs


def _provenance(**overrides: Any) -> dict[str, Any]:
    candidate = _candidate()
    data = build_provenance(
        thread_ids=[THREAD_ID],
        bundle=_bundle(),
        candidates=[candidate],
        rejected=[],
        sources={SOURCE: f"/trajectories/{SOURCE}"},
        model="fake-model",
        prompt_variants=PROMPT_VARIANTS,
        category="default",
        target=CREATE_TARGET,
        **_v4_kwargs(),
    )
    data.update(overrides)
    return data


def _write(root: Path, **kwargs: Any) -> Path:
    args: dict[str, Any] = {"root": root, "name": NAME, "provenance": _provenance(), "thread_id": THREAD_ID}
    args.update(kwargs)
    return write_proposal(SKILL, **args)


def _seq_at(path: Path, line: int) -> int:
    """The ``seq`` written on physical ``line`` of ``path`` (read like the reader does)."""
    with path.open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            if number == line:
                return json.loads(raw)["seq"]
    raise AssertionError(f"{path} has no line {line}")


def _real_skill(skill_dir: Path, name: str) -> Path:
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: Use when testing.\n---\nbody\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- provenance


def test_build_provenance_maps_episodes_candidates_and_evidence() -> None:
    shown_episode = _episode([4, 5, 6], optional=[6])
    excluded = _episode([7, 8], thread_id="thread-other")
    shown = {
        "ev1": _fragment("ev1", 4, 'tool.start bash: {"cmd": "make"}'),
        "ev2": _fragment("ev2", 5, "tool.error bash (error): exit 2"),
    }
    bundle = EvidenceBundle(
        "(text)",
        shown,
        [BundleEpisode(shown_episode, "trimmed"), BundleEpisode(excluded, "excluded")],
    )
    kept = _candidate(
        target={"action": "update", "existing_skill": "real"},
        applies_when="the build is stale",
        constraints=["run make once"],
        expected_outcome="green tests",
    )
    rejected = _candidate(title="Fabricated", evidence_refs=["ev1", "ev9"], candidate_id="")

    provenance = build_provenance(
        thread_ids=[THREAD_ID, "thread-other", THREAD_ID],
        bundle=bundle,
        candidates=[kept],
        rejected=[(rejected, "evidence not shown in the bundle: ['ev9']")],
        sources={SOURCE: "/t/a.jsonl", "thread-other.jsonl": "/t/b.jsonl"},
        model="fake-model",
        prompt_variants={"classify": "c", "generate": "g", "review": "v"},
        category="profiler",
        target=UPDATE_TARGET,
        generated_at="2026-09-04T10:00:00+00:00",
        **_v4_kwargs(),
    )

    assert provenance["provenance_version"] == PROVENANCE_VERSION == 5
    assert provenance["features_version"] == FEATURES_VERSION == 5
    assert provenance["thread_ids"] == [THREAD_ID, "thread-other"]
    assert provenance["sources"] == {SOURCE: "/t/a.jsonl", "thread-other.jsonl": "/t/b.jsonl"}
    # Extracted episodes with their bundle outcome; unshown events have no id.
    assert provenance["episodes"] == [
        {
            "kind": "error_recovery",
            "weight": 0.6,
            "thread_id": THREAD_ID,
            "source": SOURCE,
            "bundle_status": "trimmed",
            "evidence": [
                {"id": "ev1", "source": SOURCE, "line": 4, "seq": 4, "role": "event", "required": True},
                {"id": "ev2", "source": SOURCE, "line": 5, "seq": 5, "role": "event", "required": True},
                {"id": None, "source": SOURCE, "line": 6, "seq": 6, "role": "event", "required": False},
            ],
        },
        {
            "kind": "error_recovery",
            "weight": 0.6,
            "thread_id": "thread-other",
            "source": "thread-other.jsonl",
            "bundle_status": "excluded",
            "evidence": [
                {"id": None, "source": "thread-other.jsonl", "line": 7, "seq": 7, "role": "event", "required": True},
                {"id": None, "source": "thread-other.jsonl", "line": 8, "seq": 8, "role": "event", "required": True},
            ],
        },
    ]
    # Exactly what the classify model saw, by id.
    assert provenance["evidence_shown"] == {
        "ev1": {
            "source": SOURCE,
            "line": 4,
            "seq": 4,
            "role": "event",
            "required": True,
            "text": 'tool.start bash: {"cmd": "make"}',
        },
        "ev2": {
            "source": SOURCE,
            "line": 5,
            "seq": 5,
            "role": "event",
            "required": True,
            "text": "tool.error bash (error): exit 2",
        },
    }
    (stored,) = provenance["candidates"]
    assert stored["target"] == {"action": "update", "existing_skill": "real"}
    assert stored["evidence_refs"] == ["ev1", "ev2"]
    assert stored["candidate_id"] == "c1"
    assert (stored["applies_when"], stored["constraints"], stored["expected_outcome"]) == (
        "the build is stale",
        ["run make once"],
        "green tests",
    )
    assert provenance["candidates_rejected"] == [
        {"title": "Fabricated", "reason": "evidence not shown in the bundle: ['ev9']", "evidence_refs": ["ev1", "ev9"]}
    ]
    # What the generation call was quoted, per candidate of the plan (recomputed without a budget when not given).
    assert provenance["generation_evidence"] == {"c1": ["ev1", "ev2"]}
    assert provenance["generation_evidence_omitted"] == {"c1": []}
    assert provenance["generated_at"] == "2026-09-04T10:00:00+00:00"
    assert provenance["prompt_variants"] == {"classify": "c", "generate": "g", "review": "v"}
    assert provenance["category"] == "profiler"
    assert provenance["target"] == UPDATE_TARGET
    assert REQUIRED_PROVENANCE_KEYS <= set(provenance)


def test_build_provenance_records_v4_fields() -> None:
    provenance = _provenance()

    assert provenance["provenance_version"] == PROVENANCE_VERSION == 5
    assert provenance["features_version"] == FEATURES_VERSION == 5
    assert provenance["policy"] == POLICY
    assert provenance["demo"] is False
    assert provenance["config"] == CONFIG
    assert provenance["versions"] == {"features": 5, "provenance": 5, "config_schema": 2, "prompts_contract": 2}
    assert provenance["prompt_hashes"] == PROMPT_HASHES
    assert provenance["prompt_variants"] == PROMPT_VARIANTS
    assert provenance["quality_review"] == QUALITY_REVIEW
    assert provenance["verification"] == {"level": "evidence_supported", "note": "not executed by generator"}
    assert provenance["covering_skill"] is None
    assert provenance["observed_procedure_source"] == []
    assert provenance["target"] == CREATE_TARGET
    assert REQUIRED_PROVENANCE_KEYS >= {
        "policy",
        "demo",
        "config",
        "versions",
        "prompt_hashes",
        "quality_review",
        "verification",
    }
    # A demo proposal records the covering skill it duplicates.
    covering = {"display_name": "csv-column-sum", "path": "/skills/default/csv-column-sum/SKILL.md", "sha256": "f" * 64}
    demo = _provenance_with(demo=True, policy={**POLICY, "selection": "demo_workflow"}, covering_skill=covering)
    assert demo["demo"] is True and demo["covering_skill"] == covering
    assert demo["policy"]["selection"] == "demo_workflow"


def _provenance_with(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "thread_ids": [THREAD_ID],
        "bundle": _bundle(),
        "candidates": [_candidate()],
        "rejected": [],
        "sources": {SOURCE: "/t/a.jsonl"},
        "model": "fake-model",
        "prompt_variants": PROMPT_VARIANTS,
        "category": "default",
        "target": CREATE_TARGET,
        **_v4_kwargs(),
    }
    kwargs.update(overrides)
    return build_provenance(**kwargs)


def test_build_provenance_uses_given_generation_evidence_and_omitted() -> None:
    given = _provenance_with(generation_evidence={"c1": ["ev1"]}, generation_evidence_omitted={"c1": ["ev2"]})
    assert given["generation_evidence"] == {"c1": ["ev1"]}
    assert given["generation_evidence_omitted"] == {"c1": ["ev2"]}

    recomputed = _provenance_with()
    assert recomputed["generation_evidence"] == {"c1": ["ev1", "ev2"]}
    assert recomputed["generation_evidence_omitted"] == {"c1": []}


def test_build_provenance_derives_observed_procedure_source() -> None:
    observed = _episode([2, 4, 5], kind="observed_procedure", roles={2: "task", 4: "step", 5: "result"})
    other_observed = _episode([12, 14, 15], kind="observed_procedure", roles={12: "task", 14: "step", 15: "result"})
    recovery = _episode([7, 8])
    shown = {
        "ev1": _fragment("ev1", 4, 'tool.start bash: {"cmd": "python3 -c ..."}'),
        "ev2": _fragment("ev2", 5, "tool.result bash (ok): 60.0"),
        "ev3": _fragment("ev3", 7, 'tool.start bash: {"cmd": "make"}'),
        "ev4": _fragment("ev4", 14, "tool.start bash: {}"),
    }
    bundle = EvidenceBundle(
        "(text)",
        shown,
        [BundleEpisode(observed, "shown"), BundleEpisode(recovery, "shown"), BundleEpisode(other_observed, "shown")],
    )

    provenance = _provenance_with(bundle=bundle, candidates=[_candidate(evidence_refs=["ev1", "ev2", "ev3"])])

    # Every physical line of the cited observed_procedure episode; the other kinds and uncited episodes are ignored.
    assert provenance["observed_procedure_source"] == [
        {"source": SOURCE, "line": 2, "seq": 2, "role": "task"},
        {"source": SOURCE, "line": 4, "seq": 4, "role": "step"},
        {"source": SOURCE, "line": 5, "seq": 5, "role": "result"},
    ]
    assert [row["kind"] for row in provenance["episodes"]] == [
        "observed_procedure",
        "error_recovery",
        "observed_procedure",
    ]


def test_build_provenance_redacts_secrets_in_evidence_text() -> None:
    token = "x" * 24
    shown = {
        "ev1": _fragment("ev1", 4, 'tool.start bash: {"cmd": "curl -H \'Bearer ' + token + "'\"}"),
        "ev2": _fragment("ev2", 5, "tool.error bash (error): exit 2"),
    }
    bundle = EvidenceBundle("(text)", shown, [BundleEpisode(_episode([4, 5]), "shown")])

    provenance = _provenance_with(bundle=bundle)

    text = provenance["evidence_shown"]["ev1"]["text"]
    assert "[REDACTED:bearer token]" in text and token not in text
    assert text == 'tool.start bash: {"cmd": "curl -H \'[REDACTED:bearer token]\'"}'
    assert token not in json.dumps(provenance)
    assert provenance["evidence_shown"]["ev2"]["text"] == "tool.error bash (error): exit 2"


def test_build_provenance_redacts_json_keyed_credential_in_evidence_text() -> None:
    # Bundle excerpts render tool.start arguments as JSON, so the key is quoted.
    value = "hunter2" * 2
    shown = {
        "ev1": _fragment("ev1", 4, 'tool.start http_request: {"headers": {"password": "' + value + '"}}'),
        "ev2": _fragment("ev2", 5, "tool.error bash (error): exit 2"),
    }
    bundle = EvidenceBundle("(text)", shown, [BundleEpisode(_episode([4, 5]), "shown")])

    provenance = _provenance_with(bundle=bundle)

    text = provenance["evidence_shown"]["ev1"]["text"]
    assert text == 'tool.start http_request: {"headers": {"password": "[REDACTED:credential assignment]"}}'
    assert value not in json.dumps(provenance)


def test_build_provenance_rejects_demo_update_and_missing_base_hash() -> None:
    with pytest.raises(ValueError, match="demo proposal must create a new skill, got target.action 'update'"):
        _provenance_with(demo=True, target=UPDATE_TARGET)
    with pytest.raises(ValueError, match="update provenance needs target.base_sha256"):
        _provenance_with(target={**UPDATE_TARGET, "base_sha256": None})
    with pytest.raises(ValueError, match="needs target.base_sha256"):
        _provenance_with(target={"action": "update", "existing_skill": "real", "existing_path": "/x"})
    with pytest.raises(ValueError, match="unknown verification level 'tested'"):
        _provenance_with(verification={"level": "tested", "note": ""})
    with pytest.raises(ValueError, match="demo must be a bool"):
        _provenance_with(demo=1)
    with pytest.raises(ValueError, match=r"policy is missing \['source'\]"):
        _provenance_with(policy={"requested": "strict_knowledge", "selection": "strict_knowledge"})
    with pytest.raises(ValueError, match=r"prompt_hashes is missing \['review'\]"):
        _provenance_with(prompt_hashes={"classify": "sha256:c", "generate": "sha256:g"})
    assert VERIFICATION_LEVELS == ("evidence_supported", "execution_verified")
    assert _provenance_with(verification={"level": "execution_verified", "note": "ran"})["verification"]["level"] == (
        "execution_verified"
    )


def test_build_provenance_stamps_utc_time() -> None:
    stamp = datetime.fromisoformat(_provenance()["generated_at"])

    assert stamp.tzinfo is not None
    assert stamp.utcoffset().total_seconds() == 0


def _two_plan_bundle() -> EvidenceBundle:
    """One shown episode over seq 4..7 as fragments ev1..ev4."""
    shown = {
        "ev1": _fragment("ev1", 4, 'tool.start bash: {"cmd": "make"}'),
        "ev2": _fragment("ev2", 5, "tool.error bash (error): exit 2"),
        "ev3": _fragment("ev3", 6, "tool.start pytest: only-b-sees-this"),
        "ev4": _fragment("ev4", 7, "tool.result pytest (ok): nobody-cites-this"),
    }
    return EvidenceBundle("(text)", shown, [BundleEpisode(_episode([4, 5, 6, 7]), "shown")])


def _scoped(candidates: list[Candidate]) -> dict[str, Any]:
    return build_provenance(
        thread_ids=[THREAD_ID],
        bundle=_two_plan_bundle(),
        candidates=candidates,
        rejected=[],
        sources={SOURCE: "/t/a.jsonl"},
        model="fake-model",
        prompt_variants=PROMPT_VARIANTS,
        category="default",
        target=CREATE_TARGET,
        **_v4_kwargs(),
    )


def test_build_provenance_is_scoped_to_its_plan() -> None:
    a = _candidate(candidate_id="c1", evidence_refs=["ev1", "ev2"])
    b = _candidate(title="Other", rule="Only B says so.", candidate_id="c2", evidence_refs=["ev3"])

    for_a = _scoped([a])
    for_b = _scoped([b])

    assert set(for_a["evidence_shown"]) == {"ev1", "ev2"}
    assert [c["candidate_id"] for c in for_a["candidates"]] == ["c1"]
    assert for_a["generation_evidence"] == {"c1": ["ev1", "ev2"]}
    # The episode row stays complete; events this plan does not cite have no id.
    assert [i["id"] for i in for_a["episodes"][0]["evidence"]] == ["ev1", "ev2", None, None]
    dumped = json.dumps(for_a, ensure_ascii=False)
    assert "Only B says so." not in dumped and "only-b-sees-this" not in dumped

    assert set(for_b["evidence_shown"]) == {"ev3"}
    assert [c["candidate_id"] for c in for_b["candidates"]] == ["c2"]
    assert for_b["generation_evidence"] == {"c2": ["ev3"]}
    assert [i["id"] for i in for_b["episodes"][0]["evidence"]] == [None, None, "ev3", None]
    assert '"cmd": "make"' not in json.dumps(for_b, ensure_ascii=False)


def test_build_provenance_ignores_refs_outside_the_registry() -> None:
    provenance = _scoped([_candidate(candidate_id="c1", evidence_refs=["ev1", "ev9"])])

    assert set(provenance["evidence_shown"]) == {"ev1"}
    assert provenance["generation_evidence"] == {"c1": ["ev1"]}


# -------------------------------------------------------------------- writer


def test_write_proposal_writes_skill_and_provenance(tmp_path: Path) -> None:
    root = tmp_path / "skills"

    path = _write(root)

    assert path == root / PROPOSALS_DIR / THREAD_ID / NAME / "SKILL.md"
    assert path.read_text(encoding="utf-8") == SKILL
    provenance = json.loads((path.parent / "provenance.json").read_text(encoding="utf-8"))
    assert REQUIRED_PROVENANCE_KEYS <= set(provenance)
    assert provenance["features_version"] == 5
    assert provenance["provenance_version"] == 5
    assert provenance["demo"] is False and provenance["policy"]["selection"] == "strict_knowledge"
    assert sorted(p.name for p in path.parent.iterdir()) == ["SKILL.md", "provenance.json"]
    assert not (root / "default").exists()


def test_write_proposal_adds_exactly_one_trailing_newline(tmp_path: Path) -> None:
    path = _write(tmp_path)
    stripped = write_proposal(
        SKILL.rstrip(), root=tmp_path, name="other-name", provenance=_provenance(), thread_id=THREAD_ID
    )

    assert path.read_text(encoding="utf-8") == SKILL
    assert stripped.read_text(encoding="utf-8") == SKILL


def test_write_proposal_suffixes_collisions(tmp_path: Path) -> None:
    root = tmp_path / "skills"

    paths = [_write(root) for _ in range(3)]

    assert [p.parent.name for p in paths] == [NAME, f"{NAME}-2", f"{NAME}-3"]
    assert all(p.is_file() and (p.parent / "provenance.json").is_file() for p in paths)
    assert {p.parent.parent for p in paths} == {root / PROPOSALS_DIR / THREAD_ID}


def test_write_proposal_skips_half_written_dir(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    (root / PROPOSALS_DIR / THREAD_ID / NAME).mkdir(parents=True)

    assert _write(root).parent.name == f"{NAME}-2"


def test_batch_dir_name_sanitizes_thread_id() -> None:
    assert batch_dir_name("thread-signals") == "thread-signals"
    assert batch_dir_name("thread/../evil id") == "thread-..-evil-id"
    assert len(batch_dir_name("a" * 200)) == 64


@pytest.mark.parametrize("thread_id", ["", "..", "---", "/", " "])
def test_batch_dir_name_rejects_unsafe(thread_id: str) -> None:
    with pytest.raises(ValueError, match="unsafe thread id"):
        batch_dir_name(thread_id)


@pytest.mark.parametrize("name", ["../x", "Foo", "", "ab", "a b"])
def test_write_proposal_rejects_unsafe_name(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="unsafe proposal name"):
        _write(tmp_path, name=name)
    assert not (tmp_path / PROPOSALS_DIR).exists()


def test_write_proposal_rejects_unsafe_thread_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe thread id"):
        _write(tmp_path, thread_id="..")
    assert not (tmp_path / PROPOSALS_DIR).exists()


def test_write_proposal_requires_provenance_keys(tmp_path: Path) -> None:
    provenance = _provenance()
    del provenance["episodes"]
    del provenance["generated_at"]
    del provenance["evidence_shown"]
    del provenance["provenance_version"]

    with pytest.raises(
        ValueError, match=r"missing \['episodes', 'evidence_shown', 'generated_at', 'provenance_version'\]"
    ):
        _write(tmp_path, provenance=provenance)
    assert not (tmp_path / PROPOSALS_DIR).exists()


@pytest.mark.parametrize("key", ["thread_ids", "candidates"])
def test_write_proposal_requires_threads_and_candidates(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match="thread_ids and candidates"):
        _write(tmp_path, provenance=_provenance(**{key: []}))
    assert not (tmp_path / PROPOSALS_DIR).exists()


def test_write_proposal_rejects_non_json_provenance(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        _write(tmp_path, provenance=_provenance(model=object()))
    assert not (tmp_path / PROPOSALS_DIR).exists()


def test_write_proposal_refuses_package_with_secret(tmp_path: Path) -> None:
    key = "AKIA" + "A" * 16
    leaked = SKILL.replace("A green run.", f"A green run; export the key {key} first.")
    line = leaked.split("\n").index(f"A green run; export the key {key} first.") + 1

    with pytest.raises(SecretsDetected) as info:
        write_proposal(leaked, root=tmp_path, name=NAME, provenance=_provenance(), thread_id=THREAD_ID)

    message = str(info.value)
    assert message.startswith("proposal package contains possible secrets; nothing written: ")
    assert f"SKILL.md:{line}: possible secret (aws access key)" in message
    assert key not in message
    assert not (tmp_path / PROPOSALS_DIR).exists()

    # A secret only in the provenance (a candidate rule the model echoed) is refused the same way.
    token = "ghp_" + "b" * 36
    tainted = _provenance(candidates=[_candidate(rule=f"Authenticate with {token} before pushing.").model_dump()])
    with pytest.raises(SecretsDetected) as info:
        _write(tmp_path, provenance=tainted)
    assert "provenance.json:" in str(info.value) and "possible secret (github token)" in str(info.value)
    assert token not in str(info.value)
    assert not (tmp_path / PROPOSALS_DIR).exists()


def test_write_proposal_refuses_quoted_credential_in_candidate_rule(tmp_path: Path) -> None:
    # json.dumps escapes the quotes around the value; the raw text fields are scanned as well.
    value = "hunter2" * 2
    tainted = _provenance(candidates=[_candidate(rule=f'Set password: "{value}" in the config').model_dump()])

    with pytest.raises(SecretsDetected) as info:
        _write(tmp_path, provenance=tainted)

    message = str(info.value)
    assert "provenance.json: possible secret (credential assignment) in a text field" in message
    assert value not in message
    assert not (tmp_path / PROPOSALS_DIR).exists()
    clean = _provenance()
    assert scan_package({"provenance.json": json.dumps(clean)}, raw=clean) == []


def test_scan_package_names_file_line_and_label() -> None:
    bearer = "Bearer " + "y" * 24
    findings = scan_package({"SKILL.md": f"ok\n{bearer}\n", "provenance.json": '{"rule": "password: hunter2secret"}'})

    assert findings == [
        "SKILL.md:2: possible secret (bearer token)",
        "provenance.json:1: possible secret (credential assignment)",
    ]
    assert scan_package({"SKILL.md": SKILL, "provenance.json": json.dumps(_provenance())}) == []


def test_write_proposal_refuses_demo_non_create(tmp_path: Path) -> None:
    provenance = _provenance(demo=True, target=UPDATE_TARGET)

    with pytest.raises(ValueError, match=r"demo proposal must target a new skill \(create\)"):
        _write(tmp_path, provenance=provenance)
    assert not (tmp_path / PROPOSALS_DIR).exists()

    demo = _provenance(demo=True, policy={**POLICY, "selection": "demo_workflow"})
    path = write_proposal(
        SKILL.replace(f"name: {NAME}", "name: demo-build-before-test"),
        root=tmp_path,
        name="demo-build-before-test",
        provenance=demo,
        thread_id=THREAD_ID,
    )
    stored = json.loads((path.parent / "provenance.json").read_text(encoding="utf-8"))
    assert stored["demo"] is True and stored["policy"]["selection"] == "demo_workflow"


# ---------------------------------------------------------- scanner guarantee


@pytest.mark.asyncio
async def test_proposals_not_scanned_by_skill_factory(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _real_skill(root / "cat" / "real", "real")
    _real_skill(root / "flat", "flat")
    _real_skill(root / ".hidden" / "deep" / "x" / "hidden-skill", "hidden-skill")
    _write(root)

    factory = SkillFactory()
    for skills_dir in (root, [root]):
        loaded = await factory.load_skills(skills_dir)

        names = sorted(skill.name for category in loaded.values() for skill in category.values())
        assert names == ["flat", "real"]
        assert factory.get_module_map() == {"cat:real": "cat", "default:flat": "default"}


@pytest.mark.asyncio
async def test_dot_base_dir_still_loads(tmp_path: Path) -> None:
    root = tmp_path / ".msagent" / "skills"
    _real_skill(root / "cat" / "s", "s")

    loaded = await SkillFactory().load_skills(root)

    assert "s" in loaded["cat"]


def test_proposals_invisible_to_agent_factory_sources(tmp_path: Path) -> None:
    # Read-only use of the middleware side: the skill sources deepagents scans.
    from msagent.agents.factory import AgentFactory

    root = tmp_path / "skills"
    _real_skill(root / "cat" / "real", "real")
    _write(root)

    sources = AgentFactory._resolve_existing_paths([root])

    assert sources == [str(root / "cat")]


# ------------------------------------------------------------- acceptance


@pytest.mark.parametrize("fixture", ["skill_evolver_signals.jsonl", "malformed_lines.jsonl"])
def test_provenance_shown_fragments_resolve_to_source_lines(tmp_path: Path, fixture: str) -> None:
    # Real detectors, real bundle; the written provenance must let a reader
    # go from every candidate to the fragments the model saw and from every
    # fragment to the physical line that holds that very event — even with
    # corrupted lines in between (malformed_lines.jsonl).
    source = FIXTURES / fixture
    trajectory = load_trajectory(source)
    episodes = extract_episodes(trajectory)
    assert episodes
    bundle = build_evidence_bundle(episodes, [trajectory])
    cited = sorted(bundle.shown)[:2]
    kept = _candidate(evidence_refs=cited, applies_when="the tool fails on the first attempt")
    rejected = _candidate(title="Fabricated", evidence_refs=["ev999"], candidate_id="")
    provenance = build_provenance(
        thread_ids=[trajectory.thread_id],
        bundle=bundle,
        candidates=[kept],
        rejected=[(rejected, "evidence not shown in the bundle: ['ev999']")],
        sources={trajectory.source: str(trajectory.path)},
        model="fake-model",
        prompt_variants=PROMPT_VARIANTS,
        category="default",
        target=CREATE_TARGET,
        **_v4_kwargs(),
    )

    path = write_proposal(
        SKILL, root=tmp_path / "skills", name=NAME, provenance=provenance, thread_id=trajectory.thread_id
    )

    stored = json.loads((path.parent / "provenance.json").read_text(encoding="utf-8"))
    assert stored["provenance_version"] == 5
    assert stored["observed_procedure_source"] == []
    assert stored["thread_ids"] == [trajectory.thread_id]
    assert stored["sources"] == {source.name: str(source)}
    shown = stored["evidence_shown"]
    # Scoped to what the plan's candidate cites, not the whole registry.
    assert set(shown) == set(cited) <= set(bundle.shown)
    for fragment_id, entry in shown.items():
        assert entry["source"] == source.name
        assert _seq_at(source, entry["line"]) == entry["seq"]
        assert f"- [{fragment_id}] {entry['text']}" in bundle.text
    assert len(stored["episodes"]) == len(episodes)
    for episode in stored["episodes"]:
        assert episode["thread_id"] == trajectory.thread_id
        assert episode["bundle_status"] == "shown"
        for item in episode["evidence"]:
            if item["id"] is None:
                continue
            assert item["id"] in shown
            assert (shown[item["id"]]["source"], shown[item["id"]]["line"]) == (item["source"], item["line"])
    resolved = {item["id"] for episode in stored["episodes"] for item in episode["evidence"]} - {None}
    assert resolved == set(cited)
    (candidate,) = stored["candidates"]
    assert candidate["candidate_id"] == "c1"
    assert candidate["evidence_refs"] == cited
    assert set(candidate["evidence_refs"]) <= set(shown)
    assert candidate["applies_when"] == "the tool fails on the first attempt"
    assert stored["candidates_rejected"] == [
        {"title": "Fabricated", "reason": "evidence not shown in the bundle: ['ev999']", "evidence_refs": ["ev999"]}
    ]
    assert set(stored["generation_evidence"]) == {"c1"}
    assert set(stored["generation_evidence"]["c1"]) <= set(cited)
