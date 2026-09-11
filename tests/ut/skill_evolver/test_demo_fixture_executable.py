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

"""The synthetic demo fixture is honest: its commands, and the example the pipeline writes, really compute 60.

The pipeline itself never executes anything from a trajectory. This module
does, under a narrow allowlist (``python3 -c <code> <operands>``: every name
token of the code is one of ALLOWED_NAMES, ``csv.DictReader`` and
``sys.argv`` are the only attribute accesses, strings carry no backslash and
operands are plain relative names) in an isolated interpreter with an empty
PATH, so a fixture or a generated SKILL.md can never smuggle a shell command
into the test run.
"""

from __future__ import annotations

import io
import json
import keyword
import re
import shlex
import subprocess
import sys
import tokenize
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import msagent.cli.handlers  # noqa: F401
from msagent.exgraph.config import ENV_DISABLED
from msagent.exgraph.config import reset_config_cache as reset_exgraph_cache
from msagent.skill_evolver.bundle import build_evidence_bundle
from msagent.skill_evolver.config import DirectSkillGenerationConfig, effective_rules
from msagent.skill_evolver.features import DetectorNote, extract_episodes
from msagent.skill_evolver.pipeline import LazyLlm, RunContext, ThreadInput, run_thread
from msagent.skill_evolver.prompts import PromptText, StagePrompts, prompt_sha256
from msagent.skill_evolver.report import is_synthetic
from msagent.trajectory_recorder.reader import load_trajectory

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "trajectories" / "skill_evolver_demo_success.jsonl"
THREAD_ID = "thread-demo-synthetic"
SALES_CSV = "id,amount\n1,10\n2,20\n3,30\n"
# Placeholders of the generated example and the values the fixture recorded for them.
PARAMETERS = {"<csv-path>": "input/sales.csv", "<column>": "amount", "<expected-total>": "60"}

# The allowlist: python3 -c <code> <operands>. The code is tokenized; every
# NAME token must be listed here (keywords included), attribute access is
# limited to ALLOWED_ATTRIBUTES, strings carry no backslash and an f-string
# field is an allowed name; operands are plain relative names.
INTERPRETERS = ("python", "python3")
ALLOWED_NAMES = frozenset(
    {"csv", "sys", "print", "sum", "float", "DictReader", "open", "argv"}
    | {"for", "in", "if", "else", "import"}
    | {"r", "row", "total"}
)
ALLOWED_KEYWORDS = frozenset(name for name in ALLOWED_NAMES if keyword.iskeyword(name))
ALLOWED_ATTRIBUTES = frozenset({"csv.DictReader", "sys.argv"})
# ``open`` reads a positional operand and nothing else: the tokens that must follow it.
OPEN_CALL = (
    ("OP", "("),
    ("NAME", "sys"),
    ("OP", "."),
    ("NAME", "argv"),
    ("OP", "["),
    ("NUMBER", None),
    ("OP", "]"),
    ("OP", ")"),
)
# Python >= 3.12 tokenizes f-strings into FSTRING_START/MIDDLE/END plus ordinary
# tokens for the fields; older versions yield one STRING token (see _string_problem).
_FSTRING_TYPES = frozenset(
    getattr(tokenize, name) for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END") if hasattr(tokenize, name)
)
ALLOWED_TOKEN_TYPES = (
    frozenset({tokenize.NAME, tokenize.NUMBER, tokenize.STRING, tokenize.OP, tokenize.NEWLINE, tokenize.NL})
    | {tokenize.ENDMARKER}
    | _FSTRING_TYPES
)
_FSTRING_FIELD_RE = re.compile(r"\{([^{}]*)\}")
OPERAND_RE = re.compile(r"[A-Za-z0-9_./-]+")

CLASSIFY_TEMPLATE = "Library:\n{skill_library}\n\nPolicy:\n{selection_policy}\n\nBundle:\n{evidence_bundle}\n"
GENERATION_TEMPLATE = "Policy:\n{generation_policy}\n\nCandidates:\n{candidates}\n\nExisting:\n{existing_skill}\n"
REVIEW_TEMPLATE = (
    "Policy:\n{review_policy}\n\nSkill:\n{skill_md}\n\nCandidates:\n{candidates}\n\n"
    "Evidence:\n{evidence}\n\nExisting:\n{existing_skill}\n"
)
REVIEW_PASS = json.dumps({"verdict": "pass", "issues": []})
DEMO_SKILL_NAME = "demo-csv-column-sum"
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


# ------------------------------------------------------------------ sandbox


def _refuse(command: str, why: str) -> None:
    pytest.fail(f"refusing to execute a non-whitelisted example: {command!r} ({why})")


def _string_problem(literal: str) -> str | None:
    """Why a STRING token is refused: a backslash, a prefix other than ``f``, or a field that is not an allowed name."""
    if "\\" in literal:
        return "string contains a backslash"
    body = literal.lstrip("bBrRuUfF")
    prefix = literal[: len(literal) - len(body)].lower()
    if prefix not in ("", "f"):
        return f"string prefix {prefix!r} is not allowed"
    if prefix == "f":
        fields = _FSTRING_FIELD_RE.findall(body)
        if any(field not in ALLOWED_NAMES or field in ALLOWED_KEYWORDS for field in fields):
            return "f-string field is not an allowed name"
        if any(brace in _FSTRING_FIELD_RE.sub("", body) for brace in "{}"):
            return "f-string has an unbalanced or nested field"
    return None


def _opens_operand(following: list[tokenize.TokenInfo]) -> bool:
    """``following`` is exactly ``(sys.argv[<n>])``."""
    return len(following) == len(OPEN_CALL) and all(
        tokenize.tok_name[token.type] == kind and (text is None or token.string == text)
        for token, (kind, text) in zip(following, OPEN_CALL)
    )


def _code_problem(code: str) -> str | None:
    """Why ``code`` is refused, or None when every token is on the allowlist."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError) as exc:
        return f"code does not tokenize: {exc}"
    for index, token in enumerate(tokens):
        if token.type not in ALLOWED_TOKEN_TYPES:
            return f"token {token.string!r} ({tokenize.tok_name[token.type]}) is not allowed"
        if token.type == tokenize.NAME:
            if keyword.iskeyword(token.string) and token.string not in ALLOWED_KEYWORDS:
                return f"keyword {token.string!r} is not allowed"
            if token.string not in ALLOWED_NAMES:
                return f"name {token.string!r} is not allowed"
            if token.string == "open" and not _opens_operand(tokens[index + 1 : index + 1 + len(OPEN_CALL)]):
                return "open() must read exactly one positional operand, open(sys.argv[<n>])"
        elif token.type == tokenize.STRING:
            problem = _string_problem(token.string)
            if problem is not None:
                return problem
        elif token.type in _FSTRING_TYPES and "\\" in token.string:
            return "string contains a backslash"
        elif token.type == tokenize.OP and token.string == ".":
            before, after = tokens[index - 1], tokens[index + 1]
            dotted = f"{before.string}.{after.string}"
            if before.type != tokenize.NAME or after.type != tokenize.NAME or dotted not in ALLOWED_ATTRIBUTES:
                return f"attribute access {dotted!r} is not allowed"
    return None


def _run_allowlisted(command: str, cwd: Path) -> str:
    """Run ``python3 -c <code> <operands>`` in an isolated interpreter; anything else fails the test."""
    tokens = shlex.split(command)
    if len(tokens) < 3 or tokens[0] not in INTERPRETERS or tokens[1] != "-c":
        _refuse(command, "not `python3 -c <code> <operands>`")
    code, operands = tokens[2], tokens[3:]
    problem = _code_problem(code)
    if problem is not None:
        _refuse(command, problem)
    for operand in operands:
        if not OPERAND_RE.fullmatch(operand) or operand.startswith(("/", "-")) or ".." in operand or ":" in operand:
            _refuse(command, f"operand {operand!r} is not a plain relative name")
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, *operands],
        cwd=cwd,
        env={"PATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A working directory holding exactly the CSV the fixture read."""
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "sales.csv").write_text(SALES_CSV, encoding="utf-8")
    return tmp_path


def _events() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def _bash_calls(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """``(cmd, output)`` of every bash call, start joined to its result by span id."""
    starts = {e["span_id"]: e["input"]["cmd"] for e in events if e["event"] == "tool.start" and e["name"] == "bash"}
    results = {e["span_id"]: e["output"] for e in events if e["event"] == "tool.result" and e["name"] == "bash"}
    assert set(starts) == set(results)
    return [(starts[span], results[span]) for span in starts]


# ---------------------------------------------------------------- the fixture


def test_fixture_is_marked_synthetic() -> None:
    events = _events()
    header = events[0]
    assert header["event"] == "recorder.attach"
    assert header["agent"] == "SyntheticDemo" and header["working_dir"] == "/synthetic/demo"
    assert header["synthetic"] is True and header["thread_id"] == THREAD_ID
    assert len(events) == 13
    for line, event in enumerate(events, start=1):
        assert event["v"] == 1 and event["seq"] == line, event
        assert (event["thread_id"], event["agent"], event["rec"]) == (THREAD_ID, "SyntheticDemo", "rec-demo")
    trajectory = load_trajectory(FIXTURE)
    assert (trajectory.thread_id, trajectory.agent, trajectory.working_dir) == (
        THREAD_ID,
        "SyntheticDemo",
        "/synthetic/demo",
    )
    assert is_synthetic(trajectory.agent, trajectory.working_dir) is True
    assert trajectory.malformed_lines == 0


def test_fixture_command_reproduces_recorded_total_in_sandbox(sandbox: Path) -> None:
    calls = _bash_calls(_events())
    assert [output for _, output in calls] == ["60.0", "TOTAL_OK"]
    for command, output in calls:
        assert _run_allowlisted(command, sandbox) == output.strip()


# -------------------------------------------------------- the generated skill


class _Sink:
    def __init__(self) -> None:
        self.info: list[str] = []
        self.success: list[str] = []
        self.warning: list[str] = []
        self.error: list[str] = []
        self.plain: list[str] = []
        self.console = SimpleNamespace(status=lambda *_a, **_k: _NullStatus())

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


class _NullStatus:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _prompts() -> StagePrompts:
    texts = {"classify": CLASSIFY_TEMPLATE, "generate": GENERATION_TEMPLATE, "review": REVIEW_TEMPLATE}
    return StagePrompts(
        **{stage: PromptText(stage, text, f"fake/{stage}", prompt_sha256(text), 2) for stage, text in texts.items()}
    )


async def _write_demo_proposal(root: Path, fake_llm_cls) -> Path:
    """The scripted demo pipeline over the fixture; returns the written SKILL.md."""
    trajectory = load_trajectory(FIXTURE)
    notes: list[DetectorNote] = []
    episodes = extract_episodes(trajectory, demo=True, notes=notes)
    cfg = DirectSkillGenerationConfig(demo_mode=True, policy="reusable_workflow")
    bundle = build_evidence_bundle(
        episodes, [trajectory], max_chars=cfg.bundle_max_chars, excerpt_chars=cfg.excerpt_max_chars, demo=True
    )
    (episode_id,) = bundle.episode_ids
    candidate = {
        "title": "CSV column sum",
        "rule": "Sum a CSV column with csv.DictReader and compare the total with the expected value.",
        "evidence_refs": sorted(bundle.shown),
        "future_applicability": "high",
        "target": {"action": "create", "existing_skill": None},
    }
    decision = {
        "episode_ids": [episode_id],
        "decision": "accept",
        "reason_code": "accepted",
        "explanation": "scripted",
        "evidence_refs": [],
    }
    classify = json.dumps(
        {"contract_version": 2, "verdict": "save", "candidates": [candidate], "decisions": [decision]}
    )
    llm = fake_llm_cls(classify, DEMO_SKILL, REVIEW_PASS)
    sink = _Sink()
    run = RunContext(
        command="direct-skill-generation",
        working_dir=root,
        state_dir=root / "state",
        requested=cfg,
        rules=effective_rules(cfg, working_dir=root),
        prompts=_prompts(),
        skills=[],
        llm_slot=LazyLlm(session=None, llm=llm, model="fake-model", context_window=128_000),
        taken=set(),
        sink=sink,
        report_dir=None,
    )
    result = await run_thread(run, ThreadInput(trajectory, [], episodes, notes))
    assert sink.error == [] and llm.replies == [], sink.error
    (proposal,) = result.proposals
    return Path(proposal)


def _workflow_section(skill_md: str) -> str:
    match = re.search(r"^## Workflow\n(.*?)(?=^## |\Z)", skill_md, re.MULTILINE | re.DOTALL)
    assert match is not None, skill_md
    return match.group(1)


def _first_python_example(skill_md: str) -> str:
    """The first backtick-quoted ``python3 -c …`` command of the Workflow section."""
    match = re.search(r"`(python3 -c .*?)`", _workflow_section(skill_md))
    assert match is not None, skill_md
    return match.group(1)


@pytest.mark.asyncio
async def test_generated_demo_skill_example_reproduces_total(
    tmp_path: Path, sandbox: Path, fake_llm_cls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_DISABLED, "1")
    reset_exgraph_cache()
    try:
        proposal = await _write_demo_proposal(tmp_path / "project", fake_llm_cls)
    finally:
        reset_exgraph_cache()
    assert proposal == tmp_path / "project" / "skills" / ".proposals" / THREAD_ID / DEMO_SKILL_NAME / "SKILL.md"

    example = _first_python_example(proposal.read_text(encoding="utf-8"))
    for placeholder, value in PARAMETERS.items():
        example = example.replace(placeholder, value)
    assert "<" not in example and ">" not in example
    # The example is exactly the command the evidence shows, and it computes what the fixture recorded.
    (sum_command, _), _ = _bash_calls(_events())
    assert example == sum_command
    assert _run_allowlisted(example, sandbox) == "60.0"


# ---------------------------------------------------------------- allowlist


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf x",
        "python3 script.py",
        'python3 -c "import os"',
        'python3 -c "import subprocess"',
        'python3 -c "from os import system"',
        "python3 -c \"importlib.import_module('o'+'s')\"",
        "python3 -c \"open('x', 'w')\"",
        "python3 -c 'open(x, \"wt\")'",
        "python3 -c \"print(f'{__import__}')\"",
        "python3 -c \"print('\\\\x41')\"",
        'python3 -c "print(sys.argv.append)"',
        'python3 -c "print(open(sys.argv[1]).read)"',
        'python3 -c "print(1)" /etc/passwd',
        'python3 -c "print(1)" ../x',
        'python3 -c "print(1)" -x',
        'python3 -c "print(1)" a:b',
        'python3 -c "print(1)" "a b"',
        "python3",
    ],
)
def test_allowlist_refuses_anything_else(command: str, sandbox: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match="refusing to execute a non-whitelisted example"):
        _run_allowlisted(command, sandbox)
    assert sorted(p.name for p in sandbox.iterdir()) == ["input"]
