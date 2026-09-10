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

"""Tests for the SKILL.md validator: positives and a negative for every rule."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from msagent.skill_evolver.validator import (
    DESCRIPTION_PREFIX,
    FOLKLORE_PHRASES,
    MIN_WORKFLOW_STEPS,
    SecretHit,
    ValidationResult,
    redact_secrets,
    scan_secrets,
    skill_name,
    validate_skill_md,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"

VALID_NAME = "generated-source-debugging"
VALID_DESCRIPTION = "Use when diagnosing failures involving generated source artifacts."


def _skill(
    *,
    name: str = VALID_NAME,
    description: str = VALID_DESCRIPTION,
    frontmatter: str | None = None,
    inputs: str = "Build error output and the affected target.",
    workflow: str = "1. Reproduce the failure.\n2. Check whether the symbol is generated.",
    outputs: str = "A diagnosed failure and a verified fix.",
    extra: str = "",
) -> str:
    """A valid SKILL.md unless one of the parts is overridden."""
    if frontmatter is None:
        frontmatter = f"---\nname: {name}\ndescription: {description}\n---"
    sections = [
        frontmatter,
        "# Generated Source Debugging",
        f"## Inputs\n\n{inputs}",
        f"## Workflow\n\n{workflow}",
        f"## Outputs\n\n{outputs}",
    ]
    if extra:
        sections.append(extra)
    return "\n\n".join(sections) + "\n"


DEFAULT_TEXT = _skill()
BODY_ONLY = DEFAULT_TEXT[DEFAULT_TEXT.index("# Generated") :]


def _demote(title: str) -> str:
    """Turn ``## <title>`` into an H3, so the required H2 is missing."""
    return DEFAULT_TEXT.replace(f"## {title}", f"### {title}")


# ---------------------------------------------------------------- positives


def test_valid_skill_passes() -> None:
    assert validate_skill_md(DEFAULT_TEXT) == ValidationResult(ok=True, errors=[])


def test_optional_sections_with_content_pass() -> None:
    extra = "## Constraints\n\n- Keep generated files.\n\n## Examples\n\n- Run `make gen` first."
    assert validate_skill_md(_skill(extra=extra)).ok


def test_fence_and_h3_inside_workflow_stay_inside() -> None:
    workflow = "\n".join(
        [
            "1. Reproduce the failure.",
            "   ```bash",
            "   # a comment that looks like a heading",
            "   1. not a step",
            "   ```",
            "### Details",
            "",
            "2. Check the generator.",
        ]
    )
    result = validate_skill_md(_skill(workflow=workflow))
    assert result.ok, result.errors


def test_crlf_closing_hashes_and_paren_steps_pass() -> None:
    text = _skill(workflow="1) First.\n2) Second.").replace("## Inputs", "## Inputs ##")
    assert validate_skill_md(text.replace("\n", "\r\n")).ok


def test_horizontal_rule_and_nested_bullets_pass() -> None:
    workflow = "1. First.\n   - detail\n2. Second.\n\n---\n\nAfter the rule."
    assert validate_skill_md(_skill(workflow=workflow)).ok


def test_folded_and_quoted_descriptions_pass() -> None:
    folded = _skill(description=">\n  Use when the build fails\n  on generated code.")
    quoted = _skill(description='"Use when the build fails: generated code"')
    assert validate_skill_md(folded).ok
    assert validate_skill_md(quoted).ok


def test_leading_blank_lines_pass() -> None:
    assert validate_skill_md("\n\n" + DEFAULT_TEXT).ok


@pytest.mark.parametrize("name", ["abc", "a" + "b" * 48, "x1-2"])
def test_name_bounds_pass(name: str) -> None:
    assert validate_skill_md(_skill(name=name)).ok


def test_expected_name_skips_pattern_rules() -> None:
    text = _skill(name="Legacy_Name")
    assert not validate_skill_md(text).ok
    assert validate_skill_md(text, expected_name="Legacy_Name").ok


def test_taken_names_without_the_name_pass() -> None:
    assert validate_skill_md(DEFAULT_TEXT, taken_names={"other-skill"}).ok


def test_one_step_workflow_passes() -> None:
    assert MIN_WORKFLOW_STEPS == 1
    assert validate_skill_md(_skill(workflow="1. Run the check.")).ok


def test_never_use_constraint_passes() -> None:
    # Whether a prohibition is evidence-backed is the semantic reviewer's job, not folklore.
    assert validate_skill_md(_skill(outputs="Never use --force here.")).ok
    extra = "## Constraints\n\n- Never use `--force`; the evidence shows it deleted the profile."
    assert validate_skill_md(_skill(extra=extra)).ok


@pytest.mark.parametrize(
    "text",
    [
        "token: <your-token>",
        "API_KEY=$API_KEY",
        "password: {{password}}",
        "secret: REDACTED_VALUE",
        "api_key = %API_KEY%",
        "token: [see the vault]",
        "password: example-password-here",
        # Markdown backticks around a placeholder do not turn it into a value.
        "- password: `<db-password>`",
        "- token: `$GITHUB_TOKEN`",
        "- token: `GITHUB_TOKEN` environment variable",
    ],
)
def test_placeholder_credentials_pass(text: str) -> None:
    assert validate_skill_md(_skill(outputs=text)).ok
    assert scan_secrets(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "- password: required, prompted interactively",
        "- token: obtained from the CI settings page",
        "- api_key: required for private repositories",
    ],
)
def test_credential_prose_passes(text: str) -> None:
    # An Inputs line "<name>: <description>" is prose, not a credential value.
    assert validate_skill_md(_skill(inputs=text)).ok
    assert scan_secrets(text) == []


def test_prose_guard_keeps_real_values() -> None:
    # Pinned positives: a digit, an uppercase key at the end of the line, a value before more words.
    for text in ("password: " + "hunter2" + "secret", "API_KEY=" + "abcd" * 3, "token: " + "hunter2" + "secret ok"):
        assert [hit.label for hit in scan_secrets(text)] == ["credential assignment"], text
        assert not validate_skill_md(_skill(outputs=text)).ok


def test_required_prefix_rules() -> None:
    assert validate_skill_md(_skill(name="csv-sum"), required_prefix="demo-").errors == [
        "name: 'csv-sum' must start with 'demo-' (demo proposal)"
    ]
    assert validate_skill_md(_skill(name="demo-"), required_prefix="demo-").errors == [
        "name: 'demo-' has no task class after 'demo-'"
    ]
    assert validate_skill_md(_skill(name="demo-csv-sum"), required_prefix="demo-").ok
    # A demo name is never read as a task identifier (fix-/debug-/audit- anchor at the start).
    assert validate_skill_md(_skill(name="demo-fix-x"), required_prefix="demo-").ok
    # Updates keep their name; the prefix rule applies to new skills only.
    assert validate_skill_md(_skill(name="real"), expected_name="real", required_prefix="demo-").ok


# ---------------------------------------------------------------- negatives


NEGATIVE_CASES = [
    pytest.param("", "empty reply", id="empty"),
    pytest.param("```markdown\n" + DEFAULT_TEXT + "```\n", "code fence", id="fenced"),
    pytest.param(BODY_ONLY, "no YAML frontmatter", id="no-frontmatter"),
    pytest.param(
        _skill(frontmatter=f"---\nname: {VALID_NAME}\ndescription: {VALID_DESCRIPTION}"),
        "invalid frontmatter",
        id="missing-closing-delimiter",
    ),
    pytest.param(_skill(description="[oops"), "invalid frontmatter", id="yaml-error"),
    pytest.param(_skill(frontmatter="---\n- a\n- b\n---"), "invalid frontmatter", id="list"),
    pytest.param(
        _skill(frontmatter=f"---name: {VALID_NAME}\ndescription: {VALID_DESCRIPTION}\n---"),
        "alone on their lines",
        id="delimiter-not-alone",
    ),
    pytest.param(
        _skill(frontmatter=f"---\nname: {VALID_NAME}\ndescription: {VALID_DESCRIPTION}\n--- # x"),
        "start on a new line",
        id="text-after-closing-delimiter",
    ),
    pytest.param(_skill(name="123"), "name: must be a string, got int", id="name-int"),
    pytest.param(_skill(name="yes"), "name: must be a string, got bool", id="name-bool"),
    pytest.param(
        _skill(frontmatter=f"---\ndescription: {VALID_DESCRIPTION}\n---"),
        "name: missing",
        id="name-missing",
    ),
    pytest.param(_skill(name='""'), "name: must not be empty", id="name-empty"),
    pytest.param(_skill(name="Bad-Name"), "must match", id="name-uppercase"),
    pytest.param(_skill(name="foo_bar"), "must match", id="name-underscore"),
    pytest.param(_skill(name="ab"), "must match", id="name-too-short"),
    pytest.param(_skill(name="a" * 50), "must match", id="name-too-long"),
    pytest.param(_skill(name='"123"'), "must match", id="name-digits"),
    pytest.param(_skill(name="pr-42"), "task identifier", id="name-pr"),
    pytest.param(_skill(name="issue-7"), "task identifier", id="name-issue"),
    pytest.param(_skill(name="bug-12"), "task identifier", id="name-bug"),
    pytest.param(_skill(name="ticket-3"), "task identifier", id="name-ticket"),
    pytest.param(_skill(name="fix-build"), "task identifier", id="name-fix"),
    pytest.param(_skill(name="debug-x"), "task identifier", id="name-debug"),
    pytest.param(_skill(name="audit-y"), "task identifier", id="name-audit"),
    pytest.param(
        _skill(frontmatter=f"---\nname: {VALID_NAME}\n---"),
        "description: missing",
        id="description-missing",
    ),
    pytest.param(_skill(description="5"), "description: must be a string", id="description-int"),
    pytest.param(
        _skill(description="Instructions for debugging"),
        "must start with 'Use when '",
        id="description-not-proactive",
    ),
    pytest.param(_skill(description="use when x"), "must start with", id="description-lowercase"),
    pytest.param(_demote("Inputs"), "missing section '## Inputs'", id="inputs-missing"),
    pytest.param(_demote("Workflow"), "missing section '## Workflow'", id="workflow-missing"),
    pytest.param(_demote("Outputs"), "missing section '## Outputs'", id="outputs-missing"),
    pytest.param(
        _skill(extra="## Workflow\n\n1. a\n2. b"),
        "duplicate section '## Workflow'",
        id="workflow-duplicate",
    ),
    pytest.param(_skill(workflow="- a\n- b"), "found 0", id="workflow-bullets"),
    pytest.param(_skill(workflow="```\n1. a\n2. b\n```"), "found 0", id="workflow-fenced"),
    pytest.param(_skill(workflow="    1. a\n    2. b"), "found 0", id="workflow-indented"),
    pytest.param(_skill(extra="## Constraints\n\n## Examples\n\n- x"), "Constraints: section is empty", id="constraints-empty"),
    pytest.param(_skill(extra="## Examples\n"), "Examples: section is empty", id="examples-empty"),
    pytest.param(_skill(outputs="The linter is broken."), "folklore 'is broken'", id="folklore-broken"),
    pytest.param(_skill(outputs="It does not work."), "folklore 'does not work'", id="folklore-work"),
    pytest.param(_skill(outputs="这个模块不工作。"), "negative tool folklore", id="folklore-zh-1"),
    pytest.param(_skill(outputs="构建坏了。"), "negative tool folklore", id="folklore-zh-2"),
    pytest.param(
        _skill(description="Use when the build is broken."),
        "negative tool folklore",
        id="folklore-in-description",
    ),
]


@pytest.mark.parametrize(("text", "fragment"), NEGATIVE_CASES)
def test_rule_violation_is_rejected(text: str, fragment: str) -> None:
    result = validate_skill_md(text)

    assert result.ok is False
    assert any(fragment in error for error in result.errors), result.errors


@pytest.mark.parametrize(
    "text",
    [
        _skill(name="pr-42"),
        _skill(description="Instructions for debugging"),
        _demote("Inputs"),
        _skill(workflow="- only"),
        _skill(extra="## Examples\n"),
        _skill(outputs="It does not work."),
        _skill(outputs="password: " + "hunter2" + "secret"),
    ],
    ids=["task-id", "description", "section", "workflow", "empty-optional", "folklore", "secret"],
)
def test_single_violation_reports_one_error(text: str) -> None:
    assert len(validate_skill_md(text).errors) == 1


def test_zero_steps_message_is_singular() -> None:
    assert validate_skill_md(_skill(workflow="- a")).errors == ["Workflow: needs at least 1 numbered step, found 0"]


def test_expected_name_mismatch_rejected() -> None:
    result = validate_skill_md(DEFAULT_TEXT, expected_name="other-skill")
    assert result.errors == [f"name: must stay 'other-skill' (update), got '{VALID_NAME}'"]


def test_taken_name_rejected() -> None:
    result = validate_skill_md(DEFAULT_TEXT, taken_names={VALID_NAME})
    assert result.errors == [f"name: '{VALID_NAME}' already exists in the skill library"]


def test_collects_all_errors() -> None:
    text = _skill(
        name="pr-1",
        description="Instructions for debugging",
        workflow="- only",
        extra="## Constraints\n",
    )
    errors = validate_skill_md(text).errors

    assert len(errors) == 4
    assert any("task identifier" in e for e in errors)
    assert any("must start with" in e for e in errors)
    assert any(f"at least {MIN_WORKFLOW_STEPS}" in e for e in errors)
    assert any("Constraints: section is empty" in e for e in errors)


def test_folklore_error_names_the_line() -> None:
    text = _skill(outputs="First line.\nThe tool is broken.")
    line = text.split("\n").index("The tool is broken.") + 1

    assert validate_skill_md(text).errors == [f"line {line}: negative tool folklore 'is broken'"]


def test_folklore_phrases_are_the_specified_ones() -> None:
    assert FOLKLORE_PHRASES == ("is broken", "does not work", "不工作", "坏了")
    assert DESCRIPTION_PREFIX == "Use when "


# ------------------------------------------------------------------- secrets

# Values are built at runtime so no secret-shaped literal is committed.
SECRET_CASES = [
    pytest.param("AKIA" + "A" * 16, "aws access key", id="aws"),
    pytest.param("-----BEGIN RSA PRIVATE KEY-----", "private key block", id="pem"),
    pytest.param("ghp_" + "a" * 36, "github token", id="github"),
    pytest.param("github_pat_" + "b" * 22, "github token", id="github-pat"),
    pytest.param("xoxb-" + "1" * 12, "slack token", id="slack"),
    pytest.param("sk-" + "a" * 24, "openai-style key", id="openai"),
    pytest.param("eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12, "jwt", id="jwt"),
    pytest.param("Bearer " + "x" * 24, "bearer token", id="bearer"),
    pytest.param("password: " + "hunter2" + "secret", "credential assignment", id="password"),
    pytest.param("API_KEY=" + "abcd" * 3, "credential assignment", id="api-key"),
    pytest.param("export GITHUB_TOKEN=" + "abcdef0123" * 2, "credential assignment", id="prefixed-key"),
    pytest.param("AWS_SECRET_ACCESS_KEY=" + "q" * 24, "credential assignment", id="suffixed-key"),
    pytest.param('"password": "' + "hunter2" * 2 + '"', "credential assignment", id="json-key"),
]


@pytest.mark.parametrize(("value", "label"), SECRET_CASES)
def test_secret_patterns_are_rejected_with_line_and_label(value: str, label: str) -> None:
    text = _skill(outputs=f"First line.\nUse {value} here.")
    line = text.split("\n").index(f"Use {value} here.") + 1

    result = validate_skill_md(text)

    assert result.ok is False
    assert result.errors == [f"line {line}: possible secret ({label}); remove it or replace it with a parameter"]
    # The value itself is never echoed back.
    secret = value.split(" ", 1)[-1].split(": ", 1)[-1].split("=")[-1]
    assert secret not in result.errors[0]


def test_scan_secrets_reports_spans_per_line() -> None:
    bearer = "Bearer " + "x" * 24
    text = "clean\n" + f"curl -H '{bearer}'\n" + "token: " + "hunter2" + "secret"

    hits = scan_secrets(text)

    assert hits == [
        SecretHit(line=2, label="bearer token", start=9, end=9 + len(bearer)),
        SecretHit(line=3, label="credential assignment", start=7, end=7 + len("hunter2secret")),
    ]


def test_redact_secrets_replaces_value_only() -> None:
    assert redact_secrets("token: " + "hunter2" + "secret ok") == "token: [REDACTED:credential assignment] ok"
    bearer = "Bearer " + "x" * 24
    assert redact_secrets(f"a\ncurl -H '{bearer}'\nb") == "a\ncurl -H '[REDACTED:bearer token]'\nb"
    clean = "nothing to see: <your-token> and $API_KEY"
    assert redact_secrets(clean) == clean
    assert redact_secrets(DEFAULT_TEXT) == DEFAULT_TEXT


def test_skill_name_reads_validated_content() -> None:
    assert skill_name(_skill(name="foo-bar")) == "foo-bar"
    assert skill_name(_skill(name="foo-bar").replace("\n", "\r\n")) == "foo-bar"
    with pytest.raises(ValueError):
        skill_name(BODY_ONLY)


# ----------------------------------------------------------------- isolation

_ISOLATION_PROBE = """
import sys
import msagent.skill_evolver.validator as validator
import msagent.skill_evolver.writer as writer
import msagent.skill_evolver.render as render
import msagent.skill_evolver.review as review
import msagent.skill_evolver.policy as policy
for module in (validator, writer, render, review, policy):
    assert module.__file__.startswith(sys.argv[1]), module.__file__
banned = ("langchain", "langgraph", "httpx", "requests", "urllib3", "aiohttp",
          "urllib.request", "http.client")
leaked = sorted(m for m in sys.modules if m.startswith(banned))
assert not leaked, leaked
"""


def test_validator_writer_render_review_import_no_langchain_or_network() -> None:
    # The pytest process already has langchain loaded (conftest imports the CLI
    # initializer), so the check runs in a fresh interpreter; PYTHONPATH must
    # win over the wheel that may be installed in the virtualenv.
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
