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

"""Code validation of a generated SKILL.md.

:func:`validate_skill_md` checks the text the generation stage produced
against the structure the generation prompt demands: YAML frontmatter with a
durable ``name`` (with the ``demo-`` prefix when ``required_prefix`` asks for
it) and a proactive ``description``, the mandatory ``Inputs`` / ``Workflow``
/ ``Outputs`` sections, a numbered workflow of at least one step, non-empty
optional sections, no negative tool folklore and no secret-looking value.
Every violation is reported, because the corrective LLM call needs the whole
list; nothing is repaired.

A prohibition such as "Never use --force" is not rejected here: whether a
constraint is evidence-backed is the semantic reviewer's job (issue code
``unsafe_claim``). The secret rules are deterministic patterns
(:data:`SECRET_PATTERNS`); a matched value is never echoed in an error, and
:func:`redact_secrets` replaces it with ``[REDACTED:<label>]`` for
provenance.

Stdlib plus :meth:`SkillFactory.parse_frontmatter`, so "valid" means "the
skill loader reads the same frontmatter". Importing this module must not
load langchain.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, field

from msagent.skills.factory import SkillFactory

# Skill names the library accepts as folder names: kebab-case, 3..49 chars.
NAME_RE = re.compile(r"[a-z][a-z0-9-]{2,48}")
# Names that describe one task instead of a task class; tried with .match on
# a name that already satisfies NAME_RE.
TASK_ID_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d+$"),
    re.compile(r"(pr|issue|bug|ticket)-\d+"),
    re.compile(r"(fix|debug|audit)-"),
)
DESCRIPTION_PREFIX = "Use when "
REQUIRED_SECTIONS = ("Inputs", "Workflow", "Outputs")
# Optional sections that must not be empty when present.
OPTIONAL_SECTIONS = ("Constraints", "Examples")
MIN_WORKFLOW_STEPS = 1
# Negative tool folklore ("tool X is broken") the prompt forbids. Matched
# case-insensitively anywhere in the text. "never use" is not folklore: a
# prohibition is allowed when the evidence shows the failure it prevents,
# which the semantic reviewer checks (unsafe_claim).
FOLKLORE_PHRASES = (
    "is broken",
    "does not work",
    "不工作",
    "坏了",
)
# Secret-looking values a SKILL.md or a provenance record must never carry.
# Each pattern is (label, regex); the assignment rule captures the value as
# group 1 and is skipped for placeholders (see _is_placeholder).
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("slack token", re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b")),
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE)),
    (
        "credential assignment",
        # No leading \b: prefixed keys (GITHUB_TOKEN=, DB_PASSWORD=, access_token=)
        # match; the suffix group covers AWS_SECRET_ACCESS_KEY=, the optional
        # quote before the separator the JSON form ("password": "...") in which
        # bundle excerpts render tool arguments.
        re.compile(
            r"(?i)(?:password|passwd|secret|token|api[_-]?key)(?:_[a-z0-9]+)*\b\s*[\"']?\s*[:=]\s*['\"]?([^\s'\"]{8,})"
        ),
    ),
)
# A credential value is a placeholder when it starts with one of these, ...
PLACEHOLDER_STARTS = ("<", "$", "{", "%", "[")
# ... contains one of these words (case-insensitive), ...
PLACEHOLDER_MARKERS = ("your", "example", "placeholder", "redacted", "xxx")
# ... or is a bare environment-variable name.
_ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*")
# After "<key>:", a lowercase word ending a clause or followed by more words is
# prose ("- password: required, prompted interactively"), not a value.
_PROSE_WORD_RE = re.compile(r"[a-z]+[,.;:]?")

_FOLKLORE_RE = re.compile(
    "|".join(re.escape(phrase) for phrase in FOLKLORE_PHRASES),
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*$")
_CLOSING_HASHES_RE = re.compile(r"[ \t#]+$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(?:`{3,}|~{3,})")
# A top-level numbered list item: "1. step" or "1) step".
_STEP_RE = re.compile(r"^[ ]{0,3}\d+[.)][ \t]+\S")
_BODY_START_RE = re.compile(r"[ \t]*\n")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of :func:`validate_skill_md`; ``errors`` is empty when ``ok``."""

    ok: bool
    errors: list[str]


@dataclass(frozen=True, slots=True)
class SecretHit:
    """One secret-looking value: its 1-based line and the span to redact on that line."""

    line: int
    label: str
    start: int
    end: int


@dataclass(slots=True)
class _Section:
    """One H1/H2 section: its raw lines, and the same lines with fences blanked."""

    level: int
    title: str
    lines: list[str] = field(default_factory=list)
    visible: list[str] = field(default_factory=list)


def _normalize(content: str) -> str:
    """Unify line endings so the line-based checks see one newline style."""
    return content.replace("\r\n", "\n").replace("\r", "\n")


def _mask_fences(lines: list[str]) -> list[str]:
    """Blank every line inside a ``` or ~~~ fence, the fence lines included."""
    masked: list[str] = []
    in_fence = False
    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            masked.append("")
            continue
        masked.append("" if in_fence else line)
    return masked


def _split_sections(body: str) -> list[_Section]:
    """Sections delimited by H1/H2 headings outside fences; H3+ stay inside."""
    lines = body.split("\n")
    sections: list[_Section] = []
    current: _Section | None = None
    for raw, shown in zip(lines, _mask_fences(lines)):
        match = _HEADING_RE.match(shown)
        if match and len(match.group(1)) <= 2:
            title = _CLOSING_HASHES_RE.sub("", match.group(2))
            current = _Section(level=len(match.group(1)), title=title)
            sections.append(current)
        elif current is not None:
            current.lines.append(raw)
            current.visible.append(shown)
    return sections


def _split_frontmatter(
    text: str,
    errors: list[str],
) -> tuple[dict[str, object] | None, str]:
    """Parse the frontmatter as the skill loader does; return (mapping, body)."""
    if not text:
        errors.append("empty reply")
        return None, ""
    if text.startswith("```"):
        errors.append("reply is wrapped in a code fence; send the bare SKILL.md")
        return None, ""
    if not text.startswith("---"):
        errors.append("no YAML frontmatter: the file must start with '---'")
        return None, text
    try:
        frontmatter = SkillFactory.parse_frontmatter(text)
    except ValueError as exc:
        errors.append(f"invalid frontmatter: {exc}")
        return None, text
    _, yaml_text, body = text.split("---", 2)
    if not (yaml_text.startswith("\n") and yaml_text.endswith("\n")):
        errors.append("frontmatter '---' delimiters must be alone on their lines")
    if body and not _BODY_START_RE.match(body):
        errors.append("text after the closing '---' must start on a new line")
    return frontmatter, body


def _check_name(
    frontmatter: dict[str, object],
    errors: list[str],
    *,
    expected_name: str | None,
    taken_names: Collection[str],
    required_prefix: str | None,
) -> None:
    """``name``: durable kebab-case for a new skill, or exactly the updated one."""
    if "name" not in frontmatter:
        errors.append("name: missing")
        return
    raw = frontmatter["name"]
    if not isinstance(raw, str):
        errors.append(f"name: must be a string, got {type(raw).__name__}")
        return
    name = raw.strip()
    if not name:
        errors.append("name: must not be empty")
        return
    if expected_name is not None:
        if name != expected_name:
            errors.append(f"name: must stay '{expected_name}' (update), got '{name}'")
        return
    if not NAME_RE.fullmatch(name):
        errors.append(f"name: '{name}' must match ^[a-z][a-z0-9-]{{2,48}}$")
        return
    if required_prefix is not None:
        if not name.startswith(required_prefix):
            errors.append(f"name: '{name}' must start with '{required_prefix}' (demo proposal)")
        elif len(name) <= len(required_prefix):
            errors.append(f"name: '{name}' has no task class after '{required_prefix}'")
    if any(pattern.match(name) for pattern in TASK_ID_RES):
        errors.append(f"name: '{name}' is a task identifier, not a skill name")
    if name in taken_names:
        errors.append(f"name: '{name}' already exists in the skill library")


def _check_description(frontmatter: dict[str, object], errors: list[str]) -> None:
    """``description``: a proactive trigger that starts with "Use when "."""
    if "description" not in frontmatter:
        errors.append("description: missing")
        return
    raw = frontmatter["description"]
    if not isinstance(raw, str):
        errors.append(f"description: must be a string, got {type(raw).__name__}")
        return
    text = raw.strip()
    if not text:
        errors.append("description: must not be empty")
        return
    if not text.startswith(DESCRIPTION_PREFIX):
        head = text[:60]
        prefix = DESCRIPTION_PREFIX
        errors.append(f"description: must start with '{prefix}', got '{head}'")


def _check_sections(body: str, errors: list[str]) -> None:
    """Required H2 sections once each, a numbered Workflow, no empty optional ones."""
    by_title: dict[str, list[_Section]] = {}
    for section in _split_sections(body):
        if section.level == 2:
            by_title.setdefault(section.title, []).append(section)
    for title in REQUIRED_SECTIONS:
        found = by_title.get(title, [])
        if not found:
            errors.append(f"missing section '## {title}'")
        elif len(found) > 1:
            errors.append(f"duplicate section '## {title}'")
    workflow = by_title.get("Workflow")
    if workflow:
        steps = sum(1 for line in workflow[0].visible if _STEP_RE.match(line))
        if steps < MIN_WORKFLOW_STEPS:
            need = MIN_WORKFLOW_STEPS
            noun = "step" if need == 1 else "steps"
            errors.append(f"Workflow: needs at least {need} numbered {noun}, found {steps}")
    for title in OPTIONAL_SECTIONS:
        for section in by_title.get(title, []):
            if not "".join(section.lines).strip():
                errors.append(f"{title}: section is empty")


def _check_folklore(text: str, errors: list[str]) -> None:
    """Report every negative-folklore phrase with its line number."""
    for number, line in enumerate(text.split("\n"), start=1):
        for match in _FOLKLORE_RE.finditer(line):
            errors.append(f"line {number}: negative tool folklore '{match.group(0)}'")


def _is_placeholder(value: str) -> bool:
    """A credential value that is obviously not a real one (markdown backticks ignored)."""
    value = value.strip("`")
    lowered = value.lower()
    return (
        value.startswith(PLACEHOLDER_STARTS)
        or any(marker in lowered for marker in PLACEHOLDER_MARKERS)
        or _ENV_NAME_RE.fullmatch(value) is not None
    )


def _is_prose(match: re.Match[str], line: str) -> bool:
    """``<name>: <description>`` prose after a credential-named input, not an assignment (``KEY=value …``)."""
    value, head = match.group(1), line[match.start() : match.start(1)]
    if "=" in head or _PROSE_WORD_RE.fullmatch(value) is None:
        return False
    return value[-1] in ",.;:" or re.match(r"\s+\S", line[match.end() :]) is not None


def scan_secrets(text: str) -> list[SecretHit]:
    """Every secret-looking value of ``text``, by line; overlapping spans keep the earliest."""
    hits: list[SecretHit] = []
    for number, line in enumerate(_normalize(text).split("\n"), start=1):
        found: list[tuple[int, int, str]] = []
        for label, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(line):
                group = 1 if match.lastindex else 0
                if group and (_is_placeholder(match.group(1)) or _is_prose(match, line)):
                    continue
                found.append((*match.span(group), label))
        last_end = -1
        for start, end, label in sorted(found, key=lambda span: (span[0], -span[1])):
            if start < last_end:
                continue
            hits.append(SecretHit(number, label, start, end))
            last_end = end
    return hits


def redact_secrets(text: str) -> str:
    """``text`` with every :func:`scan_secrets` span replaced by ``[REDACTED:<label>]``."""
    hits = scan_secrets(text)
    if not hits:
        return text
    lines = _normalize(text).split("\n")
    for hit in reversed(hits):
        line = lines[hit.line - 1]
        lines[hit.line - 1] = f"{line[: hit.start]}[REDACTED:{hit.label}]{line[hit.end :]}"
    return "\n".join(lines)


def _check_secrets(text: str, errors: list[str]) -> None:
    """Report every secret-looking value by line and label; the value itself is never echoed."""
    for hit in scan_secrets(text):
        errors.append(f"line {hit.line}: possible secret ({hit.label}); remove it or replace it with a parameter")


def validate_skill_md(
    content: str,
    *,
    expected_name: str | None = None,
    taken_names: Collection[str] = (),
    required_prefix: str | None = None,
) -> ValidationResult:
    """Check a generated SKILL.md; every violation is listed, nothing is repaired.

    ``expected_name`` is the name of the skill being updated: the frontmatter
    name must equal it, and the naming rules for new skills are skipped.
    ``taken_names`` are library names a new skill must not reuse.
    ``required_prefix`` (``demo-`` for a demo proposal) must start the name of
    a new skill and leave a task class after it; ignored for updates.
    """
    errors: list[str] = []
    text = _normalize(content).lstrip()
    frontmatter, body = _split_frontmatter(text, errors)
    if frontmatter is not None:
        _check_name(
            frontmatter,
            errors,
            expected_name=expected_name,
            taken_names=taken_names,
            required_prefix=required_prefix,
        )
        _check_description(frontmatter, errors)
    if frontmatter is not None or body:
        _check_sections(body, errors)
    _check_folklore(text, errors)
    _check_secrets(text, errors)
    return ValidationResult(ok=not errors, errors=errors)


def skill_name(content: str) -> str:
    """Frontmatter ``name`` of a SKILL.md that passed :func:`validate_skill_md`."""
    frontmatter = SkillFactory.parse_frontmatter(_normalize(content))
    name = frontmatter.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("SKILL.md has no frontmatter name")
    return name.strip()
