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

"""Node, edge and case identities for the experience graph (schema v1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = 1

NodeType = Literal[
    "Thread",
    "TaskAnchor",
    "Case",
    "Step",
    "SubagentRun",
    "SkillDoc",
]
EdgeType = Literal[
    "HAS_TASK",
    "CONTAINS",
    "NEXT_CASE",
    "HAS_STEP",
    "PARENT_OF",
    "DELEGATES",
    "IN_SUBAGENT",
    "DERIVED_SKILL",
]
StepKind = Literal["tool", "llm"]
Outcome = Literal["golden", "warning", "unknown"]
OutcomeOverride = Literal["success", "fail"]


def thread_id(thread: str) -> str:
    return f"thread:{thread}"


def task_anchor_id(thread: str) -> str:
    """P0 grain: one task anchor per recorded thread."""
    return f"task:thread:{thread}"


def case_id(run_id: str) -> str:
    return f"case:{run_id}"


def step_id(span_id: str) -> str:
    return f"step:{span_id}"


def subagent_id(namespace: str) -> str:
    return f"subagent:{namespace}"


def skill_doc_id(path: str) -> str:
    return f"skill:{path}"


@dataclass(slots=True)
class Node:
    id: str
    type: NodeType
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, **self.attrs}


@dataclass(slots=True)
class Edge:
    id: str
    type: EdgeType
    src: str
    dst: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "src": self.src,
            "dst": self.dst,
            **self.attrs,
        }


def edge_id(kind: EdgeType, src: str, dst: str) -> str:
    return f"edge:{kind}:{src}->{dst}"


@dataclass(slots=True)
class CaseRecord:
    """EXG case payload stored beside the graph nodes."""

    id: str
    thread_id: str
    agent: str
    run_id: str
    x: str | None
    y: str | None
    r: Outcome
    sigma: dict[str, Any]
    evidence: list[int]
    outcome_policy: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperienceGraph:
    """In-memory P0 graph for one trajectory file."""

    schema_version: int
    thread_id: str
    agent: str
    source_path: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    cases: dict[str, CaseRecord] = field(default_factory=dict)

    def add_node(self, node: Node) -> Node:
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
            return node
        existing.attrs.update(node.attrs)
        return existing

    def add_edge(self, edge: Edge) -> Edge:
        self.edges[edge.id] = edge
        return edge

    def add_case(self, record: CaseRecord) -> CaseRecord:
        self.cases[record.id] = record
        return record
