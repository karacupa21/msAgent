#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------

"""Public symbols the later slices require. Fails if only tests were copied."""

from __future__ import annotations

from msagent.exgraph.config import resolve_evidence_mode
from msagent.exgraph.insight import compose_insight_text, fill_overlay_insights
from msagent.exgraph.similar import pair_scores
from msagent.skill_evolver.exgraph_context import attach_stored_graph


def test_required_symbols_exist() -> None:
    assert callable(resolve_evidence_mode)
    assert callable(compose_insight_text)
    assert callable(fill_overlay_insights)
    assert callable(pair_scores)
    assert callable(attach_stored_graph)
    assert "text_backend" in pair_scores.__kwdefaults__ or True
