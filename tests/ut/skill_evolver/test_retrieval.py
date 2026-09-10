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

"""Tests for the stdlib BM25 used by skill_gap detection (no LLM, no network)."""

from __future__ import annotations

import pytest

from msagent.skill_evolver.retrieval import STOPWORDS, BM25Index, SkillDoc, tokenize

DOCS = [
    SkillDoc("cluster-analysis", "Run the clustering workflow over profiler data"),
    SkillDoc("dit-quant", "Quantize DiT diffusion models with int8 calibration"),
    SkillDoc("ep-parallel", "Adapt expert parallel models for msmodelslim"),
    SkillDoc(
        "profiling-bottleneck",
        "Profile a training run with msprof and locate the kernel bottleneck",
    ),
]


def test_tokenize_splits_casefolds_and_drops_stopwords() -> None:
    assert tokenize("Cluster-Analysis of the 性能剖析, v2!") == [
        "cluster",
        "analysis",
        "性能",
        "能剖",
        "剖析",
        "v2",
    ]


def test_tokenize_drops_short_tokens_and_splits_underscores() -> None:
    assert tokenize("read_file x 42 的 和") == ["read", "file", "42"]


def test_stopwords_cover_both_languages() -> None:
    assert {"the", "and", "with", "的", "了", "不"} <= STOPWORDS


def test_tokenize_cuts_han_runs_into_bigrams() -> None:
    # Chinese is not space-delimited: a run is one phrase, indexed as bigrams,
    # and a stopword drops out without cutting the compound written around it.
    assert tokenize("分析配置文件的性能") == ["分析", "析配", "配置", "置文", "文件", "件的", "的性", "性能"]
    # A one-character run stays a unigram, and drops out when it is a stopword.
    assert tokenize("很慢 的 x") == ["很慢"]


def test_search_matches_a_chinese_description() -> None:
    docs = [
        SkillDoc("profiling-bottleneck", "用 msprof 分析训练性能并定位内核瓶颈"),
        SkillDoc("dit-quant", "量化 DiT 扩散模型"),
    ]

    (hit,) = BM25Index(docs).search("性能分析")

    assert hit.doc.name == "profiling-bottleneck"
    assert hit.matched == ("分析", "性能")


def test_search_ranks_by_shared_rare_terms_and_reports_matches() -> None:
    hits = BM25Index(DOCS).search("Quantize the DiT diffusion models for profiler data")

    assert [hit.doc.name for hit in hits] == [
        "dit-quant",
        "cluster-analysis",
        "ep-parallel",
    ]
    assert hits[0].matched == ("diffusion", "dit", "models", "quantize")
    assert hits[1].matched == ("data", "profiler")
    assert hits[2].matched == ("models",)
    assert hits[0].score > hits[1].score > hits[2].score > 0


def test_search_edge_cases() -> None:
    index = BM25Index(DOCS)

    assert index.search("") == []
    assert index.search("the and 的") == []
    assert index.search("nothing shared here") == []
    assert len(index.search("models")) == 2
    assert len(index.search("models", top_k=1)) == 1
    assert BM25Index([]).search("quantize") == []
    assert len(BM25Index([])) == 0
    assert len(index) == 4
    with pytest.raises(ValueError, match="top_k"):
        index.search("models", top_k=0)


def test_search_is_deterministic_with_name_tiebreak() -> None:
    twins = [SkillDoc("b-skill", "tune kernels"), SkillDoc("a-skill", "tune kernels")]
    hits = BM25Index(twins).search("tune kernels")

    assert [hit.doc.name for hit in hits] == ["a-skill", "b-skill"]
    assert hits[0].score == pytest.approx(hits[1].score)


def test_rare_terms_outrank_common_terms() -> None:
    docs = [
        SkillDoc("one", "alpha beta"),
        SkillDoc("two", "alpha gamma"),
        SkillDoc("three", "alpha delta"),
    ]
    hits = BM25Index(docs).search("alpha gamma")

    assert [hit.doc.name for hit in hits] == ["two", "one", "three"]
    assert hits[0].score > hits[1].score == pytest.approx(hits[2].score)
    assert hits[2].score > 0
