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

"""Experience graph built from recorded msAgent trajectories.

P0–P2.1: one TaskAnchor per thread, Cases/Steps from the recorder, SkillDoc
pointers, evolver Episodes + FIXED_BY + workspace Recipes, SIMILAR_TO, Insight.
Classify evidence_mode: episodes | hybrid | graph. Appendix is relations only.

This ``__init__`` imports nothing so ``python -m msagent.exgraph.export`` stays
free of langchain. Import :mod:`msagent.exgraph.cases` or ``export`` directly.
"""
