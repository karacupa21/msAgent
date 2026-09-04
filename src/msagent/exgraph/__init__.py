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

P0 materializes the execution layer only: one TaskAnchor per thread, one Case
per user turn, Steps for tools and LLM calls, optional SkillDoc nodes when a
skill proposal was generated from that thread.

This ``__init__`` imports nothing so ``python -m msagent.exgraph.export`` stays
free of langchain. Import :mod:`msagent.exgraph.cases` or ``export`` directly.
"""
