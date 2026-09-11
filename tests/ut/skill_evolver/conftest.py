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

"""Shared fixtures of the skill_evolver tests: a scripted fake LLM."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage


class FakeLLM:
    """Records every payload and answers with the scripted replies in order.

    A ``str`` reply is wrapped into an ``AIMessage`` unless ``raw`` is set;
    any other object is returned as is. Calling it more often than scripted
    is a test failure.
    """

    def __init__(self, *replies: Any, raw: bool = False) -> None:
        self.replies = list(replies)
        self.raw = raw
        self.payloads: list[list[tuple[str, str]]] = []

    async def ainvoke(self, payload: list[tuple[str, str]]) -> Any:
        self.payloads.append(list(payload))
        assert self.replies, "fake LLM called more often than scripted"
        reply = self.replies.pop(0)
        if isinstance(reply, str) and not self.raw:
            return AIMessage(content=reply)
        return reply


@pytest.fixture
def fake_llm_cls() -> type[FakeLLM]:
    return FakeLLM


@pytest.fixture(autouse=True)
def reset_daemon_config_cache() -> Iterator[None]:
    """The daemon config is cached per process; keep it from leaking between tests."""
    from msagent.skill_evolver.daemon.config import reset_config_cache

    reset_config_cache()
    yield
    reset_config_cache()
