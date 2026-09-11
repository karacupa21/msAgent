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

"""LLM-call and context budgets of one thread.

Rules:

* Every ``ainvoke`` of a thread goes through one :class:`CountingLlm`
  (limit ``generation.max_llm_calls_per_thread``). A call is counted before
  the transport call, so a failed transport still spent it; transport
  retries inside the client are invisible here. Past the limit the wrapper
  raises :class:`LlmBudgetExhausted` and nothing is written for that plan.
* The per-thread ceiling is classify (2: call + corrective retry), the
  optional expand round (2) and 6 per plan (generate 2, review 2, revise 1,
  review 1); :func:`llm_call_bound` caps it by the configured limit.
* :class:`ContextBudget` bounds the whole prompt text (template + library +
  policy + evidence + existing skill) by the model context window: 4 chars
  per token, 60 % usable, 2048 tokens reserved for the reply. An unknown
  window enforces nothing.

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# classify + one corrective retry
CLASSIFY_CALLS = 2
# the expand_context_once round (same shape)
EXPAND_CALLS = 2
# generate 2 (first + validator correction) + review 2 (first + parse retry) + revise 1 + review 1
PLAN_CALLS = 6
CHARS_PER_TOKEN = 4
USABLE_RATIO = 0.6
REPLY_RESERVE_TOKENS = 2048
# Below this many characters a bundle is not worth cutting; the thread stops instead.
MIN_BUNDLE_CHARS = 500


class LlmBudgetExhausted(RuntimeError):
    """The thread's LLM call limit is spent; the current plan is not written."""

    def __init__(self, used: int, limit: int) -> None:
        self.used = used
        self.limit = limit
        super().__init__(f"LLM call budget exhausted ({used}/{limit} calls per thread)")


@dataclass(slots=True)
class CountingLlm:
    """Counts every ``ainvoke`` of ``inner`` and refuses the call past ``limit``."""

    inner: Any
    limit: int
    calls_used: int = 0

    @property
    def exhausted(self) -> bool:
        return self.calls_used >= self.limit

    async def ainvoke(self, payload: Any) -> Any:
        if self.exhausted:
            raise LlmBudgetExhausted(self.calls_used, self.limit)
        # Counted before the transport call: a transport error still spent the call.
        self.calls_used += 1
        return await self.inner.ainvoke(payload)


def thread_call_ceiling(max_plans: int, *, expand: bool) -> int:
    """Most LLM calls one thread can make before the configured limit applies."""
    return CLASSIFY_CALLS + (EXPAND_CALLS if expand else 0) + PLAN_CALLS * max_plans


def llm_call_bound(threads: int, max_plans: int, *, max_llm_calls: int = 16, expand: bool = True) -> int:
    """Most LLM calls a run over ``threads`` passing threads can make."""
    return threads * min(max_llm_calls, thread_call_ceiling(max_plans, expand=expand))


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Characters the prompt may use for a model with ``context_window`` tokens."""

    context_window: int | None
    # None when the window is unknown (nothing is enforced); never negative otherwise.
    available_chars: int | None

    @classmethod
    def for_window(cls, context_window: int | None) -> ContextBudget:
        if context_window is None or context_window <= 0:
            return cls(None, None)
        usable = int(context_window * CHARS_PER_TOKEN * USABLE_RATIO) - REPLY_RESERVE_TOKENS * CHARS_PER_TOKEN
        return cls(context_window, max(0, usable))

    def fits(self, chars: int) -> bool:
        return self.available_chars is None or chars <= self.available_chars

    def room_for(self, overhead_chars: int) -> int | None:
        """Characters left for the variable part after ``overhead_chars``; None when not enforced."""
        if self.available_chars is None:
            return None
        return max(0, self.available_chars - overhead_chars)

    def describe(self) -> str:
        if self.available_chars is None:
            return "context window unknown; not enforced"
        return f"context window {self.context_window} tokens -> {self.available_chars} chars for the prompt"
