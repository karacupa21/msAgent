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

"""Status messages are marked by the theme color alone, without an icon glyph."""

from __future__ import annotations

import pytest
from rich.color import Color

from msagent.cli.theme.console import ThemedConsole
from msagent.cli.theme.tokyo_night import TokyoNightTheme

# Code points of the old prefixes: information, warning, cross mark, check mark, VS16.
OLD_ICON_CODE_POINTS: tuple[int, ...] = (0x2139, 0x26A0, 0x274C, 0x2705, 0xFE0F)


@pytest.mark.parametrize(
    ("method", "color"),
    [
        ("print_info", "info_color"),
        ("print_warning", "warning_color"),
        ("print_error", "error_color"),
        ("print_success", "success_color"),
    ],
)
def test_status_helpers_color_the_whole_message(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    color: str,
) -> None:
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.delenv("NO_COLOR", raising=False)
    theme = TokyoNightTheme()
    themed = ThemedConsole(theme)

    with themed.capture() as capture:
        getattr(themed, method)("3 proposals waiting")
    output = capture.get()

    triplet = Color.parse(getattr(theme, color)).triplet
    assert triplet is not None
    sgr = f"\x1b[38;2;{triplet.red};{triplet.green};{triplet.blue}m"
    # One colored run: the number is not re-colored by Rich's repr highlighter.
    assert f"{sgr}3 proposals waiting\x1b[0m" in output
    assert not any(chr(point) in output for point in OLD_ICON_CODE_POINTS)
