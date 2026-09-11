"""Rich styles and formatting utilities with theme support."""

import os
from typing import Literal

from rich.console import Console

from msagent.cli.theme.base import BaseTheme


class ThemedConsole:
    """Console wrapper with configurable theme."""

    def __init__(self, console_theme: BaseTheme):
        color_system: Literal["truecolor"] | None = None
        if (
            os.getenv("COLORTERM", "").lower() in {"truecolor", "24bit"}
            or os.getenv("WT_SESSION")
            or os.getenv("TERM_PROGRAM") in {"vscode", "WarpTerminal"}
            or os.getenv("WSL_DISTRO_NAME")
        ):
            color_system = "truecolor"

        self.console = Console(
            theme=console_theme.rich_theme,
            force_terminal=True,
            color_system=color_system,
        )

    def print(self, *args, style: str = "default", **kwargs):
        """Print with theme-aware styling."""
        self.console.print(*args, style=style, **kwargs)

    # Status helpers mark the message kind by color alone: the whole line takes the
    # semantic style; highlighting is off so numbers and paths keep that color.
    def print_error(self, content: str):
        """Print error message."""
        self.console.print(content, style="error", highlight=False)

    def print_warning(self, content: str):
        """Print warning message."""
        self.console.print(content, style="warning", highlight=False)

    def print_success(self, content: str):
        """Print success message."""
        self.console.print(content, style="success", highlight=False)

    def print_info(self, content: str):
        """Print neutral informational message."""
        self.console.print(content, style="info", highlight=False)

    def clear(self):
        """Clear the console."""
        self.console.clear()

    def capture(self):
        """Capture console output for measuring rendered lines."""
        return self.console.capture()

    @property
    def width(self):
        """Get console width."""
        return self.console.width
