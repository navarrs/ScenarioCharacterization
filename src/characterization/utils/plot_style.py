"""Single source of truth for the figure font used by every matplotlib/seaborn plot in the project."""

from logging import Logger

import matplotlib as mpl
from matplotlib import font_manager

from characterization.utils.io_utils import get_logger

_LOGGER = get_logger(__name__)

# Preferred figure font.
DEFAULT_FONT = "DM Sans"

# Caches the expensive font-availability scan: `font` is the family it was run for, `available` the result.
_font_state: dict[str, object] = {"font": None, "available": False}


def configure_fonts(font: str = DEFAULT_FONT, log: Logger | None = None) -> bool:
    """Makes `font` the default matplotlib sans-serif if it is installed, otherwise keeps matplotlib's default.

    The availability scan runs once per family and is cached, so repeated calls (e.g. from the analysis theme for
    every figure) are cheap. When the font is missing, `rcParams` is left untouched and a single warning is logged,
    so figures degrade gracefully to matplotlib's DejaVu Sans fallback.

    Args:
        font (str): Family name to look up and apply, as registered with matplotlib's font manager.
        log (Logger | None): Logger for the "font not found" warning; defaults to this module's logger.

    Returns:
        bool: True when the font was found and applied, False when it was missing and defaults were kept.
    """
    log = log or _LOGGER
    if _font_state["font"] != font:
        _font_state["font"] = font
        _font_state["available"] = font in {f.name for f in font_manager.fontManager.ttflist}
        if not _font_state["available"]:
            log.warning(
                "Font %r not found among installed system fonts; matplotlib will use its default sans-serif "
                "(DejaVu Sans). Install DM Sans system-wide (or to ~/.local/share/fonts) and clear the "
                "matplotlib cache (rm -rf ~/.cache/matplotlib) to enable it.",
                font,
            )

    if not _font_state["available"]:
        return False

    # Prepend so `font` wins while keeping the existing sans-serif fallback chain.
    existing = [f for f in mpl.rcParams["font.sans-serif"] if f != font]
    mpl.rcParams["font.sans-serif"] = [font, *existing]
    mpl.rcParams["font.family"] = "sans-serif"
    return True
