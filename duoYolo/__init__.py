"""Top-level DuoYOLO package exports and lazy model access."""

__version__ = "1.0.0"

import importlib
from typing import TYPE_CHECKING

from ultralytics import ASSETS, settings, checks, download, MODELS

MODELS = (*MODELS, "DuoYOLO")

__all__ = (
    "__version__",
    "ASSETS",
    *MODELS,
    "checks",
    "download",
    "settings",
)

if TYPE_CHECKING:
    # Enable hints for type checkers
    from ultralytics.models import YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR  # noqa
    from duoYolo.models import DuoYOLO  # noqa

def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in MODELS:
        if name == "DuoYOLO":
            return getattr(importlib.import_module("duoYolo.models"), name)
        return getattr(importlib.import_module("ultralytics.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS))

if __name__ == "__main__":
    print(__version__)