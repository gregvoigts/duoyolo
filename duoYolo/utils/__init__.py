"""Utility package initialization and default config patching for DuoYOLO."""

from pyparsing import Path
import ultralytics.utils as ul_utils
from ultralytics.utils import IterableSimpleNamespace, YAML


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"

# Merge our local default.yaml into ultralytics' DEFAULT_CFG_DICT
# This patches the `ultralytics.utils` module in-place so subsequent
# imports of `ultralytics.utils.DEFAULT_CFG` (when imported after this
# module runs) will see the extended configuration.
ul_utils.DEFAULT_CFG_DICT.update(YAML.load(DEFAULT_CFG_PATH))
ul_utils.DEFAULT_CFG_KEYS = ul_utils.DEFAULT_CFG_DICT.keys()
ul_utils.DEFAULT_CFG = IterableSimpleNamespace(**ul_utils.DEFAULT_CFG_DICT)

# Re-export the same names from this package for convenience
DEFAULT_CFG_DICT = ul_utils.DEFAULT_CFG_DICT
DEFAULT_CFG_KEYS = ul_utils.DEFAULT_CFG_KEYS
DEFAULT_CFG = ul_utils.DEFAULT_CFG
