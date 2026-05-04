# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from pathlib import Path

import pytest

from duoYolo import DuoYOLO
from duoYolo.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks

# Constants used in tests
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo11n.pt"  # test spaces in path
CFG = "yolo11n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()
TASK_MODEL_DATA = [(task, (WEIGHTS_DIR / TASK2MODEL[task] if TASK2MODEL[task].endswith(".pt") else Path(TASK2MODEL[task])), TASK2DATA[task], str(TASK2DATA[task])) for task in TASKS]
MODELS = frozenset([*list([t.replace(".yaml", ".pt") for t in TASK2MODEL.values()]), "yolo11n-grayscale.pt"])
TASK_MODEL_DATA.append(("multitask", WEIGHTS_DIR / "duoyolo11n-od-seg-cls.pt", {"task_0": "MultiRail8-signal.yaml", "task_1": "MultiRail8-track.yaml", "task_2": "MultiRail8-weather.yaml"}, "MultiRail8.yaml"))
TASK_MODEL_DATA.append(("classify", WEIGHTS_DIR / "yolo11n-cls.pt", "MultiRail8-weather.yaml", "MultiRail8-weather.yaml"))

__all__ = (
    "CFG",
    "CUDA_DEVICE_COUNT",
    "CUDA_IS_AVAILABLE",
    "MODEL",
    "SOURCE",
    "SOURCES_LIST",
)

@pytest.fixture(scope="session", autouse=True)
def create_multitask_pt_files():
    """Create dummy .pt files for all tasks to ensure they are available for tests."""
    for i in range(len(TASK_MODEL_DATA)):
        task, model_path, data = TASK_MODEL_DATA[i]
        if model_path.suffix == ".yaml":
            pt_file = WEIGHTS_DIR / model_path.stem + ".pt"
            if not pt_file.exists():
                DuoYOLO(model_path).save(pt_file)
            TASK_MODEL_DATA[i] = (task, pt_file, data)
            